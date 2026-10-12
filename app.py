"""
Toposheet Viewer and Map Exporter - single file, Pillow-only export (no matplotlib).
Run:  streamlit run app.py
Needs: streamlit, requests, geopandas, shapely, pyproj, pillow>=10.1, pmtiles
The export runs in a separate worker process ("--export <dir>") so a crash/OOM never kills the app.
"""
import os, re, io, sys, gc, json, math, time, uuid, shutil, zipfile, tempfile, textwrap, threading, subprocess
import faulthandler
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # Fix "Cannot find proj.db" (conda PROJ conflict)
    from pyproj import datadir
    os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"] = datadir.get_data_dir()
except Exception:
    pass

import requests
import geopandas as gpd
from shapely.geometry import box
from PIL import Image, ImageDraw, ImageFont
from pmtiles.reader import Reader

Image.MAX_IMAGE_PIXELS = None  # 800 DPI pages exceed Pillow's default bomb limit

GITHUB_USER, GITHUB_REPO, RELEASE_TAG = "Himanshu03111999", "toposheet-storage", "v1.0.0"
PROXY_PORT, TILE_WORKERS, JOB_TTL = 8765, 8, 1800
API = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}"
R = 6378137.0
BUF10_COLOR = "#b30000"
DPI_OPTIONS = [300, 500, 800]

TOKEN = os.getenv("GITHUB_TOKEN")
session = requests.Session()  # pooled connections
if TOKEN:
    session.headers.update({"Authorization": f"Bearer {TOKEN}"})


# =====================================================================================
# SHARED: GitHub range fetch
# =====================================================================================
@lru_cache(maxsize=256)
def fetch_range(asset_id, rng=None):
    h = {"Accept": "application/octet-stream"}
    if rng:
        h["Range"] = rng
    last = None
    for attempt in range(3):
        try:
            r = session.get(f"{API}/releases/assets/{asset_id}", headers=h, timeout=30)
            if r.status_code >= 500 or r.status_code in (401, 403, 404, 429):
                raise RuntimeError(f"HTTP {r.status_code}")
            return r.status_code, r.headers.get("Content-Range"), r.content
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise last


_BOUNDS = {}


def sheet_bounds(asset_id):
    """(min_lon, min_lat, max_lon, max_lat) of a PMTiles sheet; failures are not cached."""
    if asset_id not in _BOUNDS:
        try:
            h = Reader(lambda o, l: fetch_range(asset_id, f"bytes={o}-{o + l - 1}")[2]).header()
            _BOUNDS[asset_id] = tuple(h[k] / 1e7 for k in ("min_lon_e7", "min_lat_e7", "max_lon_e7", "max_lat_e7"))
        except Exception:
            return None
    return _BOUNDS[asset_id]


# =====================================================================================
# EXPORT WORKER
# =====================================================================================
def mosaic(asset_id, bounds, max_tiles, on_tile=None):
    """Stitch PMTiles covering `bounds` (lon/lat) into one RGBA canvas -> (canvas, [xmin,xmax,ymin,ymax] in 3857)."""
    rd = Reader(lambda o, l: fetch_range(asset_id, f"bytes={o}-{o + l - 1}")[2])
    h = rd.header()
    z = h["max_zoom"]
    sb = sheet_bounds(asset_id)
    if sb:
        bounds = [max(bounds[0], sb[0]), max(bounds[1], sb[1]), min(bounds[2], sb[2]), min(bounds[3], sb[3])]
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            return None

    def rng(zl):
        n = 2 ** zl
        tx = lambda lon: int((lon + 180) / 360 * n)
        ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
        x0, x1 = sorted((tx(bounds[0]), tx(bounds[2])))
        y0, y1 = sorted((ty(bounds[1]), ty(bounds[3])))
        return n, max(x0, 0), min(x1, n - 1), max(y0, 0), min(y1, n - 1)

    n, x0, x1, y0, y1 = rng(z)
    while z > h["min_zoom"] and (x1 - x0 + 1 > max_tiles or y1 - y0 + 1 > max_tiles):
        z -= 1
        n, x0, x1, y0, y1 = rng(z)

    canvas = Image.new("RGBA", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256), (0, 0, 0, 0))
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def load(xy):
        data = rd.get(z, *xy)
        if not data:
            return xy, None
        t = Image.open(io.BytesIO(data)).convert("RGBA")
        return xy, (t if t.size == (256, 256) else t.resize((256, 256), Image.Resampling.BILINEAR))

    with ThreadPoolExecutor(max_workers=TILE_WORKERS) as ex:
        for i, (xy, t) in enumerate(ex.map(load, tiles), 1):
            if t is not None:
                canvas.alpha_composite(t, ((xy[0] - x0) * 256, (xy[1] - y0) * 256))
                del t
            if on_tile:
                on_tile(i, len(tiles))

    ts = 2 * math.pi * R / n
    return canvas, [-math.pi * R + x0 * ts, -math.pi * R + (x1 + 1) * ts,
                    math.pi * R - (y1 + 1) * ts, math.pi * R - y0 * ts]


# ---------------- drawing helpers ----------------
def merc(lon, lat):
    return R * math.radians(lon), R * math.asinh(math.tan(math.radians(lat)))


def unmerc(x, y):
    return math.degrees(x / R), math.degrees(math.atan(math.sinh(y / R)))


def get_font(size, bold=False):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def put_text(d, xy, s, font, anchor="la"):
    try:
        d.text(xy, s, font=font, fill="black", anchor=anchor)
    except ValueError:  # bitmap fallback font has no anchors
        d.text(xy, s, font=font, fill="black")


def fit_text(d, text, w, h, size, bold=False):
    """Largest font (<= size) whose wrapped text fills the width w and still fits height h."""
    while size > 6:
        f = get_font(size, bold)
        t = "\n".join(textwrap.wrap(text, max(8, int(w / (size * 0.52)))))
        bb = d.multiline_textbbox((0, 0), t, font=f, spacing=size * .2, align="center")
        if bb[2] <= w and bb[3] <= h:
            return f, t, size * .2
        size -= 1
    return get_font(6), text, 1


def nice_step(raw):
    return next((s for s in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if s >= raw), 10)


def geom_paths(g):
    t = g.geom_type
    if t == "Polygon":
        yield [c[:2] for c in g.exterior.coords]
        for r in g.interiors:
            yield [c[:2] for c in r.coords]
    elif t in ("LineString", "LinearRing"):
        yield [c[:2] for c in g.coords]
    elif t == "Point":
        yield [(g.x, g.y)]
    elif hasattr(g, "geoms"):
        for sub in g.geoms:
            yield from geom_paths(sub)


def draw_path(d, pts, color, width, dash=None):
    """Polyline with optional (on, off) dash pattern in pixels."""
    if len(pts) == 1:
        x, y = pts[0]
        d.ellipse([x - width * 2, y - width * 2, x + width * 2, y + width * 2], outline=color, width=width)
        return
    if not dash:
        d.line(pts, fill=color, width=width, joint="curve")
        return
    on, off = dash
    period, pos = on + off, 0.0
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg == 0:
            continue
        ux, uy, t = (x1 - x0) / seg, (y1 - y0) / seg, 0.0
        while t < seg:
            ph = pos % period
            run = min((on if ph < on else period) - ph, seg - t)
            if ph < on:
                d.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * (t + run), y0 + uy * (t + run))], fill=color, width=width)
            t += run
            pos += run


def paste_mosaic(map_img, canvas, ext, ox, oy, scale, mx0, my0):
    """Resample a tile mosaic straight into the final map box (clipped to it)."""
    dx0, dx1 = ox + ext[0] * scale - mx0, ox + ext[1] * scale - mx0
    dy0, dy1 = oy - ext[3] * scale - my0, oy - ext[2] * scale - my0
    if dx1 <= dx0 or dy1 <= dy0:
        return
    tx0, ty0 = int(math.floor(max(dx0, 0))), int(math.floor(max(dy0, 0)))
    tx1, ty1 = int(math.ceil(min(dx1, map_img.width))), int(math.ceil(min(dy1, map_img.height)))
    if tx1 <= tx0 or ty1 <= ty0:
        return
    sw, sh = canvas.size
    fx, fy = sw / (dx1 - dx0), sh / (dy1 - dy0)
    bx = (max(0.0, (tx0 - dx0) * fx), max(0.0, (ty0 - dy0) * fy),
          min(float(sw), (tx1 - dx0) * fx), min(float(sh), (ty1 - dy0) * fy))
    if bx[2] > bx[0] and bx[3] > bx[1]:
        part = canvas.resize((tx1 - tx0, ty1 - ty0), Image.Resampling.BILINEAR, box=bx)
        map_img.paste(part, (tx0, ty0), part)


def export_map(sheets_pm, site, title, sheets_str, dpi, legend_bytes, progress, out_path):
    pt = lambda v: max(1, int(round(v * dpi / 72)))
    W, H = int(16 * dpi), int(11 * dpi)
    mx0, mx1, my0, my1 = int(.05 * W), int(.745 * W), int(.06 * H), int(.93 * H)
    px0, px1 = int(.757 * W), int(.99 * W)
    mw, mh, pw, ph = mx1 - mx0, my1 - my0, px1 - px0, my1 - my0

    # ---- view extent (fit to map box; the box is always fully covered by the requested bbox) ----
    progress(0.01, "Preparing extent...")
    bufs = s = None
    if site is not None:
        s = site.to_crs(3857)
        bufs = {km: s.buffer(km * 1000) for km in (10, 5, 2)}
        b = bufs[10].total_bounds
        dx, dy = (b[2] - b[0]) * .15, (b[3] - b[1]) * .15
        vx0, vx1, vy0, vy1 = b[0] - dx, b[2] + dx, b[1] - dy, b[3] + dy
    else:
        with ThreadPoolExecutor(max_workers=TILE_WORKERS) as ex:
            sbs = [b for b in ex.map(sheet_bounds, sheets_pm) if b]
        bb = ([min(b[0] for b in sbs), min(b[1] for b in sbs), max(b[2] for b in sbs), max(b[3] for b in sbs)]
              if sbs else [68, 6, 98, 36])
        (vx0, vy0), (vx1, vy1) = merc(bb[0], bb[1]), merc(bb[2], bb[3])

    scale = min(mw / (vx1 - vx0), mh / (vy1 - vy0))  # px = ox + X*scale ; py = oy - Y*scale
    cx, cy = (vx0 + vx1) / 2, (vy0 + vy1) / 2
    ox, oy = mx0 + mw / 2 - cx * scale, my0 + mh / 2 + cy * scale
    (lon0, lat0) = unmerc((mx0 - ox) / scale, (oy - my1) / scale)
    (lon1, lat1) = unmerc((mx1 - ox) / scale, (oy - my0) / scale)
    bbox = [lon0, lat0, lon1, lat1]

    def overlaps(aid):
        sb = sheet_bounds(aid)
        return sb is None or not (sb[2] <= bbox[0] or sb[0] >= bbox[2] or sb[3] <= bbox[1] or sb[1] >= bbox[3])

    sheets_pm = [a for a in sheets_pm if overlaps(a)]
    n = max(len(sheets_pm), 1)
    max_tiles = min(16, math.ceil(max(mw, mh) / 256))  # ~1 source tile per 256 output px, capped for memory/time

    # ---- tiles ----
    map_img = Image.new("RGB", (mw, mh), "white")
    for i, aid in enumerate(sheets_pm):
        res = mosaic(aid, bbox, max_tiles,
                     lambda dn, t, idx=i: progress((idx + dn / t) / n * 0.75,
                                                   f"Fetching tiles: sheet {idx + 1}/{n} ({dn}/{t})"))
        if res:
            paste_mosaic(map_img, res[0], res[1], ox, oy, scale, mx0, my0)
            del res
            gc.collect()

    # ---- overlay: graticule, site, buffers ----
    progress(0.8, "Drawing map overlay and legend...")
    md = ImageDraw.Draw(map_img)
    tp = lambda x, y: (ox + x * scale - mx0, oy - y * scale - my0)
    gdash, gcol = (pt(1.5), pt(3)), (110, 110, 110)
    xlabels, ylabels = [], []
    for vertical, a, b in ((True, lon0, lon1), (False, lat0, lat1)):
        stp = nice_step(max(b - a, 1e-6) / 5)
        dec = max(0, -int(math.floor(math.log10(stp))))
        for k in range(math.ceil(a / stp), math.floor(b / stp) + 1):
            v = k * stp
            if vertical:
                p = tp(*merc(v, lat0))[0]
                draw_path(md, [(p, 0), (p, mh)], gcol, 1, gdash)
                xlabels.append((p + mx0, f"{v:.{dec}f}°"))
            else:
                p = tp(*merc(lon0, v))[1]
                draw_path(md, [(0, p), (mw, p)], gcol, 1, gdash)
                ylabels.append((p + my0, f"{v:.{dec}f}°"))

    if site is not None:
        for geoms, c, dash, lw in ((bufs[10], BUF10_COLOR, None, 2.0), (bufs[5], "#000", (pt(6), pt(4)), 1.5),
                                   (bufs[2], "#000", (pt(1.5), pt(3)), 1.8), (s.geometry, "#000", None, 2.0)):
            for g in geoms:
                for path in geom_paths(g):
                    draw_path(md, [tp(x, y) for x, y in path], c, pt(lw), dash)

    # ---- compose page ----
    img = Image.new("RGB", (W, H), "white")
    img.paste(map_img, (mx0, my0))
    del map_img
    d = ImageDraw.Draw(img)
    d.rectangle([mx0 - 1, my0 - 1, mx1, my1], outline="black", width=pt(1))
    f_tick = get_font(pt(9))
    for x, lab in xlabels:
        put_text(d, (x, my1 + pt(4)), lab, f_tick, "mt")
    for y, lab in ylabels:
        put_text(d, (mx0 - pt(4), y), lab, f_tick, "rm")
    put_text(d, ((mx0 + mx1) / 2, H * .03), "Topographic MAP", get_font(pt(22), True), "mm")

    # ---- side panel ----
    PX = lambda u: px0 + u * pw
    PY = lambda v: my1 - v * ph
    d.rectangle([px0, my0, px1, my1], outline="black", width=pt(1.5))

    # true-scale bar (1 real km = 1000 / cos(lat) Mercator metres)
    px_per_km = 1000 * scale / math.cos(math.radians(unmerc(cx, cy)[1]))
    km = max([0.1] + [c for c in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if c * px_per_km <= pw * .45])
    bl, bx, by = km * px_per_km, PX(.08), PY(.945)
    d.line([(bx, by), (bx + bl, by)], fill="black", width=pt(1.5))
    f_sb = get_font(pt(8.5), True)
    for frac, lab in ((0, "0"), (.5, f"{km / 2:g}"), (1, f"{km:g} km")):
        d.line([(bx + bl * frac, by), (bx + bl * frac, by - pt(5))], fill="black", width=pt(1.5))
        put_text(d, (bx + bl * frac, by - pt(7)), lab, f_sb, "ms")

    put_text(d, (PX(.8), PY(.965)), "N", get_font(pt(18), True), "mm")  # north arrow
    ax_, ay_, hh, ww = PX(.8), PY(.925), pt(9), pt(7)
    d.polygon([(ax_, ay_ - hh), (ax_ - ww, ay_ + hh), (ax_ + ww, ay_ + hh)], fill="black")

    for v in (.905, .74, .545):
        d.line([(px0, PY(v)), (px1, PY(v))], fill="black", width=pt(1))
    put_text(d, (PX(.5), PY(.88)), "Title :", get_font(pt(11.5), True), "mm")

    # title: font + wrapping chosen so the text fills the panel width
    f_t, t_txt, sp = fit_text(d, title.replace("\n", " "), pw * .94, (PY(.74) - PY(.905)) * .9, pt(15))
    try:
        d.multiline_text((PX(.5), PY(.8225)), t_txt, font=f_t, fill="black", anchor="mm", align="center", spacing=sp)
    except ValueError:
        d.multiline_text((px0 + pw * .03, PY(.895)), t_txt, font=f_t, fill="black", align="center", spacing=sp)

    put_text(d, (PX(.5), PY(.715)), "Legends", get_font(pt(10.5), True), "mm")
    f_leg = get_font(pt(8.5))
    for y, lab, dash, lw, c in ((.672, "Project Site", None, 1.8, "black"),
                                (.637, "Core Zone : 2 km", (pt(1.5), pt(3)), 1.8, "black"),
                                (.602, "Buffer Zone 5 km", (pt(6), pt(4)), 1.5, "black"),
                                (.567, "Buffer Zone 10 km", None, 2.0, BUF10_COLOR)):
        x0_, x1_, y0_, y1_ = PX(.08), PX(.20), PY(y + .022), PY(y)
        draw_path(d, [(x0_, y0_), (x1_, y0_), (x1_, y1_), (x0_, y1_), (x0_, y0_)], c, pt(lw), dash)
        put_text(d, (PX(.25), PY(y + .011)), lab, f_leg, "lm")

    if legend_bytes:  # conventional symbols: stretched to fill the box
        try:
            li = Image.open(io.BytesIO(legend_bytes)).convert("RGBA")
            x0_, y0_, x1_, y1_ = int(PX(.005)), int(PY(.54)), int(PX(.995)), int(PY(.005))
            li = li.resize((x1_ - x0_, y1_ - y0_), Image.Resampling.LANCZOS)
            img.paste(li, (x0_, y0_), li)
            del li
        except Exception:
            pass

    f_ft = get_font(pt(12), True)
    put_text(d, (.15 * W, .975 * H), "Source : Survey of India", f_ft, "lm")
    put_text(d, (.42 * W, .975 * H), f"Topo sheet No. {sheets_str}", f_ft, "lm")

    progress(0.9, f"Encoding PNG at {dpi} DPI...")
    tmp = out_path + ".tmp"
    img.save(tmp, format="PNG", compress_level=3, dpi=(dpi, dpi))
    del img
    gc.collect()
    os.replace(tmp, out_path)


def worker_main(job_dir):
    def progress(frac, msg):
        p = os.path.join(job_dir, "progress.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"progress": frac, "msg": msg}, f)
        os.replace(p + ".tmp", p)

    with open(os.path.join(job_dir, "input.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    site = None
    if os.path.exists(os.path.join(job_dir, "site.geojson")):
        site = gpd.read_file(os.path.join(job_dir, "site.geojson"))
        site = site.set_crs(4326) if site.crs is None else site.to_crs(4326)
    legend = None
    if os.path.exists(os.path.join(job_dir, "legend.png")):
        with open(os.path.join(job_dir, "legend.png"), "rb") as f:
            legend = f.read()
    progress(0.0, "Starting export process...")
    export_map(cfg["sheets_pm"], site, cfg["title"], cfg["sheets_str"], cfg["dpi"], legend,
               progress, os.path.join(job_dir, "result.png"))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--export":
    faulthandler.enable()  # native crash tracebacks go to log.txt
    worker_main(sys.argv[2])
    sys.exit(0)


# =====================================================================================
# STREAMLIT APP
# =====================================================================================
import streamlit as st
import streamlit.components.v1 as components

if not TOKEN:
    try:
        TOKEN = st.secrets.get("GITHUB_TOKEN")
    except Exception:
        TOKEN = None
    if TOKEN:
        session.headers.update({"Authorization": f"Bearer {TOKEN}"})

st.set_page_config(page_title="Toposheet Viewer and Map Exporter", layout="wide")
st.markdown("""
<style>
header[data-testid="stHeader"]{display:none!important}
.block-container{padding:2.5rem 1.5rem .5rem!important}
.big-main-title{font-size:38px;font-weight:800;line-height:1.25;margin-bottom:.6rem}
html,body,[class*="css"],.stMarkdown,p,span,label{font-size:12px!important}
.stApp,[data-testid="stAppViewContainer"]{opacity:1!important;transition:none!important}
[data-testid="stStatusWidget"]{display:none!important}
.stButton>button{font-size:12px!important;font-weight:600!important}
hr{margin:.4rem 0!important}
.custom-footer{text-align:center;color:#888;font-size:10px;padding:4px 0;border-top:1px solid #e0e0e0;margin-top:10px}
.custom-footer a{color:#06c;text-decoration:none}
.lg{display:flex;align-items:center;gap:10px;margin:7px 0}
.lg i{display:inline-block;width:34px;height:14px;box-sizing:border-box;flex:none}
</style>
<div class="big-main-title">🌐 Toposheet Viewer and Map Exporter</div>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def release_assets():
    r = session.get(f"{API}/releases/tags/{RELEASE_TAG}",
                    headers={"Accept": "application/vnd.github+json"}, timeout=30)
    r.raise_for_status()
    return {a["name"]: str(a["id"]) for a in r.json()["assets"]}


@st.cache_data(show_spinner=False)
def cached_bounds(aid):  # persists across Streamlit reruns; failures raise so they are never cached
    b = sheet_bounds(aid)
    if b is None:
        raise RuntimeError("no bounds")
    return b


class Proxy(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        try:
            code, cr, body = fetch_range(self.path.strip("/").split("/")[-1], self.headers.get("Range"))
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            if cr:
                self.send_header("Content-Range", cr)
            self._cors(); self.end_headers(); self.wfile.write(body)
        except Exception:
            try:
                self.send_response(500); self._cors(); self.end_headers()
            except Exception:
                pass

    def log_message(self, *a):
        pass


@st.cache_resource
def start_proxy():
    srv = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Proxy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


start_proxy()


@st.cache_resource
def job_store():
    return {}


JOBS = job_store()


def legend_image(assets):
    key = next((k for k in assets if "legend_toposheet" in k.lower() or "legend_topsheet" in k.lower()), None)
    if key:
        try:
            code, _, body = fetch_range(assets[key], None)
            if code == 200:
                return body
        except Exception:
            pass
    for path in ["legend_toposheet.png", "assets/legend_toposheet.png"]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    return None


@st.cache_data
def load_osm_index():
    paths = ["OSM_SHEET_INDEX.shp", "assets/OSM_SHEET_INDEX.shp", "assets/OSM_SHEET_INDEX/OSM_SHEET_INDEX.shp"]
    target = next((p for p in paths if os.path.exists(p)), None)
    if target:
        try:
            gdf = gpd.read_file(target)
            return gdf.set_crs(4326) if gdf.crs is None else gdf.to_crs(4326)
        except Exception as e:
            st.sidebar.warning(f"Could not load OSM_SHEET_INDEX.shp: {e}")
    return None


def load_vector(f):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, os.path.basename(f.name))
        with open(p, "wb") as fh:
            fh.write(f.getbuffer())
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                zf.extractall(d)
            p = next((os.path.join(r, x) for r, _, fs in os.walk(d) for x in fs if x.lower().endswith(".shp")), None)
            if not p:
                return None
        g = gpd.read_file(p)
    return g.set_crs(4326) if g.crs is None else g.to_crs(4326)


def sheet_no(name):
    s = re.findall(r"\d{1,3}[A-P]/?\d{1,2}", name, re.I)
    if not s:
        return None
    s = s[0].upper()
    return s if "/" in s else re.sub(r"([A-P])(\d+)", r"\1/\2", s)


def describe_exit(rc):
    """Plain-text meaning of a worker exit code (Windows NTSTATUS or POSIX signal)."""
    if rc < 0:
        return f"killed by signal {-rc}" + (" (SIGKILL - usually out of memory)" if rc == -9 else "")
    code = rc & 0xFFFFFFFF
    known = {0xC0000005: "access violation (native crash)", 0xC0000017: "not enough memory",
             0xC0000135: "DLL not found", 0xC00000FD: "stack overflow",
             0xC0000409: "native stack buffer overrun",
             0xC06D007F: "DLL function not found (library version conflict)"}
    return f"{known[code]} [0x{code:08X}]" if code in known else f"exit code {rc}"


def start_export(sheets_pm, site, title, sheets_str, dpi, legend_bytes):
    d = tempfile.mkdtemp(prefix="topo_export_")
    with open(os.path.join(d, "input.json"), "w", encoding="utf-8") as f:
        json.dump({"sheets_pm": sheets_pm, "title": title, "sheets_str": sheets_str, "dpi": dpi}, f)
    if site is not None:
        with open(os.path.join(d, "site.geojson"), "w", encoding="utf-8") as f:
            f.write(site.to_json())
    if legend_bytes:
        with open(os.path.join(d, "legend.png"), "wb") as f:
            f.write(legend_bytes)
    env = dict(os.environ, **({"GITHUB_TOKEN": TOKEN} if TOKEN else {}))
    with open(os.path.join(d, "log.txt"), "wb") as log:
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--export", d],
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"status": "running", "progress": 0.0, "msg": "Starting export process...", "start": time.time(),
            "end": None, "result": None, "error": None, "dpi": dpi, "dir": d, "proc": proc}


def refresh_job(job):
    if not job or job["status"] != "running":
        return
    d = job["dir"]
    try:
        with open(os.path.join(d, "progress.json"), encoding="utf-8") as f:
            p = json.load(f)
        job["progress"], job["msg"] = p["progress"], p["msg"]
    except Exception:
        pass
    rc = job["proc"].poll()
    if rc is None:
        return
    job["end"] = time.time()
    res = os.path.join(d, "result.png")
    if rc == 0 and os.path.exists(res):
        with open(res, "rb") as f:
            job["result"] = f.read()
        job.update(status="done", progress=1.0, msg="Done")
    else:
        try:
            with open(os.path.join(d, "log.txt"), encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-12:]).strip()
        except Exception:
            tail = ""
        job.update(status="error", error=f"{describe_exit(rc)} | last step: {job.get('msg')}" + (f"\n{tail}" if tail else ""))
    shutil.rmtree(d, ignore_errors=True)


def cancel_job(job):
    try:
        job["proc"].terminate()
        job["proc"].wait(timeout=5)
    except Exception:
        pass
    job.update(status="cancelled", end=time.time())
    shutil.rmtree(job["dir"], ignore_errors=True)


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Data Inputs")
assets = {}
try:
    assets = release_assets()
except Exception as e:
    st.sidebar.error(f"GitHub error: {e}")
pm_assets = {n: i for n, i in assets.items() if n.endswith(".pmtiles")}

basemap = st.sidebar.selectbox("🗺️ Choose Base Map",
                               ["OpenStreetMap", "Google Satellite", "Google Hybrid", "Google Roadmap", "Esri Satellite"])
show_topo = st.sidebar.toggle("🗾 Show Topo Layer", value=True)
show_osm_index = st.sidebar.toggle("📐 Show OSM Sheet Index", value=False)
chosen = st.sidebar.multiselect("Toposheets (PMTiles)", list(pm_assets))
vec = st.sidebar.file_uploader("Upload Vector (.geojson, .kml, .zip)", type=["geojson", "kml", "gpx", "zip"])

site = None
if vec:
    try:
        site = load_vector(vec)
        st.sidebar.caption("✅ Vectors Loaded" if site is not None else "")
        if site is None:
            st.sidebar.error("No .shp found inside the zip.")
    except Exception as e:
        st.sidebar.error(f"Vector error: {e}")

osm_gdf = load_osm_index() if show_osm_index else None

# Sheet numbers: all selected sheets, or - once a vector is loaded - only those touching the 10 km buffer
m3857 = site.to_crs(3857) if site is not None else None
buf_ll = {km: m3857.buffer(km * 1000).to_crs(4326) for km in (10, 5, 2)} if site is not None else None


def touches_buffer(name):
    try:
        bx = box(*cached_bounds(pm_assets[name]))
    except Exception:
        return True
    return any(g.intersects(bx) for g in buf_ll[10])


names = [n for n in chosen if site is None or touches_buffer(n)]
sheets = [s for s in dict.fromkeys(sheet_no(n) for n in names) if s]
sheets_str = ", ".join(sheets) or ("None within 10 km buffer" if chosen else "56F/12, 56F/8, 56G/5, 56G/9")

cfg = {
    "basemap": basemap, "site": None, "buffers": [], "buf10_color": BUF10_COLOR,
    "pm": [{"name": n, "url": f"http://127.0.0.1:{PROXY_PORT}/{pm_assets[n]}"} for n in chosen] if show_topo else [],
    "osm_index": json.loads(osm_gdf.to_json()) if osm_gdf is not None else None,
}
if site is not None:
    cfg["site"] = json.loads(site.to_json())
    cfg["buffers"] = [{"km": km, "gj": json.loads(buf_ll[km].to_json())} for km in (10, 5, 2)]

# ---------------- Page ----------------
project_title = st.text_input(
    "Project Title",
    value="Preliminary Assessment Study for the 14MW Solar Project at Gadgi, Bidar District, Karnataka, India")
st.markdown("---")

HTML = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@3.0.7/dist/pmtiles.js"></script>
<div id="m" style="height:480px;border:1px solid #ccc"></div>
<script>
const C = __CFG__;
const BASE = {
 "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
 "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
 "Google Hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
 "Google Roadmap": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
 "Esri Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
};

const map = L.map('m', {preferCanvas: true, zoomSnap: 0.5}).setView([20, 78], 5);
L.control.scale().addTo(map);
L.tileLayer(BASE[C.basemap], {maxZoom: 20, attribution: C.basemap}).addTo(map);

map.createPane('topo'); map.getPane('topo').style.zIndex = 300;
map.createPane('osm_pane'); map.getPane('osm_pane').style.zIndex = 400;

let siteBounds = null;
if (C.site) {
  const sty = {10: [C.buf10_color, 2.5, null], 5: ['#333', 2.5, '4,4'], 2: ['#000', 2.5, '2,4']};
  C.buffers.forEach(b => {
    const g = L.geoJSON(b.gj, {style: {color: sty[b.km][0], weight: sty[b.km][1], dashArray: sty[b.km][2], fill: false}}).addTo(map);
    if (b.km == 10) siteBounds = g.getBounds().pad(0.15);
  });
  L.geoJSON(C.site, {style: {color: '#000', weight: 3, fill: false}}).addTo(map);
}

(async () => {
  let topoBounds = null;
  let capZoom = 20;
  for (const p of C.pm) {
    try {
      const src = new pmtiles.PMTiles(p.url);
      const h = await src.getHeader();
      pmtiles.leafletRasterLayer(src, {
        pane: 'topo', minZoom: h.minZoom, maxZoom: h.maxZoom, opacity: 0.85, keepBuffer: 4
      }).addTo(map);
      capZoom = Math.min(capZoom, h.maxZoom);
      const b = L.latLngBounds([[h.minLat, h.minLon], [h.maxLat, h.maxLon]]);
      topoBounds = topoBounds ? topoBounds.extend(b) : b;
    } catch (e) { console.error("PMTiles error", p.name, e); }
  }
  if (siteBounds) map.fitBounds(siteBounds, {maxZoom: capZoom});
  else if (topoBounds && topoBounds.isValid()) map.fitBounds(topoBounds, {maxZoom: capZoom});
})();

if (C.osm_index) {
  L.geoJSON(C.osm_index, {
    pane: 'osm_pane',
    style: () => ({color: '#ff4400', weight: 1, fill: false, fillOpacity: 0}),
    onEachFeature: function(feature, layer) {
      const sheetName = feature.properties ? (feature.properties.EVEREST_SH || 'N/A') : 'N/A';
      layer.bindTooltip(sheetName, {sticky: true, direction: 'auto'});
      layer.on({
        mouseover: e => { e.target.setStyle({color: '#000000', weight: 2.5, fill: false, fillOpacity: 0}); e.target.bringToFront(); },
        mouseout:  e => { e.target.setStyle({color: '#ff4400', weight: 1, fill: false, fillOpacity: 0}); }
      });
    }
  }).addTo(map);
}
</script>
"""

leg_bytes = legend_image(assets)
col_map, col_leg = st.columns([3.2, 1.1])
with col_map:
    components.html(HTML.replace("__CFG__", json.dumps(cfg)), height=490)
    c1, c2 = st.columns([1, 2])
    c1.markdown("**Source : Survey of India**")
    c2.markdown(f"**Topo sheet No. {sheets_str}**")
with col_leg:
    with st.container(height=480, border=True):
        st.markdown("### **Legends**")
        st.markdown("".join(
            f'<div class="lg"><i style="border:{w}px {ls} {c}"></i><b>{lab}</b></div>'
            for lab, ls, w, c in (("Project Site", "solid", 2, "#000"), ("Core Zone 2 km", "dotted", 2, "#000"),
                                  ("Buffer Zone 5 km", "dashed", 2, "#000"),
                                  ("Buffer Zone 10 km", "solid", 3, BUF10_COLOR))), unsafe_allow_html=True)
        if leg_bytes:
            st.image(leg_bytes, caption="CONVENTIONAL SYMBOLS", width="stretch")

st.markdown("---")

for jid, j in list(JOBS.items()):  # drop finished jobs (and their PNG bytes) after JOB_TTL
    if j["status"] != "running" and j.get("end") and time.time() - j["end"] > JOB_TTL:
        JOBS.pop(jid, None)
job = JOBS.get(st.session_state.get("job_id"))
refresh_job(job)


@st.fragment(run_every=1.5 if (job and job["status"] == "running") else None)
def export_ui():
    job = JOBS.get(st.session_state.get("job_id"))
    refresh_job(job)
    running = bool(job) and job["status"] == "running"
    if job and not running and st.session_state.pop("polling", False):
        st.rerun()

    cl, cd, cb = st.columns([1.2, 1.5, 3])
    cl.markdown("### 📥 Export Map")
    dpi = cd.selectbox("Resolution (DPI)", DPI_OPTIONS, index=0, label_visibility="collapsed", disabled=running)

    if running:
        if cb.button("⛔ Cancel export", width="stretch"):
            cancel_job(job)
            st.session_state.pop("polling", None)
            st.rerun()
    elif cb.button("🚀 Generate High-Res Map", width="stretch"):
        if not chosen:
            st.warning("Select at least one toposheet in the sidebar first.")
        else:
            if job:
                JOBS.pop(st.session_state.get("job_id"), None)
            jid = uuid.uuid4().hex
            JOBS[jid] = start_export([pm_assets[n] for n in chosen], None if site is None else site.copy(),
                                     project_title, sheets_str, dpi, leg_bytes)
            st.session_state["job_id"] = jid
            st.session_state["polling"] = True
            st.rerun()

    if job:
        elapsed = int((job["end"] or time.time()) - job["start"])
        if job["status"] == "running":
            st.progress(min(job["progress"], 0.99), text=f"{job['msg']}  |  elapsed {elapsed}s")
            st.caption("You can change basemap / layers while this runs. The export keeps going.")
        elif job["status"] == "done":
            st.success(f"Map ready in {elapsed}s ({job['dpi']} DPI).")
            st.download_button("⬇️ Download Ready High-Res Map (PNG)", job["result"],
                               file_name="Topographic_Map_HighRes.png", mime="image/png", width="stretch")
        elif job["status"] == "cancelled":
            st.warning("Export cancelled.")
        elif job["status"] == "error":
            st.error(f"Export failed: {job['error']}")


export_ui()

st.markdown("""
<div class="custom-footer">Developed by <b>Himanshu Gupta</b> &nbsp;|&nbsp;
📧 <a href="mailto:himanshu.guptta.311@gmail.com">himanshu.guptta.311@gmail.com</a> &nbsp;|&nbsp;
🔗 <a href="https://www.linkedin.com/in/himanshu-gupta-4a7942224" target="_blank">LinkedIn Profile</a></div>
""", unsafe_allow_html=True)