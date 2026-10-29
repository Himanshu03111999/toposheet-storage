"""
Geospatial Site Analysis & Mapping Dashboard - single file.   Run: streamlit run app.py   (login: APP_USER / APP_PASS env or secrets)
Needs: streamlit requests geopandas shapely>=2 pyproj pillow>=10.1 pmtiles numpy contourpy mapbox-vector-tile huggingface_hub hf_xet
       opencv-python-headless pymupdf tifffile (the last few auto-install when needed).
Data: HF dataset himanshuguptta/geotiff-hub (HF_TOKEN), byte-range reads. R2_BASE (env/secret): toposheets from R2 first.
Workers: "--export <dir>" (PNG) and "--georef <dir>" (PDF/GeoTIFF -> PMTiles).
Optional: .streamlit/config.toml -> [server] enableStaticServing = true gives a live tile preview of fresh sheets (else a JPEG preview is used).
"""
import os as _os
LOGIN_USER, LOGIN_PASS = _os.getenv("APP_USER", "Kosher"), _os.getenv("APP_PASS", "Himanshugis")

import os, re, io, sys, gc, json, math, time, uuid, zlib, base64, shutil, zipfile, tempfile, textwrap, \
    subprocess, types, hmac, importlib, faulthandler, threading
from collections import defaultdict
from functools import lru_cache
from urllib.parse import quote
from xml.sax.saxutils import escape
from concurrent.futures import ThreadPoolExecutor

try:
    from pyproj import datadir
    os.environ["PROJ_LIB"] = os.environ["PROJ_DATA"] = datadir.get_data_dir()
except Exception:
    pass

import numpy as np
import requests
from requests.adapters import HTTPAdapter
import geopandas as gpd
import shapely
from shapely import wkb as swkb, make_valid
from shapely.geometry import box, shape, mapping, LineString, Point, GeometryCollection
from shapely.ops import unary_union
from PIL import Image, ImageDraw, ImageFont, ImageFilter, TiffImagePlugin
from pmtiles.reader import Reader

Image.MAX_IMAGE_PIXELS = None

APP_TITLE = "Geospatial Site Analysis & Mapping Dashboard"
HF_REPO = "himanshuguptta/geotiff-hub"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main?recursive=true"
R2_BASE = (os.getenv("R2_BASE") or "").rstrip("/")
TILE_WORKERS, JOB_TTL, LIST_TTL, WATCH_EVERY = 32, 1800, 20, 30
GEO_MIN_ZOOM, GEO_QUALITY, GEO_TTL, GEO_DPI = 6, 90, 3 * 3600, 200
R = 6378137.0
COLORS = ["#000000", "#d00000", "#0057ff"]
DPI_OPTIONS = [300, 500, 800]
LOGO_SCALE, LOGO_H_PT, PREVIEW_PX = 0.6, 100, 3200
STATIC_GEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "geo")
try:
    os.makedirs(STATIC_GEO, exist_ok=True)
except OSError:
    pass

BASEMAPS = {"White": None,
            "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "Google Hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "Google Roadmap": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            "Esri Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"}
SAT_BASE = BASEMAPS["Google Satellite"]
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

SITE_W = .8   # project-site line thickness (points) - used by export legend, export map and live map
TITLE_HEAD_FS, TITLE_FS = 16, 22  # "Title :" label / project name (starting size, auto-shrinks to fit)
LEG_HEAD_FS, LEG_FS = 18, 14       # "Legends" heading / legend row text


DEM_PAL = ["#ffffd4", "#fed98e", "#fe9929", "#d95f0e", "#993404"]
DEM_SRC = "NASA SRTM 30 meter Digital Elevation"
ROAD_SRC, ROAD_COL = "Survey of India", "#8b4513"
WATER_SRC = ("Bhuvan LULC-10k:SIS-DP Phase 2 (2018-23) datasets rectified with current "
             "Google Earth Imagery (2025-26)")
SAT_SRC = "Google Earth Imagery"
DRAIN = {"canal": ("Canal / Drain", "canal_drain", "#00897b"),
         "lake": ("Lake / Pond", "lake_pond", "#29b6f6"),
         "river": ("River / Stream", "river_stream", "#1f4fd8"),
         "reservoir": ("Reservoir / Tank", "reservoir_tank", "#8e44ad")}
tint = lambda c, a=.45: "#" + "".join(f"{int(255 - (255 - int(c[i:i + 2], 16)) * a):02x}" for i in (1, 3, 5))
norm = lambda s: re.sub(r"[\s_\-]+", "", s.lower())
stem_of = lambda k: norm(os.path.splitext(k)[0])
is_vec = lambda k: stem_of(k).startswith("roads") or stem_of(k) in {norm(v[1]) for v in DRAIN.values()}
WM_SVG = ('<svg width="22" height="14" viewBox="-17 -11 34 22" style="flex:none"><path d="M0 -2L0 -11M0 -2L-7.8 2.5M0 -2L7.8 2.5M0 -2L0 10" '
          'stroke="#000" stroke-width="1.8" stroke-linecap="round" fill="none"/><circle cx="0" cy="-2" r="2" fill="#000"/></svg>')
CON_COL = "#707070"
CONTOUR_STEP, LABEL_STEP, CONTOUR_MIN_LEN = 10, 20, 500
KML = "application/vnd.google-earth.kml+xml"
VS_CC, VS_BATCHES, VS_EXCL_M, VS_ELEV_W = 0.87, 3, 50, 0.25
VS_COL = ["#b00020", "#0b4f8a", "#4a148c"]   # dark, saturated colours (stay visible on satellite imagery)
VS_RAS = "#00fbff"                           # visibility raster: dark sky blue
VS_BUF_COL = "#000000"                       # WTG-buffer outline: black
VS_RAS_MAXPX = 1600                          # visibility raster PNG is capped to this many pixels on its longest side
KINDS = {"Toposheet Map": (0, 0, 0, 0, 0), "DEM - Elevation Map": (1, 0, 0, 0, 0), "DEM - Contour Map": (1, 1, 0, 0, 0),
         "Roads & Transportation Map": (0, 0, 1, 0, 0), "Drainage Map": (0, 0, 0, 1, 0), "Satellite Map": (0, 0, 0, 0, 1)}

TOKEN = os.getenv("HF_TOKEN")
session = requests.Session()
if TOKEN:
    session.headers.update({"Authorization": f"Bearer {TOKEN}"})
web = requests.Session()  # other hosts: never sends the HF token
web.headers["User-Agent"] = "Mozilla/5.0 (toposheet-app)"
for _s in (session, web):
    _s.mount("https://", HTTPAdapter(pool_connections=16, pool_maxsize=64))

_S = sys.modules.get("_topo_state")  # shared state survives Streamlit reruns
if _S is None:
    _S = sys.modules["_topo_state"] = types.SimpleNamespace()
for _k in ("sig", "bounds", "r2"):
    if not hasattr(_S, _k):
        setattr(_S, _k, {})


def ensure(mod, pkg):
    try:
        return importlib.import_module(mod)
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkg.split()], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"Could not install {pkg} - run: pip install -U {pkg}\n" + r.stderr[-300:])
        importlib.invalidate_caches()
        return importlib.import_module(mod)


def list_assets():
    out, sig, url = {}, {}, HF_TREE
    while url:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        for f in r.json():
            if f.get("type") == "file":
                out[os.path.basename(f["path"])] = f["path"]
                sig[f["path"]] = f"{(f.get('lfs') or {}).get('oid') or f.get('oid')}:{f.get('size')}"
        url = r.links.get("next", {}).get("url")
    _S.sig = sig
    return out


def fetch_range(path, rng=None):
    h = {"Accept-Encoding": "identity", **({"Range": rng} if rng else {})}
    if R2_BASE and path.lower().endswith(".pmtiles") and _S.r2.get(path, True):
        try:
            r = web.get(f"{R2_BASE}/{quote(os.path.basename(path))}", headers=h, timeout=30)
            if r.status_code in (200, 206):
                return r.status_code, r.headers.get("Content-Range"), r.content
            if r.status_code in (403, 404):
                _S.r2[path] = False
        except Exception:
            pass
    for a in range(3):
        try:
            r = session.get(f"{HF_BASE}/resolve/main/{quote(path)}", headers=h, timeout=30)
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After", 2)), 10))
                continue
            r.raise_for_status()
            return r.status_code, r.headers.get("Content-Range"), r.content
        except Exception:
            if a == 2:
                raise
            time.sleep(.5 * (a + 1))
    raise RuntimeError(f"Hugging Face rate limit while reading {path}")


@lru_cache(maxsize=512)
def _wb(url):
    for a in range(3):
        try:
            r = web.get(url, timeout=20)
            if r.status_code == 200:
                return r.content
            if r.status_code in (400, 403, 404):
                return None
        except Exception:
            pass
        time.sleep(.5 * (a + 1))
    raise OSError(url)


def web_img(url):
    try:
        b = _wb(url)
        return Image.open(io.BytesIO(b)) if b else None
    except Exception:
        return None


def reader(aid):
    if os.path.isfile(aid):
        def gl(o, l):
            with open(aid, "rb") as f:
                f.seek(o)
                return f.read(l)
        return Reader(gl)
    memo_, locks, lk = {}, {}, threading.Lock()

    def gb(o, l):
        k = (o, l)
        if k not in memo_:
            with lk:
                kl = locks.setdefault(k, threading.Lock())
            with kl:
                if k not in memo_:
                    memo_[k] = fetch_range(aid, f"bytes={o}-{o + l - 1}")[2]
        return memo_[k]
    return Reader(gb)


def sheet_bounds(aid):
    k = (aid, _S.sig.get(aid))
    if k not in _S.bounds:
        try:
            h = reader(aid).header()
            _S.bounds[k] = tuple(h[c] / 1e7 for c in ("min_lon_e7", "min_lat_e7", "max_lon_e7", "max_lat_e7"))
        except Exception:
            return None
    return _S.bounds[k]


# ---------------- vectors + buffers ----------------
def parse_buffers(txt):
    v = [float(n) * (1000 if u == "km" else 1)
         for n, u in re.findall(r"(\d+(?:\.\d+)?)\s*(km|m)(?:eters?)?(?![a-z])", txt.lower())]
    return sorted({m for m in v if 0 < m <= 20000})


fmt_m = lambda m: f"{m / 1000:g}km" if m >= 1000 else f"{m:g}m"
buf_color = lambda i, n, cols=COLORS: cols[0 if i == 0 else 2 if i == n - 1 else 1]


def default_labels(i, bufs):
    if i == 1:
        return ["Project Site"] + [("Core Zone " if k == 0 else "Buffer Zone ") + fmt_m(m) for k, m in enumerate(bufs)]
    return ["Transmission Line"] + [f"Transmission Line Buffer {fmt_m(m)}" for m in bufs]


def make_layer(g, labels, buf_m, dash):
    u = g.to_crs(g.estimate_utm_crs())
    one = unary_union(list(u.geometry))
    gj = lambda geom: json.loads(gpd.GeoSeries([geom], crs=u.crs).to_crs(4326).to_json())
    feats = [{"props": f["properties"], "geom": f["geometry"]}
             for f in json.loads(g.to_json(default=str))["features"] if f["geometry"]]
    return {"name": labels[0], "dash": dash, "site": gj(one), "feats": feats,
            "pts": any(p.geom_type == "Point" for p in getattr(one, "geoms", [one])),
            "bufs": [{"m": m, "label": lb, "gj": gj(one.buffer(m))} for m, lb in zip(buf_m, labels[1:])]}


light = lambda l: {k: v for k, v in l.items() if k != "feats"}
site_fc = lambda l: {"type": "FeatureCollection",
                     "features": [{"type": "Feature", "properties": f["props"], "geometry": f["geom"]} for f in l["feats"]]}


def legend_rows(layers):
    return [r for l in layers for r in [(l["name"], COLORS[0], SITE_W, l["dash"], l.get("pts", False))] +
            [(b["label"], buf_color(i, len(l["bufs"]), l.get("cols", COLORS)), 2, l["dash"], False) for i, b in enumerate(l["bufs"])]]


def line_rows(rd, con, water=None):
    rows = [("Roads & Transportation", ROAD_COL, 2, "line")] if rd else []
    for k, fc in (water or {}).items():
        rows.append((DRAIN[k][0], DRAIN[k][2], 1.5,
                     "fill" if any("Polygon" in f["geometry"]["type"] for f in fc["features"]) else "line"))
    if con:
        rows.append((f"Contour {CONTOUR_STEP} meter", CON_COL, 1, "line"))
    return rows


dem_labels = lambda edges: [f"{edges[i]:.0f} - {edges[i + 1]:.0f}" for i in range(5)]


def source_text(kind):
    hd, hc, hr, hw, hs = KINDS[kind]
    return WATER_SRC if hw else SAT_SRC if hs else DEM_SRC if (hd or hc) else ROAD_SRC


# ---------------- KML ----------------
def kml_geom(g):
    ring = lambda c: "<coordinates>" + " ".join(f"{x:.6f},{y:.6f},0" for x, y, *_ in c) + "</coordinates>"
    t = g.geom_type
    if g.is_empty:
        return ""
    if t == "Point":
        return f"<Point><coordinates>{g.x:.6f},{g.y:.6f},0</coordinates></Point>"
    if t == "Polygon":
        return ("<Polygon><outerBoundaryIs><LinearRing>" + ring(g.exterior.coords) + "</LinearRing></outerBoundaryIs>" +
                "".join("<innerBoundaryIs><LinearRing>" + ring(r.coords) + "</LinearRing></innerBoundaryIs>"
                        for r in g.interiors) + "</Polygon>")
    if t in ("LineString", "LinearRing"):
        return "<LineString><tessellate>1</tessellate>" + ring(g.coords) + "</LineString>"
    return "<MultiGeometry>" + "".join(kml_geom(s) for s in g.geoms) + "</MultiGeometry>"


def feat_name(props, fallback):
    return next((str(v) for k, v in props.items() if k.lower() == "name" and v not in (None, "")), fallback)


def to_kml(name, feats, color="#000000", width=2):
    c = "ff" + color[5:7] + color[3:5] + color[1:3]
    sty = f"<Style><LineStyle><color>{c}</color><width>{width}</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>"

    def placemark(n, g, props=None):
        props = props or {}
        desc = next((str(v) for k, v in props.items() if k.lower() == "description" and v not in (None, "")), "")
        ext = "".join(f'<Data name="{escape(str(k), {chr(34): "&quot;"})}"><value>{escape(str(v))}</value></Data>'
                      for k, v in props.items() if k.lower() not in ("name", "description") and v not in (None, ""))
        return (f"<Placemark><name>{escape(n)}</name>" + (f"<description>{escape(desc)}</description>" if desc else "") +
                sty + (f"<ExtendedData>{ext}</ExtendedData>" if ext else "") + f"{kml_geom(g)}</Placemark>")

    return ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2">'
            f'<Document><name>{escape(name)}</name>{"".join(placemark(*f) for f in feats)}</Document></kml>')


# ---------------- roads / water / DEM / contours ----------------
def to_fc(g, cols=()):
    rnd = lambda c: [round(c[0], 5), round(c[1], 5)] if isinstance(c[0], float) else [rnd(p) for p in c]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": dict(zip(cols, v)),
         "geometry": {"type": m.geom_type, "coordinates": rnd(mapping(m)["coordinates"])}}
        for m, *v in zip(g.geometry, *[g[c] for c in cols])]}


def clip_geoms(g, ou, tol, kinds=("LineString",)):
    g = g[g.geometry.notna()].copy()
    g["geometry"] = g.geometry.apply(make_valid).intersection(ou).simplify(tol)
    g = g[~g.geometry.is_empty].explode(index_parts=False)
    g = g[g.geom_type.isin(kinds)]
    return g[~g.geometry.is_empty]


def tile_range(b, z):
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    x0, x1 = sorted((tx(b[0]), tx(b[2])))
    y0, y1 = sorted((ty(b[1]), ty(b[3])))
    return n, max(x0, 0), min(x1, n - 1), max(y0, 0), min(y1, n - 1)


def read_vt(path, ou_wkb, kinds):
    mvt = ensure("mapbox_vector_tile", "mapbox-vector-tile")
    ou, rd = swkb.loads(ou_wkb), reader(path)
    h = rd.header()
    z = h["max_zoom"]
    while True:
        n, x0, x1, y0, y1 = tile_range(ou.bounds, z)
        if z <= h["min_zoom"] or (x1 - x0 + 1) * (y1 - y0 + 1) <= 300:
            break
        z -= 1

    def one(xy):
        x, y = xy
        data = rd.get(z, x, y)
        if data and data[:2] == b"\x1f\x8b":
            data = zlib.decompress(data, 31)
        out = []
        for ly in (mvt.decode(data, y_coord_down=True).values() if data else []):
            e = ly.get("extent", 4096)
            ll = lambda c: np.column_stack(((x + c[:, 0] / e) / n * 360 - 180,
                                            np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + c[:, 1] / e) / n))))))
            out += [shapely.transform(shape(f["geometry"]), ll) for f in ly["features"]]
        return out

    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = [g for r in ex.map(one, [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]) for g in r]
    groups = [[make_valid(g) for g in parts if any(t in g.geom_type for t in tp)]
              for k, tp in (("LineString", ("LineString",)), ("Polygon", ("Polygon",))) if k in kinds]
    geoms = [unary_union(gr) for gr in groups if gr]
    if not geoms:
        return None
    g = clip_geoms(gpd.GeoDataFrame(geometry=geoms, crs=4326), ou, 1e-5, kinds)
    return to_fc(g) if len(g) else None


build_roads = lambda path, ou_wkb, v=None: read_vt(path, ou_wkb, ("LineString",))
build_water = lambda path, ou_wkb, v=None: read_vt(path, ou_wkb, ("Polygon", "LineString"))


def contour_label_points(g, lens_m, step=LABEL_STEP, per=3000, cap=1500):
    out = []
    for geom, e, L in zip(g.geometry, g["elev"], lens_m):
        if e % step:
            continue
        n = max(1, int(L // per))
        for k in range(n):
            q = geom.interpolate((k + .5) / n, normalized=True)
            out.append([round(q.y, 5), round(q.x, 5), int(e)])
    return out[:cap]


def dem_grid(b, zmax=13, span=12):
    z = zmax
    while z > 8:
        _, x0, x1, y0, y1 = tile_range(b, z)
        if max(x1 - x0, y1 - y0) < span:
            break
        z -= 1
    cv, ext = mosaic((z, z, lambda zz, x, y: web_img(TERRAIN_URL.format(z=zz, x=x, y=y)), None), list(b), 99)
    a = np.asarray(cv, dtype="float32")
    e = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    e[a[..., 3] == 0] = np.nan
    del a
    if np.isnan(e).all():
        raise RuntimeError("no elevation tiles for this area")
    return e, (ext[0], ext[3], (ext[1] - ext[0]) / cv.width)


def build_terrain(ou_wkb):
    from contourpy import contour_generator
    ou = swkb.loads(ou_wkb)
    if ou.area == 0:
        raise RuntimeError("add a buffer distance first (the DEM is clipped to the outermost buffer)")
    e, (EX0, EY1, p) = dem_grid(ou.bounds, 13, 12)
    om = gpd.GeoSeries([ou], crs=4326).to_crs(3857).iloc[0]
    bx0, by0, bx1, by1 = om.bounds
    c0, c1 = max(int((bx0 - EX0) // p), 0), min(int((bx1 - EX0) // p) + 1, e.shape[1])
    r0, r1 = max(int((EY1 - by1) // p), 0), min(int((EY1 - by0) // p) + 1, e.shape[0])
    e, X0, Y1 = e[r0:r1, c0:c1], EX0 + c0 * p, EY1 - r0 * p
    m = Image.new("L", e.shape[::-1], 0)
    dm = ImageDraw.Draw(m)
    for pg in getattr(om, "geoms", [om]):
        for ring, v in [(pg.exterior, 255)] + [(r, 0) for r in pg.interiors]:
            dm.polygon([((x - X0) / p, (Y1 - y) / p) for x, y in ring.coords], fill=v)
    zi = np.where(np.asarray(m) > 0, e, np.nan).astype("float32")
    v = zi[~np.isnan(zi)]
    if not v.size:
        raise RuntimeError("no DEM data inside the buffer")
    lo, hi = math.floor(v.min()), math.ceil(v.max())
    edges = np.linspace(lo, max(hi, lo + 1), 6).tolist()
    f = np.where(np.isnan(e), np.nanmin(e), e)  # light smoothing -> cleaner contours
    pd = np.pad(f, 1, mode="edge")
    f = sum(pd[i:i + f.shape[0], j:j + f.shape[1]] for i in range(3) for j in range(3)) / 9
    xs = X0 + (np.arange(f.shape[1]) + .5) * p
    ys = (Y1 - (np.arange(f.shape[0]) + .5) * p)[::-1]
    cg = contour_generator(xs, ys, f[::-1])
    rows = [(lv, LineString(l)) for lv in range(math.ceil(v.min() / CONTOUR_STEP) * CONTOUR_STEP,
                                                  math.floor(v.max()) + 1, CONTOUR_STEP)
            for l in cg.lines(lv) if len(l) > 1]
    cont, clen, labs = {"type": "FeatureCollection", "features": []}, 0.0, []
    if rows:
        gl = gpd.GeoDataFrame({"elev": [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=3857)
        cl_ = clip_geoms(gl, om, p / 2).to_crs(4326)
        if len(cl_):
            lens_m = cl_.to_crs(cl_.estimate_utm_crs()).length.values
            keep = lens_m >= CONTOUR_MIN_LEN
            cl_, lens_m = cl_[keep], lens_m[keep]
            if len(cl_):
                cont, clen, labs = to_fc(cl_, ["elev"]), float(lens_m.sum()) / 1000, contour_label_points(cl_, lens_m)
    return zi, (X0, Y1, p), edges, cont, clen, labs


def dem_rgba(z, g, edges):
    X0, Y1, p = g
    h, w = z.shape
    ok = ~np.isnan(z)
    rgba = np.zeros((h, w, 4), np.uint8)
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in DEM_PAL], np.uint8)
    rgba[ok, :3] = pal[np.digitize(z[ok], edges[1:-1])]
    rgba[ok, 3] = 255
    return Image.fromarray(rgba), [X0, X0 + w * p, Y1 - h * p, Y1]


def geotiff_bytes(a, g, nodata, epsg=3857):
    X0, Y1, p = g
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550], ifd[33922] = (p, p, 0.0), (0.0, 0.0, 0.0, float(X0), float(Y1), 0.0)
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, epsg)
    ifd[42113] = str(nodata)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "TIFF", tiffinfo=ifd, compression="tiff_deflate")
    return buf.getvalue()


# ---------------- viewshed + vantage-point selection ----------------
def bilin(a, x, y):
    h, w = a.shape
    x, y = np.clip(x, 0, w - 1.001), np.clip(y, 0, h - 1.001)
    x0, y0 = x.astype(int), y.astype(int)
    fx, fy = x - x0, y - y0
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy) +
            a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)


def viewshed(z, g, ox, oy, h_obs, h_tgt, rad):
    """Line of sight from (ox, oy) [EPSG:3857] -> (row0, col0, bool window) or None if outside the DEM."""
    X0, Y1, p = g
    pm = p * math.cos(math.radians(unmerc(ox, oy)[1]))
    c, r = (ox - X0) / p - .5, (Y1 - oy) / p - .5
    ci, ri, (H, W), rp = int(round(c)), int(round(r)), z.shape, rad / pm
    if not (0 <= ci < W and 0 <= ri < H):
        return None
    k = int(math.ceil(rp)) + 1
    r0, r1, c0, c1 = max(ri - k, 0), min(ri + k + 1, H), max(ci - k, 0), min(ci + k + 1, W)
    sub, oc, orr = z[r0:r1, c0:c1], c - c0, r - r0
    z0 = float(sub[ri - r0, ci - c0]) + h_obs
    A, d = max(360, int(2 * math.pi * rp * 1.5)), np.arange(1, int(rp * 2) + 2) * .5
    a = np.linspace(0, 2 * math.pi, A, endpoint=False)[:, None]
    dm = d * pm
    hh = bilin(sub, oc + np.cos(a) * d, orr - np.sin(a) * d) - VS_CC * dm ** 2 / (2 * R)
    prev = np.maximum.accumulate((hh - z0) / dm, axis=1)
    prev = np.concatenate([np.full((A, 1), -np.inf), prev[:, :-1]], axis=1)
    vis = (hh + h_tgt - z0) / dm >= prev
    yy, xx = np.mgrid[0:r1 - r0, 0:c1 - c0]
    dx, dy = xx - oc, orr - yy
    dist = np.hypot(dx, dy)
    ai = np.round(np.arctan2(dy, dx) % (2 * math.pi) / (2 * math.pi) * A).astype(int) % A
    win = vis[ai, np.clip(np.round(dist * 2).astype(int) - 1, 0, len(d) - 1)] & (dist * pm <= rad)
    win[ri - r0, ci - c0] = True
    return r0, c0, win


def vantage_batches(pts, h_obs, h_tgt, rad):
    """pts = ((lon, lat, name), ...) WTGs. Greedy vantage selection, VS_BATCHES distinct batches, real viewshed summed into a count raster."""
    lo, la = [q[0] for q in pts], [q[1] for q in pts]
    m = (rad + 300) / 111320
    cl = math.cos(math.radians(sum(la) / len(la)))
    e, g = dem_grid((min(lo) - m / cl, min(la) - m, max(lo) + m / cl, max(la) + m), 13, 9)
    X0, Y1, p = g
    H, W = e.shape
    lo_e, hi_e = float(np.nanmin(e)), float(np.nanmax(e))
    zf = np.where(np.isnan(e), lo_e, e)
    nrm = (zf - lo_e) / ((hi_e - lo_e) or 1.0)
    pm = p * cl
    obs = [merc(x, y) for x, y, _ in pts]
    cell = lambda x, y: (int(round((Y1 - y) / p - .5)), int(round((x - X0) / p - .5)))
    excl = np.zeros((H, W), bool)
    for x, y in obs:
        r, c = cell(x, y)
        excl[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = True
    wins = [viewshed(zf, g, x, y, h_obs, h_tgt, rad) for x, y in obs]
    sees = lambda w, r, c: bool(w) and 0 <= r - w[0] < w[2].shape[0] and 0 <= c - w[1] < w[2].shape[1] and w[2][r - w[0], c - w[1]]
    bp, used, out = max(1, int(round(VS_EXCL_M / pm))), np.zeros((H, W), bool), []
    for _ in range(VS_BATCHES):
        left, sel = set(range(len(pts))), []
        while left:
            cov = np.zeros((H, W), np.float32)
            for i in left:
                if wins[i]:
                    r0, c0, w = wins[i]
                    cov[r0:r0 + w.shape[0], c0:c0 + w.shape[1]] += w
            cov[excl | used] = 0
            if cov.max() == 0:
                break
            r, c = np.unravel_index(np.argmax(np.where(cov > 0, cov + nrm * VS_ELEV_W, 0)), cov.shape)
            hit = [i for i in left if sees(wins[i], r, c)]
            left -= set(hit)
            x, y = X0 + (c + .5) * p, Y1 - (r + .5) * p
            lon, lat = unmerc(x, y)
            sel.append({"x": x, "y": y, "lon": lon, "lat": lat, "wtg_count": len(hit),
                        "elev_m": float(e[r, c]) if not np.isnan(e[r, c]) else 0.0, "wtg_names": ", ".join(pts[i][2] for i in hit)})
            used[max(r - bp, 0):r + bp + 1, max(c - bp, 0):c + bp + 1] = True
        cum = np.zeros((H, W), np.uint16)
        for s in sel:
            w = viewshed(zf, g, s["x"], s["y"], h_obs, h_tgt, rad)
            if w:
                cum[w[0]:w[0] + w[2].shape[0], w[1]:w[1] + w[2].shape[1]] += w[2]
        out.append({"pts": sel, "cov": cum, "left": len(left), "missed": ", ".join(pts[i][2] for i in sorted(left))})
    return {"g": g, "batches": out}


def vp_kml(n, bt):
    return to_kml(f"Vantage Points Batch {n}",
                  [(f"Vantage {i}", Point(q["lon"], q["lat"]),
                    {"batch": n, "wtg_count": q["wtg_count"], "elev_m": round(q["elev_m"], 2), "wtg_names": q["wtg_names"]})
                   for i, q in enumerate(bt["pts"], 1)], VS_COL[(n - 1) % 3])


def wtg_buffer(pts, rad):
    u = gpd.GeoSeries([Point(x, y) for x, y, _ in pts], crs=4326)
    u = u.to_crs(u.estimate_utm_crs())
    return json.loads(gpd.GeoSeries([unary_union(list(u.buffer(rad)))], crs=u.crs).to_crs(4326).to_json())


def cov_png(cov, g, col):
    """Visibility count raster -> (PNG data URL, [[S, W], [N, E]]); more WTGs visible = darker / more opaque."""
    X0, Y1, p = g
    h, w = cov.shape
    m, rgba = cov.astype(np.float32), np.zeros((h, w, 4), np.uint8)
    base = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], np.float32)
    k = np.where(m > 0, 1.0 - 0.45 * m / max(float(m.max()), 1), 1.0)[..., None]  # more WTGs -> darker shade
    rgba[..., :3] = np.clip(base * k, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(m > 0, 150 + 105 * m / max(float(m.max()), 1), 0).astype(np.uint8)
    im = Image.fromarray(rgba)
    if max(im.size) > VS_RAS_MAXPX:   # keep the data URL small so the map iframe always loads it
        f = VS_RAS_MAXPX / max(im.size)
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    (lo0, la0), (lo1, la1) = unmerc(X0, Y1 - h * p), unmerc(X0 + w * p, Y1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), [[la0, lo0], [la1, lo1]]


# =====================================================================================
# EXPORT WORKER
# =====================================================================================
def pm_src(aid):
    rd = reader(aid)
    h = rd.header()

    def get(z, x, y):
        data = rd.get(z, x, y)
        return Image.open(io.BytesIO(data)) if data else None
    return h["min_zoom"], h["max_zoom"], get, sheet_bounds(aid)


xyz_src = lambda url: (0, 19, lambda z, x, y: web_img(url.format(z=z, x=x, y=y)), None)


def mosaic(src, bounds, max_tiles, on_tile=None):
    zmin, z, get, sb = src
    if sb:
        bounds = [max(bounds[0], sb[0]), max(bounds[1], sb[1]), min(bounds[2], sb[2]), min(bounds[3], sb[3])]
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            return None
    n, x0, x1, y0, y1 = tile_range(bounds, z)
    while z > zmin and (x1 - x0 + 1 > max_tiles or y1 - y0 + 1 > max_tiles):
        z -= 1
        n, x0, x1, y0, y1 = tile_range(bounds, z)
    if z == zmin and zmin > 8 and (x1 - x0 + 1 > max_tiles or y1 - y0 + 1 > max_tiles):
        return None
    canvas = Image.new("RGBA", ((x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256), (0, 0, 0, 0))
    tiles = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]

    def read(zl, x, y):
        for a in range(3):
            try:
                im = get(zl, x, y)
                if im is None:
                    return None
                t = im.convert("RGBA")
                return t if t.size == (256, 256) else t.resize((256, 256), Image.Resampling.BILINEAR)
            except Exception:
                time.sleep(.3 * (a + 1))

    def load(xy):  # missing tile -> upscale parent tile
        x, y = xy
        for s in range(min(3, z - zmin) + 1):
            t = read(z - s, x >> s, y >> s)
            if t is not None:
                if s:
                    q, m = 256 >> s, (1 << s) - 1
                    cx0, cy0 = (x & m) * q, (y & m) * q
                    t = t.crop((cx0, cy0, cx0 + q, cy0 + q)).resize((256, 256), Image.Resampling.BICUBIC)
                return xy, t
        return xy, None

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


merc = lambda lon, lat: (R * math.radians(lon), R * math.asinh(math.tan(math.radians(lat))))
unmerc = lambda x, y: (math.degrees(x / R), math.degrees(math.atan(math.sinh(y / R))))


def get_font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arialbd.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"])
    for n in names:
        for p in (os.path.join(FONT_DIR, n), n):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def put_text(d, xy, s, font, anchor="la"):
    try:
        d.text(xy, s, font=font, fill="black", anchor=anchor)
    except ValueError:
        d.text(xy, s, font=font, fill="black")


def center_text(d, s, cx, cy, w, h, size, bold=False):
    s = " ".join(s.split())
    for sz in range(size, 5, -1):
        f, sp = get_font(sz, bold), sz * .2
        lines, cur = [], ""
        for word in s.split(" "):
            t = (cur + " " + word).strip()
            if not cur or d.textlength(t, font=f) <= w:
                cur = t
            else:
                lines.append(cur)
                cur = word
        t = "\n".join(lines + [cur])
        bb = d.multiline_textbbox((0, 0), t, font=f, spacing=sp, align="center")
        if bb[2] - bb[0] <= w and bb[3] - bb[1] <= h:
            break
    d.multiline_text((cx - (bb[0] + bb[2]) / 2, cy - (bb[1] + bb[3]) / 2), t, font=f, fill="black",
                     align="center", spacing=sp)


def prepared_by(img, d, cx, ya, yb, w, size, logo_bytes, logo_h):
    li = None
    if logo_bytes:
        try:
            li = Image.open(io.BytesIO(logo_bytes)).convert("RGB")
        except Exception:
            pass
    if li is None:
        center_text(d, "Prepared by Kosher Climate", cx, (ya + yb) / 2, w * .92, (yb - ya) * .8, size, True)
        return
    f, label, gap = get_font(size, True), "Prepared by", max(4, int(size * .5))
    tw = d.textlength(label, font=f)
    h = max(8, min(int((yb - ya) * .98), logo_h))
    lw = int(li.width * h / li.height)
    room = w * .92 - tw - gap
    if lw > room > 20:
        lw, h = int(room), max(4, int(li.height * room / li.width))
    lw, h = max(int(lw * LOGO_SCALE), 1), max(int(h * LOGO_SCALE), 4)
    li = li.resize((lw, h), Image.Resampling.LANCZOS)
    x, cy = cx - (tw + gap + li.width) / 2, (ya + yb) / 2
    put_text(d, (x, cy), label, f, "lm")
    img.paste(li, (int(x + tw + gap), int(cy - li.height / 2)))


nice_step = lambda raw: next((s for s in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if s >= raw), 10)


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


def draw_windmill(d, x, y, s, color="black"):
    w = max(1, int(s * .12))

    def blades():
        for a in (90, 210, 330):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            yield [(x - sa * s * .1, y - ca * s * .1), (x + s * ca, y - s * sa), (x + sa * s * .1, y + ca * s * .1)]

    d.line([(x, y), (x, y + s * 1.15)], fill="white", width=w * 3)
    for poly in blades():
        d.polygon(poly, fill="white", outline="white", width=w)
    d.line([(x, y), (x, y + s * 1.15)], fill=color, width=w)
    for poly in blades():
        d.polygon(poly, fill=color)
    r = s * .13
    d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white")


def draw_fc(md, fc, tp, color, width):
    if fc and fc["features"]:
        for geom in gpd.GeoDataFrame.from_features(fc["features"], crs=4326).to_crs(3857).geometry:
            for path in geom_paths(geom):
                draw_path(md, [tp(x, y) for x, y in path], color, width)


def draw_water(map_img, md, water, tp, pt):
    for k, fc in (water or {}).items():
        if not fc["features"]:
            continue
        col = DRAIN[k][2]
        gm = gpd.GeoDataFrame.from_features(fc["features"], crs=4326).to_crs(3857)
        mask = Image.new("L", map_img.size, 0)
        mdr = ImageDraw.Draw(mask)
        for g in gm.geometry:
            for sub in getattr(g, "geoms", [g]):
                if sub.geom_type == "Polygon":
                    mdr.polygon([tp(x, y) for x, y in (c[:2] for c in sub.exterior.coords)], fill=170)
                    for r in sub.interiors:
                        mdr.polygon([tp(x, y) for x, y in (c[:2] for c in r.coords)], fill=0)
        map_img.paste(tuple(int(col[i:i + 2], 16) for i in (1, 3, 5)), (0, 0), mask)
        del mask, mdr
        for g in gm.geometry:
            for sub in getattr(g, "geoms", [g]):
                for path in geom_paths(sub):
                    draw_path(md, [tp(x, y) for x, y in path], col, pt(.8) if sub.geom_type == "Polygon" else pt(1.2))


def _along(pts, seg, dist):
    acc = 0.0
    for (a, b), s in zip(zip(pts[:-1], pts[1:]), seg):
        if s and acc + s >= dist:
            t = (dist - acc) / s
            return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        acc += s
    return pts[-1]


def draw_contour_labels(map_img, fc, tp, pt, step):
    if not (fc and fc["features"]):
        return
    gm = gpd.GeoDataFrame.from_features(fc["features"], crs=4326).to_crs(3857)
    font, sw = get_font(pt(8), True), max(1, pt(1.2))
    probe = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    W_, H_ = map_img.size
    margin, min_len, min_gap, placed = pt(18), pt(90), pt(45), []
    for geom, e in zip(gm.geometry, gm["elev"]):
        if e % step:
            continue
        txt = str(int(e))
        for path in geom_paths(geom):
            pts = [tp(x, y) for x, y in path]
            seg = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:])]
            L = sum(seg)
            if L < min_len:
                continue
            n = max(1, int(L // pt(700)))
            for k in range(n):
                t = L * (k + .5) / n
                cx_, cy_ = _along(pts, seg, t)
                if not (margin < cx_ < W_ - margin and margin < cy_ < H_ - margin):
                    continue
                if any(math.hypot(cx_ - px, cy_ - py) < min_gap for px, py in placed):
                    continue
                wdw = pt(14)
                p0, p1 = _along(pts, seg, max(0, t - wdw)), _along(pts, seg, min(L, t + wdw))
                ang = math.degrees(math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0]))
                ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
                bb = probe.textbbox((0, 0), txt, font=font, stroke_width=sw)
                pad = sw + 2
                tile = Image.new("RGBA", (bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad), (0, 0, 0, 0))
                ImageDraw.Draw(tile).text((pad - bb[0], pad - bb[1]), txt, font=font, fill="black",
                                          stroke_width=sw, stroke_fill="white")
                rot = tile.rotate(ang, expand=True, resample=Image.Resampling.BICUBIC)
                map_img.paste(rot, (int(cx_ - rot.width / 2), int(cy_ - rot.height / 2)), rot)
                placed.append((cx_, cy_))


def paste_mosaic(map_img, canvas, ext, ox, oy, scale, mx0, my0):
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


def whiten_outside(map_img, geoms, tp):
    mask = Image.new("L", map_img.size, 255)
    md = ImageDraw.Draw(mask)
    for g in geoms:
        for sub in getattr(g, "geoms", [g]):
            if sub.geom_type == "Polygon":
                md.polygon([tp(x, y) for x, y in (c[:2] for c in sub.exterior.coords)], fill=0)
                for r in sub.interiors:
                    md.polygon([tp(x, y) for x, y in (c[:2] for c in r.coords)], fill=255)
    map_img.paste((255, 255, 255), (0, 0), mask)


def export_map(sheets_pm, layers, title, sheets_str, dpi, legend_bytes, progress, out_path, kind="Toposheet Map", ex=None):
    ex = ex or {}
    has_dem, has_con, has_rd, has_wt, has_sat = KINDS[kind]
    topo = not any(KINDS[kind])
    pt = lambda v: max(1, int(round(v * dpi / 72)))
    W, H = int(16 * dpi), int(11 * dpi)
    mx0, mx1, my0, my1 = int(.035 * W), int(.75 * W), int(.055 * H), int(.935 * H)
    px0, px1 = mx1, int(.995 * W)
    mw, mh, pw, ph = mx1 - mx0, my1 - my0, px1 - px0, my1 - my0

    progress(0.01, "Preparing extent...")
    to_m = lambda gj: gpd.GeoDataFrame.from_features(gj["features"], crs=4326).to_crs(3857).geometry
    for l in layers:
        l["site"], l["bufs"] = to_m(l["site"]), [dict(b, gj=to_m(b["gj"])) for b in l["bufs"]]
    if layers:
        bs = [g.total_bounds for l in layers for g in [l["site"]] + [b["gj"] for b in l["bufs"]]]
        b = [min(v[0] for v in bs), min(v[1] for v in bs), max(v[2] for v in bs), max(v[3] for v in bs)]
        dx, dy = (b[2] - b[0]) * .075, (b[3] - b[1]) * .075
        vx0, vx1, vy0, vy1 = b[0] - dx, b[2] + dx, b[1] - dy, b[3] + dy
    else:
        with ThreadPoolExecutor(max_workers=TILE_WORKERS) as exr:
            sbs = [b for b in exr.map(sheet_bounds, sheets_pm) if b]
        bb = ([min(b[0] for b in sbs), min(b[1] for b in sbs), max(b[2] for b in sbs), max(b[3] for b in sbs)]
              if sbs else [68, 6, 98, 36])
        (vx0, vy0), (vx1, vy1) = merc(bb[0], bb[1]), merc(bb[2], bb[3])

    scale = min(mw / (vx1 - vx0), mh / (vy1 - vy0))
    cx, cy = (vx0 + vx1) / 2, (vy0 + vy1) / 2
    ox, oy = mx0 + mw / 2 - cx * scale, my0 + mh / 2 + cy * scale
    (lon0, lat0) = unmerc((mx0 - ox) / scale, (oy - my1) / scale)
    (lon1, lat1) = unmerc((mx1 - ox) / scale, (oy - my0) / scale)
    bbox = [lon0, lat0, lon1, lat1]
    tp = lambda x, y: (ox + x * scale - mx0, oy - y * scale - my0)

    def overlaps(aid):
        sb = sheet_bounds(aid)
        return sb is None or not (sb[2] <= bbox[0] or sb[0] >= bbox[2] or sb[3] <= bbox[1] or sb[1] >= bbox[3])

    with ThreadPoolExecutor(max_workers=16) as exr:
        list(exr.map(sheet_bounds, sheets_pm))
    sheets_pm = [a for a in sheets_pm if overlaps(a)]
    items = [(os.path.basename(a), lambda a=a: pm_src(a)) for a in sheets_pm] + \
            ([("basemap", lambda: xyz_src(ex["base"]))] if ex.get("base") else [])
    n = max(len(items), 1)
    max_tiles = min(16, math.ceil(max(mw, mh) / 256))

    map_img, skipped = Image.new("RGB", (mw, mh), "white"), []
    for i, (nm, mk) in enumerate(items):
        for attempt in range(3):
            try:
                res = mosaic(mk(), bbox, max_tiles,
                             lambda dn, t, idx=i, nm=nm: progress((idx + dn / t) / n * 0.75,
                                                                  f"Fetching tiles: {nm} {idx + 1}/{n} ({dn}/{t})"))
                if res:
                    paste_mosaic(map_img, res[0], res[1], ox, oy, scale, mx0, my0)
                    del res
                break
            except Exception:
                time.sleep(2)
        else:
            skipped.append(nm)
        gc.collect()
    if items and len(skipped) == len(items) and (sheets_pm or has_sat):
        raise RuntimeError("None of the selected sources could be loaded: " + ", ".join(skipped))
    if has_sat and layers:
        whiten_outside(map_img, [g for l in layers for g in (l["bufs"][-1]["gj"] if l["bufs"] else l["site"])], tp)
    if has_dem:
        z, gtr, edges = ex["dem"]
        im, e3 = dem_rgba(z, gtr, edges)
        paste_mosaic(map_img, im, e3, ox, oy, scale, mx0, my0)
        del im

    progress(0.8, "Drawing map overlay and legend...")
    md = ImageDraw.Draw(map_img)
    if has_wt:
        draw_water(map_img, md, ex.get("water"), tp, pt)
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

    if has_con:
        draw_fc(md, ex.get("contours"), tp, CON_COL, pt(.8))
    if has_rd:
        draw_fc(md, ex.get("roads"), tp, ROAD_COL, pt(1.2))
    vec, wm = [], []
    for l in layers:
        dash, n_, cols = ((pt(6), pt(4)) if l["dash"] else None), len(l["bufs"]), l.get("cols", COLORS)
        for geoms, c, lw in [(b["gj"], buf_color(i, n_, cols), 2) for i, b in enumerate(l["bufs"])] + [(l["site"], COLORS[0], SITE_W)]:
            for g in geoms:
                for path in geom_paths(g):
                    pts_ = [tp(x, y) for x, y in path]
                    if len(pts_) == 1:
                        wm.append(pts_[0])
                    else:
                        vec.append((pts_, c, pt(lw), dash))
    if has_sat:
        for pts, c, lw, dash in vec:
            draw_path(md, pts, "white", lw + pt(1))
    for pts, c, lw, dash in vec:
        draw_path(md, pts, c, lw, dash)
    for x_, y_ in wm:
        draw_windmill(md, x_, y_, pt(12))
    if has_con:
        draw_contour_labels(map_img, ex.get("contours"), tp, pt, LABEL_STEP)

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
    put_text(d, ((mx0 + mx1) / 2, H * .03), "Topographic MAP" if topo else kind, get_font(pt(22), True), "mm")
    PX = lambda u: px0 + u * pw
    PY = lambda v: my1 - v * ph
    d.rectangle([px0, my0, px1, my1], outline="black", width=pt(1.5))
    px_per_km = 1000 * scale / math.cos(math.radians(unmerc(cx, cy)[1]))
    km = max([0.1] + [c for c in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if c * px_per_km <= pw * .65])
    bl, bx, by = km * px_per_km, PX(.06), PY(.945)
    d.line([(bx, by), (bx + bl, by)], fill="black", width=pt(1.5))
    f_sb = get_font(pt(10), True)
    for frac, lab in ((0, "0"), (.5, f"{km / 2:g}"), (1, f"{km:g} km")):
        d.line([(bx + bl * frac, by), (bx + bl * frac, by - pt(5))], fill="black", width=pt(1.5))
        put_text(d, (bx + bl * frac, by - pt(7)), lab, f_sb, "ms")

    put_text(d, (PX(.82), PY(.975)), "N", get_font(pt(22), True), "mm")
    ax_, ay_, hh, ww = PX(.82), PY(.929), pt(13), pt(10)
    top, lft, rgt, mid = (ax_, ay_ - hh), (ax_ - ww, ay_ + hh), (ax_ + ww, ay_ + hh), (ax_, ay_ + hh * .45)
    d.polygon([top, lft, mid], fill="black")
    d.polygon([top, mid, rgt], fill="white", outline="black", width=pt(1.2))
    d.line([top, lft, mid, top], fill="black", width=pt(1.2))

    rows = legend_rows(layers)
    lrows = [] if has_sat else line_rows(has_rd, has_con, ex.get("water") if has_wt else None)
    nrow = len(rows) + len(lrows)
    sep = min(.5 if topo else .46 if has_dem else .22, .672 - .03 * nrow + .02)
    for v in (.905, .74, sep):
        d.line([(px0, PY(v)), (px1, PY(v))], fill="black", width=pt(1))
    tt, tb = PY(.905), PY(.74)
    put_text(d, (PX(.5), tt + (tb - tt) * .13), "Title :", get_font(pt(TITLE_HEAD_FS), True), "mm")
    center_text(d, title, PX(.5), tt + (tb - tt) * .58, pw * .88, (tb - tt) * .62, pt(TITLE_FS))

    put_text(d, (PX(.5), PY(.715)), "Legends", get_font(pt(LEG_HEAD_FS), True), "mm")
    f_leg, y = get_font(pt(LEG_FS)), .672
    if not (topo or has_dem) and nrow:
        y = min(.672, (.69 + sep) / 2 + (.03 * (nrow - 1) + .02) / 2 - .02)
    for lab, c, lw, dashed, icon in rows:
        x0_, x1_, y0_, y1_ = PX(.06), PX(.19), PY(y + .02), PY(y)
        if icon:
            draw_windmill(d, (x0_ + x1_) / 2, (y0_ + y1_) / 2 - ph * .006, ph * .011)
        else:
            draw_path(d, [(x0_, y0_), (x1_, y0_), (x1_, y1_), (x0_, y1_), (x0_, y0_)], c, pt(lw), (pt(6), pt(4)) if dashed else None)
        put_text(d, (PX(.22), PY(y + .01)), lab, f_leg, "lm")
        y -= .03
    for lab, c, lw, sty in lrows:
        if sty == "fill":
            d.rectangle([PX(.06), PY(y + .02), PX(.19), PY(y)], fill=tint(c), outline=c, width=pt(lw))
        else:
            d.line([(PX(.06), PY(y + .01)), (PX(.19), PY(y + .01))], fill=c, width=pt(lw))
        put_text(d, (PX(.22), PY(y + .01)), lab, f_leg, "lm")
        y -= .03

    if topo:
        if legend_bytes:
            try:
                li = Image.open(io.BytesIO(legend_bytes)).convert("RGBA")
                x0_, y0_, x1_, y1_ = int(PX(.005)), int(PY(sep - .005)), int(PX(.995)), int(PY(.005))
                li = li.resize((x1_ - x0_, y1_ - y0_), Image.Resampling.LANCZOS)
                img.paste(li, (x0_, y0_), li)
                del li
            except Exception:
                pass
        f_ft = get_font(pt(12), True)
        put_text(d, (.15 * W, .975 * H), "Source : Survey of India", f_ft, "lm")
        put_text(d, (.42 * W, .975 * H), f"Topo sheet No. {sheets_str}", f_ft, "lm")
    else:
        wts = [1.2, 1] + ([3] if has_dem else [])
        u, eg = (sep - .01) / sum(wts), [.005]
        for w_ in wts:
            eg.append(eg[-1] + w_ * u)
        for v in eg[1:-1]:
            d.line([(px0, PY(v)), (px1, PY(v))], fill="black", width=pt(1))
        for k, txt in ((0, None), (1, "Source : " + source_text(kind))):
            ya, yb = PY(eg[k + 1]), PY(eg[k])
            if txt is None:
                prepared_by(img, d, PX(.5), ya, yb, pw, pt(11), ex.get("logo"), pt(LOGO_H_PT))
            else:
                center_text(d, txt, PX(.5), (ya + yb) / 2, pw * .92, (yb - ya) * .8, pt(11), True)
        if has_dem:
            y0r, y1r = PY(sep - .005), PY(eg[2])
            rh = (y1r - y0r) / 6
            put_text(d, (PX(.5), y0r + rh / 2), "Elevation in (m)", get_font(pt(12), True), "mm")
            for i, (c, lb) in enumerate(zip(DEM_PAL, dem_labels(edges))):
                cy_ = y0r + rh * (i + 1.5)
                d.rectangle([PX(.1), cy_ - rh * .3, PX(.28), cy_ + rh * .3], fill=c, outline="black", width=pt(.8))
                put_text(d, (PX(.34), cy_), lb, f_leg, "lm")

    progress(0.9, f"Encoding PNG at {dpi} DPI..." + (f" (skipped: {', '.join(skipped)})" if skipped else ""))
    tmp = out_path + ".tmp"
    img.save(tmp, format="PNG", compress_level=3, dpi=(dpi, dpi))
    del img
    gc.collect()
    os.replace(tmp, out_path)
    return skipped


def make_progress(job_dir):
    def progress(frac, msg):
        p = os.path.join(job_dir, "progress.json")
        try:
            with open(p + ".tmp", "w", encoding="utf-8") as f:
                json.dump({"progress": frac, "msg": msg}, f)
            for a in range(8):
                try:
                    os.replace(p + ".tmp", p)
                    return
                except PermissionError:
                    time.sleep(.05 * (a + 1))
        except OSError:
            pass
    return progress


def worker_main(job_dir):
    progress = make_progress(job_dir)

    def read(name, mode="rb"):
        p = os.path.join(job_dir, name)
        if os.path.exists(p):
            with open(p, mode, **({"encoding": "utf-8"} if "b" not in mode else {})) as f:
                return f.read()

    cfg = json.loads(read("input.json", "r"))
    _S.sig.update(cfg.get("sig") or {})
    for a, v, b in cfg.get("bounds") or []:
        _S.bounds[(a, v)] = tuple(b)
    ex = {"base": cfg.get("base"), "logo": read("logo.bin")}
    if os.path.exists(os.path.join(job_dir, "dem.npz")):
        with np.load(os.path.join(job_dir, "dem.npz")) as nz:
            ex["dem"] = (nz["z"], tuple(nz["g"]), nz["edges"].tolist())
    for k in ("contours", "roads", "water"):
        txt = read(k + ".json", "r")
        if txt:
            ex[k] = json.loads(txt)
    progress(0.0, "Starting export process...")
    skipped = export_map(cfg["sheets_pm"], cfg["layers"], cfg["title"], cfg["sheets_str"], cfg["dpi"], read("legend.png"),
                         progress, os.path.join(job_dir, "result.png"), cfg["kind"], ex)
    if skipped:
        with open(os.path.join(job_dir, "warn.txt"), "w") as f:
            f.write(", ".join(skipped))


# =====================================================================================
# GEOREFERENCING WORKER
# =====================================================================================
def geo_corners(cv2, rgb, dpi):
    mask = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), (120, 50, 50), (170, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    best, amax = None, 10000
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        a = cv2.contourArea(c)
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True) if a > amax else ()
        if len(ap) == 4:
            best, amax = ap, a
    if best is None:
        raise RuntimeError("grid not detected (magenta neatline frame not found in the PDF)")
    s = best.reshape(-1, 2).astype(np.float32)
    s = s[s[:, 1].argsort()]
    tl, tr = s[:2][s[:2][:, 0].argsort()]
    bl, br = s[2:][s[2:][:, 0].argsort()]
    base = np.array([tl, tr, br, bl])
    inset = (np.array([2100, 1102]) - np.array([1922, 924])) * (dpi / 300.0)
    v1, v2 = base[1] - base[0], base[3] - base[0]
    n1, n2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    return [base[0] + inset[0] * n1 + inset[1] * n2, base[1] - inset[0] * n1 + inset[1] * n2,
            base[2] - inset[0] * n1 - inset[1] * n2, base[3] + inset[0] * n1 - inset[1] * n2]


def geo_tif(cv2, img, M, b, path):
    minx, miny, maxx, maxy = [float(v) for v in b]
    w = int(max(1, round(float((maxx - minx) * math.hypot(M[0, 0], M[0, 1])))))
    h = int(max(1, round(float((maxy - miny) * math.hypot(M[1, 0], M[1, 1])))))
    dx, dy = (maxx - minx) / w, (maxy - miny) / h
    lon0, lat0 = minx + dx / 2, maxy - dy / 2
    A = np.ascontiguousarray([[M[0, 0] * dx, -M[1, 0] * dy, M[0, 0] * lon0 + M[1, 0] * lat0 + M[2, 0]],
                              [M[0, 1] * dx, -M[1, 1] * dy, M[0, 1] * lon0 + M[1, 1] * lat0 + M[2, 1]]], np.float64)
    fl = cv2.WARP_INVERSE_MAP
    img = np.ascontiguousarray(img)
    out = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_CUBIC | fl, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    msk = cv2.warpAffine(np.full(img.shape[:2], 255, np.uint8), A, (w, h), flags=cv2.INTER_NEAREST | fl,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    keys = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
    ensure("tifffile", "tifffile").imwrite(
        path, np.ascontiguousarray(out), photometric="rgb", compression="deflate",
        extratags=[(33550, "d", 3, (dx, dy, 0.0), True), (33922, "d", 6, (0.0, 0.0, 0.0, minx, maxy, 0.0), True),
                   (34735, "H", len(keys), keys, True), (42113, "s", 2, "0", True)])
    return out, msk


def geo_tiles(cv2, img, M, b, pm_path, prog):
    from pmtiles.writer import write
    from pmtiles.tile import zxy_to_tileid, TileType, Compression
    minx, miny, maxx, maxy = b
    lat_c = math.radians((miny + maxy) / 2)
    res_src = (111320 * math.cos(lat_c) / math.hypot(M[0, 0], M[0, 1]) + 110574 / math.hypot(M[1, 0], M[1, 1])) / 2
    zlo = GEO_MIN_ZOOM
    zmax = int(min(20, max(zlo, round(math.log2(156543.03392 * math.cos(lat_c) / res_src)))))
    f = 156543.03392 * math.cos(lat_c) / 2 ** zmax / res_src
    src, sx, sy = img, 1.0, 1.0
    if f > 1.05:
        src = cv2.resize(img, (max(1, round(img.shape[1] / f)), max(1, round(img.shape[0] / f))), interpolation=cv2.INTER_AREA)
        sx, sy = src.shape[1] / img.shape[1], src.shape[0] / img.shape[0]
    n, x0, x1, y0, y1 = tile_range(b, zmax)
    u = (np.arange(256) + .5) / 256
    sharp = ImageFilter.UnsharpMask(radius=.8, percent=70, threshold=3)

    def make_tile(x, y):
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        LON, LAT = np.meshgrid((x + u) / n * 360 - 180, np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + u) / n)))))
        inside = (LON >= minx) & (LON <= maxx) & (LAT >= miny) & (LAT <= maxy)
        if not inside.any():
            return None
        px = ((LON * M[0, 0] + LAT * M[1, 0] + M[2, 0] + .5) * sx - .5).astype(np.float32)
        py = ((LON * M[0, 1] + LAT * M[1, 1] + M[2, 1] + .5) * sy - .5).astype(np.float32)
        rgb = cv2.remap(src, px, py, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return Image.fromarray(np.dstack([rgb, (inside * 255).astype(np.uint8)]), "RGBA")

    def enc(im):
        if im.getchannel("A").getextrema()[0] == 255:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "WEBP", quality=GEO_QUALITY, method=2, **({"alpha_quality": 100} if im.mode == "RGBA" else {}))
        return buf.getvalue()

    def shrink(kids):
        cv = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        for (dx, dy), im in kids.items():
            cv.paste(im, (dx * 256, dy * 256))
        r, g, bl, a = cv.resize((256, 256), Image.Resampling.LANCZOS).split()
        r, g, bl = Image.merge("RGB", (r, g, bl)).filter(sharp).split()
        return Image.merge("RGBA", (r, g, bl, a))

    def group(pxy):
        kids = {(dx, dy): t for dy in (0, 1) for dx in (0, 1)
                if (t := make_tile(2 * pxy[0] + dx, 2 * pxy[1] + dy)) is not None}
        return [(2 * pxy[0] + dx, 2 * pxy[1] + dy, enc(t)) for (dx, dy), t in kids.items()], (shrink(kids) if kids else None)

    tiles, level = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, min(os.cpu_count() or 4, 8))) as ex:
        pxs = sorted({(x >> 1, y >> 1) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)})
        for i, (pxy, (encd, parent)) in enumerate(zip(pxs, ex.map(group, pxs)), 1):
            tiles.update({zxy_to_tileid(zmax, x, y): data for x, y, data in encd})
            if parent is not None:
                level[pxy] = parent
            prog(.8 * i / len(pxs), f"cutting tiles z{zmax}")
        for z in range(zmax - 1, zlo - 1, -1):
            items = list(level.items())
            for (x, y), data in zip((k for k, _ in items), ex.map(enc, (v for _, v in items))):
                tiles[zxy_to_tileid(z, x, y)] = data
            if z > zlo:
                grp = defaultdict(dict)
                for (x, y), im in items:
                    grp[(x >> 1, y >> 1)][(x & 1, y & 1)] = im
                level = dict(zip(grp, ex.map(shrink, grp.values())))
            prog(.8 + .15 * (zmax - z) / max(1, zmax - zlo), f"building overview z{z}")

    prog(.96, "writing tiles")
    e7 = lambda v: int(round(v * 1e7))
    with write(pm_path) as w:
        for tid in sorted(tiles):
            w.write_tile(tid, tiles[tid])
        w.finalize({"tile_type": TileType.WEBP, "tile_compression": Compression.NONE, "min_zoom": zlo, "max_zoom": zmax,
                    "min_lon_e7": e7(minx), "min_lat_e7": e7(miny), "max_lon_e7": e7(maxx), "max_lat_e7": e7(maxy),
                    "center_zoom": zmax, "center_lon_e7": e7((minx + maxx) / 2), "center_lat_e7": e7((miny + maxy) / 2)},
                   {"name": os.path.basename(pm_path), "format": "webp", "type": "overlay",
                    "minzoom": str(zlo), "maxzoom": str(zmax), "bounds": f"{minx},{miny},{maxx},{maxy}"})
    return zlo, zmax, len(tiles)


def save_preview(cv2, img, path):
    k = min(1.0, PREVIEW_PX / max(img.shape[:2]))
    pv = cv2.resize(img, (max(1, round(img.shape[1] * k)), max(1, round(img.shape[0] * k))), interpolation=cv2.INTER_AREA)
    Image.fromarray(pv).save(path, "JPEG", quality=82)


def tif_read(path):
    with ensure("tifffile", "tifffile").TiffFile(path) as t:
        tg = t.pages[0].tags
        sc, tie, gk = tg.get(33550), tg.get(33922), tg.get(34735)
        if sc is None or tie is None:
            raise RuntimeError("not a georeferenced GeoTIFF (geo tags missing)")
        v = [int(x) for x in gk.value] if gk else [0, 0, 0, 0]
        keys = {v[k]: v[k + 3] for k in range(4, 4 + 4 * v[3], 4)}
        if keys.get(2048) != 4326:
            raise RuntimeError("GeoTIFF must be in EPSG:4326 (WGS84 lat/lon)")
        dx, dy = float(sc.value[0]), float(sc.value[1])
        i, j, _, x0, y0, _ = [float(x) for x in tie.value[:6]]
        a = t.asarray()
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    elif a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = (a >> 8).astype(np.uint8) if a.dtype == np.uint16 else \
            (a.astype(np.float32) * 255 / max(float(a.max()), 1)).astype(np.uint8)
    h, w = a.shape[:2]
    minx, maxy = x0 - i * dx, y0 + j * dy
    if keys.get(1025) == 2:
        minx, maxy = minx - dx / 2, maxy + dy / 2
    return np.ascontiguousarray(a), (minx, maxy - h * dy, minx + w * dx, maxy)


def tif_one(cv2, path, name, out_dir, prog):
    prog(.05, "reading GeoTIFF")
    img, b = tif_read(path)
    minx, miny, maxx, maxy = b
    h, w = img.shape[:2]
    M = np.array([[w / (maxx - minx), 0], [0, -h / (maxy - miny)], [-minx * w / (maxx - minx), maxy * h / (maxy - miny)]], float)
    save_preview(cv2, img, os.path.join(out_dir, name + ".jpg"))
    pm = os.path.join(out_dir, name + ".pmtiles")
    zlo, zmax, nt = geo_tiles(cv2, img, M, b, pm, lambda fr, m: prog(.1 + .88 * fr, m))
    return {"name": name, "pb": [[miny, minx], [maxy, maxx]], "zoom": [zlo, zmax], "tiles": nt,
            "tif_mb": None, "pm_mb": round(os.path.getsize(pm) / 1e6, 1)}


def georef_one(cv2, pdf_path, name, b, out_dir, dpi, prog):
    fz = ensure("pymupdf", "pymupdf")
    prog(.02, "rendering PDF")
    with fz.open(pdf_path) as doc:
        pix = doc.load_page(0).get_pixmap(matrix=fz.Matrix(dpi / 72, dpi / 72), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
    prog(.1, "detecting grid")
    pts = geo_corners(cv2, img, dpi)
    minx, miny, maxx, maxy = b
    ll = np.array([(minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)], float)
    M = cv2.solve(np.column_stack([ll, np.ones(4)]), np.array(pts, np.float64), flags=cv2.DECOMP_SVD)[1]
    prog(.15, "writing GeoTIFF")
    tif, pm = os.path.join(out_dir, name + ".tif"), os.path.join(out_dir, name + ".pmtiles")
    out, msk = geo_tif(cv2, img, M, b, tif)
    save_preview(cv2, out, os.path.join(out_dir, name + ".jpg"))
    del out, msk
    prog(.25, "cutting tiles")
    zlo, zmax, nt = geo_tiles(cv2, img, M, b, pm, lambda fr, m: prog(.25 + .73 * fr, m))
    return {"name": name, "pb": [[miny, minx], [maxy, maxx]], "zoom": [zlo, zmax], "tiles": nt,
            "tif_mb": round(os.path.getsize(tif) / 1e6, 1), "pm_mb": round(os.path.getsize(pm) / 1e6, 1)}


def georef_main(job_dir):
    progress = make_progress(job_dir)
    with open(os.path.join(job_dir, "input.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    progress(0.0, "Loading libraries...")
    cv2, results, items = ensure("cv2", "opencv-python-headless"), [], cfg["items"]
    for i, it in enumerate(items):
        lo, span = i / len(items), 1 / len(items)
        try:
            p, cb = os.path.join(job_dir, it["file"]), lambda fr, m, lo=lo, span=span, nm=it["name"]: progress(lo + span * fr, f"{nm}: {m}")
            results.append(tif_one(cv2, p, it["name"], job_dir, cb) if it["file"].endswith(".tif")
                           else georef_one(cv2, p, it["name"], it["bounds"], job_dir, cfg["dpi"], cb))
        except Exception as e:
            results.append({"name": it["name"], "error": f"{type(e).__name__}: {e}"})
        gc.collect()
    with open(os.path.join(job_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f)
    progress(1.0, "Done")


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] in ("--export", "--georef"):
    faulthandler.enable()
    (worker_main if sys.argv[1] == "--export" else georef_main)(sys.argv[2])
    sys.exit(0)


# =====================================================================================
# STREAMLIT APP
# =====================================================================================
import streamlit as st
import streamlit.components.v1 as components

if not TOKEN:
    try:
        TOKEN = st.secrets.get("HF_TOKEN")
    except Exception:
        TOKEN = None
    if TOKEN:
        session.headers.update({"Authorization": f"Bearer {TOKEN}"})
if not R2_BASE:
    try:
        R2_BASE = (st.secrets.get("R2_BASE") or "").rstrip("/")
    except Exception:
        pass

st.set_page_config(page_title=APP_TITLE, page_icon="🛰️", layout="wide", initial_sidebar_state="collapsed")

if not st.session_state.get("auth"):
    st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 18% 12%,rgba(138,167,123,.32),transparent 45%),
  radial-gradient(circle at 85% 88%,rgba(47,93,70,.40),transparent 42%),#14231b!important}
.block-container{max-width:480px!important;margin:0 auto!important;padding-top:17vh!important}
[data-testid="stForm"]{background:rgba(255,255,255,.09);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.28);
  border-radius:22px;padding:32px 30px 24px;box-shadow:0 24px 70px rgba(0,0,0,.5)}
.lg-t{text-align:center;color:#fff;font-size:28px;font-weight:800}
.lg-s{text-align:center;color:#c7d8c4;font-size:12px;margin:4px 0 16px;line-height:1.5}
[data-testid="stForm"] label p{color:#e4eddc!important;font-weight:600}
[data-testid="stForm"] div[data-baseweb="input"],[data-testid="stForm"] div[data-baseweb="base-input"]{background:rgba(255,255,255,.96)!important;border-radius:10px}
[data-testid="stForm"] input{color:#10223d!important;-webkit-text-fill-color:#10223d!important}
[data-testid="stForm"] button[kind]{background:linear-gradient(90deg,#2f5d46,#6b8f5e);color:#fff;border:0;border-radius:10px;font-weight:700}
</style>""", unsafe_allow_html=True)
    with st.form("login"):
        st.markdown(f'<div class="lg-t">🛰️ Welcome</div><div class="lg-s">{APP_TITLE}<br>Sign in to continue</div>', unsafe_allow_html=True)
        u_in, p_in = st.text_input("Username"), st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", width="stretch"):
            if hmac.compare_digest(u_in.encode(), LOGIN_USER.encode()) and hmac.compare_digest(p_in.encode(), LOGIN_PASS.encode()):
                st.session_state["auth"] = True
                st.rerun()
            st.error("Invalid username or password")
    st.stop()

release_assets = st.cache_data(ttl=LIST_TTL, show_spinner=False)(list_assets)
terrain = st.cache_resource(show_spinner="Loading...", max_entries=3)(build_terrain)
road_data = st.cache_resource(show_spinner="Loading...", max_entries=4)(build_roads)
water_data = st.cache_resource(show_spinner="Loading...", max_entries=8)(build_water)
vs_data = st.cache_resource(show_spinner="Loading...", max_entries=2)(vantage_batches)

st.markdown("""<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap");
:root{--g:#2f5d46;--g2:#4f7f5f;--moss:#8aa77b;--sand:#f6f3ea;--sand2:#e9e4d3;--card:#fffdf7;--line:#d3ccb6;--ink:#26302a;--acc:#b7791f;--acc2:#975f14}
header[data-testid="stHeader"],section[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],[data-testid="stStatusWidget"]{display:none!important}
.stApp{background:var(--sand);color:var(--ink);font-family:Inter,"Segoe UI",system-ui,sans-serif}
footer,#MainMenu{visibility:hidden}
.block-container{max-width:1600px;padding:.9rem 1.5rem .5rem!important}
html,body,[class*="css"],.stMarkdown,p,span,label{font-size:12px!important}
.st-key-hdr{background:#fff;border:1px solid var(--line);border-top:4px solid var(--g);border-radius:12px;padding:6px 16px;margin-bottom:8px;box-shadow:0 2px 10px rgba(38,48,42,.08)}
.st-key-hdr [data-testid="stMarkdownContainer"],.st-key-hdr p{margin:0}
.st-key-hdr button{background:#fff;color:var(--g);border:1px solid var(--g);min-height:32px}
.st-key-hdr button:hover{background:var(--g);color:#fff}
.st-key-hdr div[data-baseweb="select"]>div{min-height:32px;font-size:11px}
.hd{display:flex;align-items:center;gap:12px}
.hd img{height:34px;width:auto;max-height:34px;object-fit:contain;flex:none}
.hd .ic{font-size:26px}
.hd b{display:block;color:var(--g);font-size:19px!important;font-weight:800;line-height:1.2}
.hd small{display:block;color:#6b705f;font-size:11px!important;line-height:1.3}
.pillw{display:flex;justify-content:center}
.pill{white-space:nowrap;background:#eef3ea;color:var(--g);border:1px solid var(--moss);border-radius:20px;padding:4px 14px;font-size:11.5px!important;font-weight:600}
/* ---- tabs: one compact line, Contribute pinned to the far right corner ---- */
.stTabs [data-baseweb="tab-list"]{display:flex;flex-wrap:nowrap;width:100%;gap:2px;border-bottom:2px solid var(--moss);overflow-x:auto;scrollbar-width:thin}
.stTabs [data-baseweb="tab"]{flex:none;height:32px;padding:0 9px;border-radius:8px 8px 0 0;color:#5d6657;white-space:nowrap}
.stTabs [data-baseweb="tab"] p{font-size:11.5px!important}
.stTabs [data-baseweb="tab"]:hover{background:var(--sand2)}
.stTabs [aria-selected="true"]{background:var(--g)}
.stTabs [aria-selected="true"] p{color:#fff!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab-panel"]{background:var(--card);border:1px solid var(--line);border-top:0;border-radius:0 0 10px 10px;padding:12px 14px}
.stTabs [role="tablist"],.stTabs [role="tablist"] :has(>[role="tab"]){display:flex;flex-wrap:nowrap;width:100%}
.stTabs [role="tab"]:last-of-type{margin-left:auto!important;background:#7c3aed;box-shadow:0 2px 8px rgba(124,58,237,.4)}
.stTabs [role="tab"]:last-of-type p{color:#fff!important;font-weight:700}
.stTabs [role="tab"]:last-of-type[aria-selected="true"]{background:#4c1d95}
/* ---- small buffer colour pickers, tight spacing ---- */
[data-testid="stColorPickerBlock"]{width:20px!important;height:20px!important;border-radius:50%!important}
[data-testid="stColorPicker"]{width:auto!important}
.st-key-pk1 [data-testid="stHorizontalBlock"],.st-key-pk2 [data-testid="stHorizontalBlock"]{gap:.2rem!important;flex-wrap:nowrap!important}
.st-key-pk1 [data-testid="stColumn"],.st-key-pk2 [data-testid="stColumn"]{flex:0 0 auto!important;width:auto!important;min-width:0!important}
.stButton>button,.stDownloadButton>button{background:var(--g);color:#fff;border:0;border-radius:8px;font-size:12px!important;font-weight:600!important;transition:.15s}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--g2);color:#fff;box-shadow:0 3px 8px rgba(47,93,70,.25)}
.stButton>button[data-testid="stBaseButton-primary"],.stButton>button[kind="primary"]{background:var(--acc)}
.stButton>button[data-testid="stBaseButton-primary"]:hover,.stButton>button[kind="primary"]:hover{background:var(--acc2);box-shadow:0 3px 8px rgba(183,121,31,.3)}
.stButton>button:disabled,.stDownloadButton>button:disabled{background:#c9c4b0;color:#8d8a7a}
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"]{background:#fff!important;border-color:var(--line)!important;border-radius:8px}
[data-testid="stFileUploaderDropzone"]{background:#fff;border:1.5px dashed var(--moss);border-radius:10px}
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border-color:var(--line)!important;border-radius:10px;box-shadow:0 1px 3px rgba(38,48,42,.08)}
[data-testid="stProgress"]>div>div>div{background:var(--g)!important}
[data-testid="stProgress"]>div>div{background:var(--sand2)!important}
.lg{display:flex;align-items:center;gap:4px;margin:7px 0}
.lg i{display:inline-block;width:22px;height:10px;box-sizing:border-box;flex:none}
.custom-footer{text-align:center;color:#7a7d6c;font-size:10px;padding:6px 0;border-top:1px solid var(--line);margin-top:10px}
.custom-footer a{color:var(--g);text-decoration:none}
.sec{font-weight:700;font-size:14px!important;color:var(--g);border-left:4px solid var(--acc);padding-left:8px;margin:6px 0}
@media(max-width:700px){.block-container{padding:.4rem .5rem!important}.hd small{display:none}.hd b{font-size:15px!important}
  .pillw{justify-content:flex-start}.stTabs [data-baseweb="tab-panel"]{padding:8px}}
</style>""", unsafe_allow_html=True)

with st.container(key="hdr"):
    _h1, _h2, _h3, _h4 = st.columns([4.6, 2.4, 1.7, 1.0], vertical_alignment="center")
    hdr_slot, pill_slot = _h1.empty(), _h2.empty()
    basemap = _h3.selectbox("Base map", list(BASEMAPS), index=2, label_visibility="collapsed", format_func=lambda k: "🗺️ " + k)
    if _h4.button("🚪 Log out", width="stretch"):
        st.session_state["auth"] = False
        st.rerun()


def load_vector(name, data):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, os.path.basename(name))
        with open(p, "wb") as fh:
            fh.write(data)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as zf:
                zf.extractall(d)
            p = next((os.path.join(r, x) for r, _, fs in os.walk(d) for x in fs if x.lower().endswith(".shp")), None)
            if not p:
                return None
        g = gpd.read_file(p)
    return g.set_crs(4326) if g.crs is None else g.to_crs(4326)


@st.cache_resource(show_spinner="Loading...", max_entries=6)
def layer_for(name, data, bm, labels, dash, cols):
    g = load_vector(name, data)
    if g is None:
        raise ValueError("No .shp found inside the zip.")
    l = make_layer(g, list(labels), list(bm), dash)
    l["cols"], l["key"] = list(cols), f"{name}|{len(data)}|{bm}|{labels}|{dash}|{cols}"
    return l


@st.cache_resource(show_spinner=False, max_entries=2)
def dem_web(wkb):
    z, g, edges = terrain(wkb)[:3]
    im, e = dem_rgba(z, g, edges)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    (lo0, la0), (lo1, la1) = unmerc(e[0], e[2]), unmerc(e[1], e[3])
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), [[la0, lo0], [la1, lo1]]


@st.cache_resource(show_spinner=False, max_entries=2)
def dem_tif_bytes(wkb):
    z, g = terrain(wkb)[:2]
    return geotiff_bytes(np.where(np.isnan(z), -9999, z).astype("float32"), g, -9999)


@st.cache_resource(show_spinner=False)
def _memo_store():
    return {}


def memo(key, fn):
    s = _memo_store()
    if key not in s:
        if len(s) > 40:
            s.clear()
        s[key] = fn()
    return s[key]


def sticky(key, fn):
    """Like memo, but a failed (None) result is retried on the next rerun."""
    s = _memo_store()
    if s.get(key) is None:
        v = fn()
        if v is None:
            return None
        s[key] = v
    return s[key]


def show_html(html, height):
    if hasattr(st, "iframe"):
        try:
            st.iframe(html, height=height)
            return
        except Exception:
            pass
    components.html(html, height=height)


def find_asset(assets, stem, ext):
    return next((v for k, v in assets.items() if k.lower().endswith(ext) and stem_of(k) == norm(stem)), None)


@st.cache_resource
def job_store():
    return {}


@st.cache_resource
def georef_store():
    return {}


JOBS, GEO = job_store(), georef_store()


def first_bytes(assets, keys, local):
    for k in keys:
        try:
            code, _, body = fetch_range(assets[k])
            if code == 200 and body:
                return body
        except Exception:
            pass
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(here, local), local, "assets/" + local]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()


legend_image = lambda assets: first_bytes(
    assets, [k for k in assets if "legend_toposheet" in k.lower() or "legend_topsheet" in k.lower()], "legend_toposheet.png")


def logo_image(assets):
    keys = [k for k in assets if k.lower().endswith(("logo.jpg", "logo.jpeg"))]
    root = [k for k in keys if not assets[k].startswith("contributions/")]
    contrib = sorted((k for k in keys if assets[k].startswith("contributions/")), reverse=True)
    return first_bytes(assets, root + contrib, "logo.jpg")


def read_osm_index(v):
    a = release_assets()
    if "OSM_SHEET_INDEX.shp" not in a:
        raise FileNotFoundError("OSM_SHEET_INDEX.shp not found in the HF repo listing")
    with tempfile.TemporaryDirectory() as d:
        for ext in ("shp", "shx", "dbf", "prj", "cpg"):
            if f"OSM_SHEET_INDEX.{ext}" in a:
                with open(os.path.join(d, f"OSM_SHEET_INDEX.{ext}"), "wb") as f:
                    f.write(fetch_range(a[f"OSM_SHEET_INDEX.{ext}"])[2])
        os.environ["SHAPE_RESTORE_SHX"] = "YES"
        g = gpd.read_file(os.path.join(d, "OSM_SHEET_INDEX.shp"))
    return g.set_crs(4326) if g.crs is None else g.to_crs(4326)


@st.cache_resource(show_spinner="Loading...", max_entries=1)
def osm_geojson(v):
    g = read_osm_index(v)
    g["geometry"] = g.geometry.simplify(0.0005)
    return json.loads(g.to_json(default=str))


@st.cache_resource(show_spinner="Loading...", max_entries=1)
def osm_sheet_table(v):
    g = read_osm_index(v)
    col = "OSM_SHEET_" if "OSM_SHEET_" in g.columns else next(c for c in g.columns if c != "geometry" and g[c].dtype == object)
    out = {}
    for nm, r in zip(g[col].astype(str), g.geometry.bounds.itertuples(index=False)):
        out.setdefault(re.sub(r"\s+", "", nm).upper(), (float(r.minx), float(r.miny), float(r.maxx), float(r.maxy)))
    return out


def match_sheet(filename, table):
    return next(((c.upper(), table[c.upper()]) for c in re.findall(r"[A-Za-z0-9]+", os.path.splitext(filename)[0])
                 if c.upper() in table), None)

fmt_sheet = lambda k: re.sub(r"^(\d{1,3}[A-P])0*(\d+)$", r"\1/\2", k)   # 56G05 -> 56G/5; unchanged if the name has another pattern
def sheet_no(name):
    s = re.findall(r"\d{1,3}[A-P]/?\d{1,2}", name, re.I)
    if not s:
        return None
    s = s[0].upper()
    return s if "/" in s else re.sub(r"([A-P])(\d+)", r"\1/\2", s)


def describe_exit(rc):
    if rc < 0:
        return f"killed by signal {-rc}" + (" (SIGKILL - usually out of memory)" if rc == -9 else "")
    code = rc & 0xFFFFFFFF
    known = {0xC0000005: "access violation (native crash)", 0xC0000017: "not enough memory", 0xC0000135: "DLL not found",
             0xC00000FD: "stack overflow", 0xC0000409: "native stack buffer overrun",
             0xC06D007F: "DLL function not found (library version conflict)"}
    return f"{known[code]} [0x{code:08X}]" if code in known else f"exit code {rc}"


def wtg_points(layers):
    out = []
    for l in layers:
        for f in l["feats"]:
            g = shape(f["geom"])
            out += [(round(q.x, 6), round(q.y, 6), feat_name(f["props"], f"WTG {len(out) + 1}"))
                    for q in getattr(g, "geoms", [g]) if q.geom_type == "Point"]
    return tuple(out)


# ---------------- background jobs ----------------
def spawn(mode, d):
    env = dict(os.environ, **({"HF_TOKEN": TOKEN} if TOKEN else {}), **({"R2_BASE": R2_BASE} if R2_BASE else {}))
    with open(os.path.join(d, "log.txt"), "wb") as lg:
        return subprocess.Popen([sys.executable, os.path.abspath(__file__), mode, d], env=env, stdout=lg,
                                stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def new_job(mode, d, msg, **extra):
    return {"status": "running", "progress": 0.0, "msg": msg, "start": time.time(), "end": None, "error": None,
            "dir": d, "proc": spawn(mode, d), **extra}


def poll(job, ok_file, on_ok):
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
    if rc == 0 and os.path.exists(os.path.join(d, ok_file)):
        on_ok(job)
        job.update(status="done", progress=1.0, msg="Done")
    else:
        try:
            with open(os.path.join(d, "log.txt"), encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-12:]).strip()
        except Exception:
            tail = ""
        job.update(status="error", error=f"{describe_exit(rc)} | last step: {job.get('msg')}" + (f"\n{tail}" if tail else ""))
        shutil.rmtree(d, ignore_errors=True)


def kill_job(job, status=None):
    if not job:
        return
    try:
        if job["status"] == "running":
            job["proc"].terminate()
            job["proc"].wait(timeout=5)
    except Exception:
        pass
    if status:
        job.update(status=status, end=time.time())
    shutil.rmtree(job["dir"], ignore_errors=True)


def start_export(sheets_pm, layers, title, sheets_str, dpi, legend_bytes, kind, extra):
    d = tempfile.mkdtemp(prefix="topo_export_")
    with open(os.path.join(d, "input.json"), "w", encoding="utf-8") as f:
        json.dump({"sheets_pm": sheets_pm, "layers": [light(l) for l in layers], "title": title, "sheets_str": sheets_str,
                   "dpi": dpi, "kind": kind, "base": extra.get("base"), "sig": dict(_S.sig),
                   "bounds": [[k[0], k[1], list(v)] for k, v in _S.bounds.items()]}, f)
    for name, data in (("legend.png", legend_bytes), ("logo.bin", extra.get("logo"))):
        if data:
            with open(os.path.join(d, name), "wb") as f:
                f.write(data)
    if extra.get("dem"):
        z, g, edges = extra["dem"]
        np.savez_compressed(os.path.join(d, "dem.npz"), z=z, g=np.array(g), edges=np.array(edges))
    for k in ("contours", "roads", "water"):
        if extra.get(k):
            with open(os.path.join(d, k + ".json"), "w", encoding="utf-8") as f:
                json.dump(extra[k], f)
    return new_job("--export", d, "Starting export process...", result=None, dpi=dpi, kind=kind)


def _export_done(job):
    with open(os.path.join(job["dir"], "result.png"), "rb") as f:
        job["result"] = f.read()
    try:
        with open(os.path.join(job["dir"], "warn.txt")) as f:
            job["warn"] = f.read()
    except Exception:
        pass
    shutil.rmtree(job["dir"], ignore_errors=True)


refresh_job = lambda job: poll(job, "result.png", _export_done)


def start_georef(items, dpi):
    d = tempfile.mkdtemp(prefix="topo_georef_")
    for i, it in enumerate(items):
        it["file"] = f"in_{i}{it.pop('ext', '.pdf')}"
        with open(os.path.join(d, it["file"]), "wb") as f:
            f.write(it.pop("data"))
    with open(os.path.join(d, "input.json"), "w", encoding="utf-8") as f:
        json.dump({"dpi": dpi, "items": items}, f)
    return new_job("--georef", d, "Starting...", results=[], push=None, skipped=[])


def _geo_done(job):
    d = job["dir"]
    with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
        job["results"] = json.load(f)
    for x in os.listdir(d):
        if x.startswith("in_") and x.endswith((".pdf", ".tif")):
            os.remove(os.path.join(d, x))


refresh_geo = lambda job: poll(job, "results.json", _geo_done)


def write_token():
    try:
        return os.getenv("HF_WRITE_TOKEN") or st.secrets.get("HF_WRITE_TOKEN") or TOKEN
    except Exception:
        return TOKEN


def hf_commit(pairs, wt, msg):
    api = ensure("huggingface_hub", "huggingface_hub hf_xet").HfApi(token=wt)
    for local, dest in pairs:
        for attempt in range(3):
            try:
                api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=HF_REPO, repo_type="dataset",
                                commit_message=f"{msg}: {os.path.basename(dest)}")
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))


def publish_static(gid, j):
    d = os.path.join(STATIC_GEO, gid)
    os.makedirs(d, exist_ok=True)
    for r in j["results"]:
        src, dst = os.path.join(j["dir"], r["name"] + ".pmtiles"), os.path.join(d, r["name"] + ".png")
        if "error" not in r and os.path.isfile(src) and not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copyfile(src, dst)


def drop_static(gid):
    if gid:
        shutil.rmtree(os.path.join(STATIC_GEO, gid), ignore_errors=True)


def push_bg(job, pairs, wt):
    job["push"] = {"status": "running"}

    def run():
        try:
            hf_commit(pairs, wt, "Georeferenced toposheet")
            job["push"] = {"status": "done", "paths": [d for _, d in pairs]}
        except Exception as e:
            job["push"] = {"status": "error", "msg": str(e)}
    threading.Thread(target=run, daemon=True).start()


def hf_upload(files, wt):
    ts, pairs, tmps = int(time.time()), [], []
    for f in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as t:
            t.write(f.getbuffer())
        tmps.append(t.name)
        pairs.append((t.name, f"contributions/{ts}_{re.sub(r'[^\w.-]', '_', f.name)}"))
    try:
        hf_commit(pairs, wt, "Contribution")
    finally:
        for t in tmps:
            try:
                os.remove(t)
            except OSError:
                pass
    return [d for _, d in pairs]


# ---------------- data + header ----------------
assets = {}
try:
    assets = release_assets()
except Exception as e:
    st.error(f"Hugging Face error: {e}")
st.session_state["hf_sig"] = hash(tuple(sorted(_S.sig.items())))


@st.fragment(run_every=WATCH_EVERY)
def watch_hf():
    try:
        release_assets()
    except Exception:
        return
    if hash(tuple(sorted(_S.sig.items()))) != st.session_state.get("hf_sig"):
        st.rerun()


watch_hf()
pm_assets = {n: i for n, i in assets.items() if n.lower().endswith(".pmtiles") and not is_vec(n)}

_ver = lambda *keys: tuple(sorted((k, _S.sig.get(p)) for k, p in assets.items() if any(w in k.lower() for w in keys)))
leg_bytes = sticky(("legend", _ver("legend_t")), lambda: legend_image(assets))
logo_bytes = sticky(("logo", _ver("logo")), lambda: logo_image(assets))
hdr_slot.markdown(
    '<div class="hd">' + (f'<img src="data:image/jpeg;base64,{base64.b64encode(logo_bytes).decode()}">' if logo_bytes
                          else '<span class="ic">🛰️</span>') +
    f'<div><b>{APP_TITLE}</b><small>Site boundaries &amp; buffer zones &middot; Toposheets &middot; DEM &amp; contours &middot; '
    'Roads &middot; Surface water &middot; Satellite &middot; Vantage points</small></div></div>', unsafe_allow_html=True)
pill_slot.markdown(f'<div class="pillw"><span class="pill">● {len(pm_assets)} toposheets in library</span></div>', unsafe_allow_html=True)

project_title = st.text_input(
    "Project Title",
    value="Preliminary Assessment Study for the 14MW Solar Project at Gadgi, Bidar District, Karnataka, India")

for _gid, _gj in list(GEO.items()):
    if _gj["status"] != "running" and _gj.get("end") and time.time() - _gj["end"] > GEO_TTL:
        kill_job(_gj)
        drop_static(_gid)
        GEO.pop(_gid, None)
gjob = GEO.get(st.session_state.get("geo_id"))
refresh_geo(gjob)
geo_busy = bool(gjob) and (gjob["status"] == "running" or (gjob.get("push") or {}).get("status") == "running")

# ---------------- controls ----------------
T = st.tabs(["📍 Site vectors", "🧭 Georeferencing", "🗾 Toposheet viewer", "💧 Drainage", "⛰️ DEM & Contour",
             "🛣️ Roads & Transportation", "🛰️ Satellite map", "🎯 Vantage-Point Selection", "🤝 Contribute"])
notes = st.container()
st.markdown('<div class="sec">Live map</div>', unsafe_allow_html=True)

with T[1]:
    c1, c2 = st.columns([1.4, 1])
    show_osm_index = c2.toggle("📐 Show OSM Sheet Index on the map", value=False)
    c2.caption("Hover a sheet on the map to see its name. A PDF file name must contain its sheet name.")
    with c1:
        geo_files = st.file_uploader("Toposheet PDF(s) or GeoTIFF(s)", type=["pdf", "tif", "tiff"], accept_multiple_files=True,
                                     key="geo_pdfs", disabled=geo_busy)
        if st.button("▶️ Start Georeferencing", type="primary", width="stretch", disabled=geo_busy or not geo_files, key="geo_start"):
            try:
                tbl = osm_sheet_table(_S.sig.get(assets.get("OSM_SHEET_INDEX.shp")))
            except Exception as e:
                tbl = {}
                st.error(f"Sheet index: {e}")
            items, bad, have, seen = [], [], [], set()
            for f in geo_files:
                is_tif = f.name.lower().endswith((".tif", ".tiff"))
                hit = None if is_tif else match_sheet(f.name, tbl)
                stem = re.sub(r"[^\w.-]", "_", os.path.splitext(f.name)[0])
                if not (hit or is_tif):
                    bad.append(f.name)
                elif stem + ".pmtiles" in pm_assets or stem in seen:
                    have.append(stem + ".pmtiles")
                else:
                    seen.add(stem)
                    items.append({"name": stem, "bounds": list(hit[1]) if hit else None, "data": f.getvalue(),
                                  "ext": ".tif" if is_tif else ".pdf"})
            if have:
                st.session_state["geo_select"] = list(set(st.session_state.get("geo_select", [])) | set(have))
            if items:
                _old = st.session_state.get("geo_id")
                kill_job(GEO.pop(_old, None))
                drop_static(_old)
                gid = uuid.uuid4().hex
                GEO[gid] = start_georef(items, GEO_DPI)
                GEO[gid]["skipped"] = bad
                st.session_state["geo_id"], st.session_state["geo_poll"] = gid, True
                st.rerun()
            elif bad:
                st.error("No PDF name matches a sheet: " + ", ".join(bad))
            else:
                st.rerun()

with T[2]:
    _pend = st.session_state.get("geo_select", [])
    if any(n in pm_assets for n in _pend):
        st.session_state["chosen"] = sorted(set(st.session_state.get("chosen", [])) | {n for n in _pend if n in pm_assets})
        st.session_state["geo_select"] = [n for n in _pend if n not in pm_assets]
    b1, b2, b3 = st.columns([1, 1, 3])
    if b1.button("Select all", width="stretch"):
        st.session_state["chosen"] = list(pm_assets)
    if b2.button("Clear", width="stretch"):
        st.session_state["chosen"] = []
    show_topo = b3.toggle("🗾 Show topo layer on the map", value=True)
    chosen = st.multiselect(f"Toposheets in the database ({len(pm_assets)})", list(pm_assets), key="chosen")

with T[3]:
    show_dr = {k: col.toggle(f"💧 {v[0]}", value=False) for col, (k, v) in zip(st.columns(4), DRAIN.items())}
    st.caption("Drainage layers from the Bhuvan LULC database, clipped to the outermost buffer. Needs a site vector + buffer.")

with T[4]:
    c = st.columns(2)
    show_dem = c[0].toggle("⛰️ Show DEM (Elevation)", value=False)
    show_con = c[1].toggle(f"〰️ Show Contours ({CONTOUR_STEP} m, lines ≥ {CONTOUR_MIN_LEN} m long)", value=False)
    st.caption("NASA SRTM 30 m elevation, clipped to the outermost buffer. Needs a site vector + buffer.")

with T[5]:
    show_roads = st.toggle("🛣️ Show Roads & Transportation", value=False)
    st.caption("Survey of India roads, clipped to the outermost buffer. Needs a site vector + buffer.")

with T[6]:
    st.markdown("**Satellite map**: Google Satellite imagery inside the outermost buffer, white outside, with the site and buffers on top.")
    st.caption("Choose *Satellite Map* under Export Map once a site vector is loaded. For the live map use the Base map selector in the header.")

layers, sig = [], []
with T[0]:
    for i, col in zip((1, 2), st.columns(2)):
        with col:
            vf = st.file_uploader(f"Vector {i} (.geojson, .kml, .zip)", type=["geojson", "kml", "gpx", "zip"], key=f"vec{i}")
            tb, pkc = st.columns([6, 1.4], vertical_alignment="bottom")
            bt = tb.text_input(f"Vector {i} buffers (e.g. 2km, 5km, 10km)", "2km, 5km, 10km" if i == 1 else "2km", key=f"buf{i}")
            with pkc, st.container(key=f"pk{i}"):   # three small colour dots, tight spacing
                cols_ = [p.color_picker("c", COLORS[k], key=f"bc{i}{k}", label_visibility="collapsed",
                                        help=("Inner", "Middle", "Outer")[k] + " buffer colour")
                         for k, p in enumerate(st.columns(3))]
            bm = parse_buffers(bt)
            dl = default_labels(i, bm)
            lt = st.text_input(f"Vector {i} legend text (comma separated: site, then each buffer)", ", ".join(dl), key=f"lab{i}_{bt}")
            lab = [x.strip() for x in lt.split(",")]
            labels = [lab[k] if k < len(lab) and lab[k] else d for k, d in enumerate(dl)]
            if vf:
                try:
                    layers.append(layer_for(vf.name, vf.getvalue(), tuple(bm), tuple(labels), i == 2, tuple(cols_)))
                    sig.append(vf.name)
                except Exception as e:
                    st.error(f"Vector {i} error: {e}")

# ---- vantage-point selection for the WTGs (point features of Vector 1 / 2) ----
wtg, vs_res, vs_on, vs_args = wtg_points(layers), None, [], None
with T[7]:
    v1, v2, v3 = st.columns(3)
    vs_h = v1.number_input("Observer height (m)", 0.0, 100.0, 1.6, 0.1)
    vs_t = v2.number_input("Target height (m)", 0.0, 300.0, 0.0, 1.0)
    vs_d = v3.number_input("Search radius = buffer around each WTG (m)", 100.0, 10000.0, 2000.0, 100.0)
    st.caption("Vantage points for the point features (WTGs) of Vector 1 / 2 on SRTM 30 m terrain. The map and table show Batch 1; "
               "all 3 batches (KML + GeoTIFF) are in Deliverables below the map. Click a vantage point to see which WTGs it covers.")
    if st.button("▶️ Run vantage-point selection", type="primary", width="stretch"):
        if wtg:
            st.session_state["vs_args"] = (wtg, vs_h, vs_t, vs_d)
        else:
            st.warning("Upload a point vector (WTG locations) in the Site vectors tab first.")
    vs_args = st.session_state.get("vs_args")
    if vs_args:
        try:
            vs_res = vs_data(*vs_args)
        except Exception as e:
            st.error(f"Vantage-point error: {e}")
        if vs_res:
            if vs_args[0] != wtg:
                st.info("The point vectors changed since this run - run the analysis again to update.")
            tot, b_ = len(vs_args[0]), vs_res["batches"][0]   # only Batch 1 is shown
            st.caption(f"**Batch 1:** {len(b_['pts'])} vantage points cover {tot - b_['left']} of {tot} WTGs"
                       + (f" - not covered: {b_['missed']}" if b_["missed"] else ""))
            st.dataframe([{"Vantage": f"Vantage {k}", "Lon": round(q["lon"], 5), "Lat": round(q["lat"], 5),
                           "Elev (m)": round(q["elev_m"]), "WTGs": q["wtg_count"], "WTGs covered": q["wtg_names"]}
                          for k, q in enumerate(b_["pts"], 1)], hide_index=True, width="stretch")
            vs_on = [st.toggle("👁️ Visibility raster - Batch 1", key="vson1")]
            st.caption("Visibility raster: the darker the colour, the more WTGs are visible from that cell.")
vs_b = vs_res["batches"][:1] if vs_res else []   # map / legend / table: Batch 1 only (Deliverables export all batches)

with T[8]:
    cfs = st.file_uploader("Any file: toposheet, GeoTIFF, vector, PNG, PDF, ...", accept_multiple_files=True, key="contrib")
    if cfs and st.button("Submit contribution"):
        wt = write_token()
        if not wt:
            st.error("No write token: set HF_WRITE_TOKEN (env or Streamlit secret).")
        else:
            try:
                with st.spinner("Uploading (large files can take a few minutes)..."):
                    up = hf_upload(cfs, wt)
                release_assets.clear()
                st.success("Uploaded to contributions/: " + ", ".join(os.path.basename(p) for p in up))
            except Exception as e:
                st.error(f"Upload failed: {e}")

osm_json = None
if show_osm_index:
    try:
        osm_json = osm_geojson(_S.sig.get(assets.get("OSM_SHEET_INDEX.shp")))
    except Exception as e:
        notes.error(f"OSM index: {e}")

outer = [shape(f["geometry"]) for l in layers for f in (l["bufs"][-1]["gj"] if l["bufs"] else l["site"])["features"]]
if outer and chosen:
    with ThreadPoolExecutor(max_workers=8) as _ex:
        list(_ex.map(sheet_bounds, [pm_assets[n] for n in chosen]))


def touches_buffer(name):
    b = sheet_bounds(pm_assets[name])
    return True if b is None else any(g.intersects(box(*b)) for g in outer)


sheets_str = ""
if outer:   # sheet numbers only when a vector is loaded: OSM index sheets that touch the outermost buffer
    try:
        _tbl = osm_sheet_table(_S.sig.get(assets.get("OSM_SHEET_INDEX.shp")))
        _u = unary_union(outer)
        sheets_str = ", ".join(fmt_sheet(k) for k, b in sorted(_tbl.items()) if _u.intersects(box(*b)))
    except Exception as e:
        notes.error(f"Sheet index: {e}")
sheets_str = sheets_str or "-"

ou_wkb = unary_union(outer).wkb if outer else None
okey = f"{zlib.crc32(ou_wkb)}:{len(ou_wkb)}" if ou_wkb else ""
terr = road_fc = dem_ov = None
water, rv, wv = {}, None, {}
if not ou_wkb and (show_roads or show_dem or show_con or any(show_dr.values())):
    notes.warning("Roads / water / DEM / contours need a vector (Vector 1 or 2) with a buffer. Upload one first.")
if ou_wkb and (show_dem or show_con):
    try:
        terr = terrain(ou_wkb)
        dem_ov = dem_web(ou_wkb) if show_dem else None
        if show_con and not terr[3]["features"]:
            notes.warning(f"No contour lines of {CONTOUR_MIN_LEN} m or longer inside the buffer.")
    except Exception as e:
        notes.error(f"DEM error: {e}")
if ou_wkb and show_roads:
    rk = next((v for k, v in assets.items() if k.lower().endswith(".pmtiles") and stem_of(k).startswith("roads")), None)
    try:
        if not rk:
            raise FileNotFoundError("roads.pmtiles not found in the HF repo listing")
        rv = _S.sig.get(rk)
        road_fc = road_data(rk, ou_wkb, rv)
        if not road_fc:
            notes.warning("No roads inside the buffer.")
    except Exception as e:
        notes.error(f"Roads error: {e}")
if ou_wkb:
    for k, (lab, stem, _) in DRAIN.items():
        if not show_dr[k]:
            continue
        try:
            wk = find_asset(assets, stem, ".pmtiles")
            if not wk:
                raise FileNotFoundError(f"{stem}.pmtiles not found in the HF repo listing")
            wv[k] = _S.sig.get(wk)
            fc = water_data(wk, ou_wkb, wv[k])
            if fc:
                water[k] = fc
            else:
                notes.warning(f"No {lab} inside the buffer.")
        except Exception as e:
            notes.error(f"{lab} error: {e}")
on_dem, on_con, on_rd, on_wt = bool(dem_ov), bool(show_con and terr and terr[3]["features"]), bool(road_fc), bool(water)

pmc = lambda n: {"name": n, "v": zlib.crc32(str(_S.sig.get(pm_assets[n])).encode()),
                 "r2": f"{R2_BASE}/{quote(n)}" if R2_BASE else None,
                 "alt": f"{HF_BASE}/resolve/main/{quote(pm_assets[n])}"}
warm = [{"r2": p["r2"], "v": p["v"]} for p in map(pmc, pm_assets)] if R2_BASE else []

geo_ov = [r for r in (gjob["results"] if gjob and gjob["status"] == "done" else [])
          if "error" not in r and r["name"] + ".pmtiles" not in pm_assets]


def preview_url(job, name):
    def rd():
        with open(os.path.join(job["dir"], name + ".jpg"), "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    return memo(("geoprev", st.session_state["geo_id"], name), rd)


geo_local = [os.path.join(gjob["dir"], r["name"] + ".pmtiles") for r in geo_ov]

if geo_ov:
    publish_static(st.session_state["geo_id"], gjob)

geo_img = [{"name": r["name"], "v": zlib.crc32(st.session_state["geo_id"].encode()), "b": r["pb"],
            "alt": f"/app/static/geo/{st.session_state['geo_id']}/{r['name']}.png",
            "url": preview_url(gjob, r["name"])} for r in geo_ov] if show_topo else []

vp = [{"color": VS_COL[i % 3], "pts": [[q["lat"], q["lon"], f"Batch {i + 1} - Vantage {k}", q["wtg_names"], round(q["elev_m"]), q["wtg_count"]]
                                       for k, q in enumerate(b["pts"], 1)]}
      for i, b in enumerate(vs_b)] if vs_res else []
vr = [dict(zip(("url", "b"), memo(("vr2", hash(vs_args), n), lambda b=b: cov_png(b["cov"], vs_res["g"], VS_RAS))))
      for n, (b, on) in enumerate(zip(vs_b, vs_on), 1) if on] if vs_res else []
vbuf = memo(("vbuf", hash(vs_args)), lambda: wtg_buffer(vs_args[0], vs_args[3])) if vs_res else None

cfg = {
    "basemap": basemap, "base": BASEMAPS[basemap], "colors": COLORS, "layers": [dict(light(l), fc=site_fc(l)) for l in layers],
    "sig": "|".join(sig + [g["name"] for g in geo_ov]),
    "pm": [pmc(n) for n in chosen] if show_topo else [], "geo_img": geo_img,
    "all": warm, "wv": str(zlib.crc32(json.dumps(warm).encode())),
    "osm_index": osm_json, "georef": [r["pb"] for r in geo_ov], "vp": vp,
    "dem": {"url": dem_ov[0], "b": dem_ov[1]} if on_dem else None,
    "contours": terr[3] if on_con else None, "con_labels": terr[5] if on_con else None,
    "roads": road_fc, "water": water or None, "wcol": {k: v[2] for k, v in DRAIN.items()},
    "cc": CON_COL, "rc": ROAD_COL, "vbc": VS_BUF_COL, "sw": SITE_W,
}
if vp:
    cfg["sig"] += "|vs"
    cfg.update(vr=vr, vbuf=vbuf, wtg=[[q[1], q[0], q[2]] for q in vs_args[0]])


# ---------------- Page ----------------
HTML = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@3.0.7/dist/pmtiles.js"></script>
<style>
html,body{margin:0}
.cl{display:block;width:30px;text-align:center;font:bold 10px Arial,sans-serif;color:#000;
    text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;pointer-events:none}
.vsr{image-rendering:pixelated}
</style>
<div id="m" style="height:480px;border:2px solid #d3ccb6;border-radius:10px;background:#fff"></div>
<script>
const C = __CFG__;
const log = m => console.error(m);
const abs = u => { try { return u.startsWith('/') ? window.parent.location.origin + u : u; } catch (e) { return u; } };
let keep = null;
try { keep = JSON.parse(sessionStorage.getItem('topoView')); } catch (e) {}
if (keep && keep.sig !== C.sig) keep = null;
const map = L.map('m', {preferCanvas: true, zoomSnap: 0.5}).setView(keep ? keep.c : [21, 79], keep ? keep.z : 5.5);
map.on('moveend', () => { try { sessionStorage.setItem('topoView', JSON.stringify({sig: C.sig, c: map.getCenter(), z: map.getZoom()})); } catch (e) {} });
L.control.scale().addTo(map);
if (C.base) L.tileLayer(C.base, {maxZoom: 20, attribution: C.basemap}).addTo(map);
[['topo', 300], ['dem_pane', 310], ['water_pane', 315], ['con_pane', 320], ['conlab_pane', 325], ['road_pane', 330], ['vsr_pane', 335], ['osm_pane', 400]]
  .forEach(([n, z]) => { map.createPane(n).style.zIndex = z; });

if (C.dem) L.imageOverlay(C.dem.url, C.dem.b, {pane: 'dem_pane', opacity: 0.75}).addTo(map);
if (C.water) Object.entries(C.water).forEach(([k, fc]) => L.geoJSON(fc, {pane: 'water_pane', interactive: false,
  style: f => f.geometry.type.indexOf('Line') >= 0
    ? {color: C.wcol[k], weight: 1.6, fill: false}
    : {color: C.wcol[k], weight: 0.8, fillColor: C.wcol[k], fillOpacity: 0.55}}).addTo(map));
if (C.contours) L.geoJSON(C.contours, {pane: 'con_pane', interactive: false, style: {color: C.cc, weight: 0.8}}).addTo(map);
if (C.con_labels) {
  C.con_labels.forEach(l => L.marker([l[0], l[1]], {pane: 'conlab_pane', interactive: false, keyboard: false,
    icon: L.divIcon({className: '', html: '<span class="cl">' + l[2] + '</span>', iconSize: [30, 12], iconAnchor: [15, 6]})}).addTo(map));
  const updLab = () => { map.getPane('conlab_pane').style.display = map.getZoom() >= 12 ? '' : 'none'; };
  map.on('zoomend', updLab); updLab();
}
if (C.roads) L.geoJSON(C.roads, {pane: 'road_pane', interactive: false, style: {color: C.rc, weight: 2}}).addTo(map);

const wmIcon = L.divIcon({className: '', iconSize: [30, 34], iconAnchor: [15, 12],
  html: '<svg width="30" height="34" viewBox="-15 -13 30 34"><g stroke="#fff" stroke-width="4.5" stroke-linecap="round"><path d="M0 0L0 -11M0 0L-9.5 5.5M0 0L9.5 5.5M0 0L0 18"/></g>'
      + '<path d="M0 0L0 -11M0 0L-9.5 5.5M0 0L9.5 5.5M0 0L0 18" stroke="#000" stroke-width="2" stroke-linecap="round"/><circle r="2.6" fill="#000" stroke="#fff"/></svg>'});
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// siteG = vectors + their buffers (2/5/10 km ...) shown on every tab except Vantage; vsG = WTG site + WTG buffer + vantage results (Vantage tab only)
const siteG = L.layerGroup().addTo(map), vsG = L.layerGroup();
let all = null;
C.layers.forEach(lay => {
  const n = lay.bufs.length, dash = lay.dash ? '8,6' : null, cols = lay.cols || C.colors;
  const grow = g => { all = all ? all.extend(g.getBounds()) : g.getBounds(); };
  lay.bufs.forEach((b, i) => grow(L.geoJSON(b.gj, {style: {color: cols[i == 0 ? 0 : i == n - 1 ? 2 : 1], weight: 2.5, dashArray: dash, fill: false},
    pointToLayer: (f, ll) => L.marker(ll, {icon: wmIcon})}).bindTooltip(b.label, {sticky: true}).addTo(siteG)));
    grow(L.geoJSON(lay.fc, {
    style: {color: C.colors[0], weight: Math.max(C.sw * 2.4, 0.8), dashArray: dash, fill: true, fillOpacity: 0},
    pointToLayer: (f, ll) => L.marker(ll, {icon: wmIcon}),
    onEachFeature: (f, layer) => {
      const p = f.properties || {};
      const nk = Object.keys(p).find(k => k.toLowerCase() === 'name' && p[k] !== null && p[k] !== '');
      layer.bindTooltip(nk ? String(p[nk]) : lay.name, {sticky: true});
      const rows = Object.entries(p).filter(([k, v]) => v !== null && v !== '')
        .map(([k, v]) => '<tr><td style="padding:2px 8px 2px 0;color:#555;vertical-align:top"><b>' + esc(k) + '</b></td><td style="padding:2px 0;word-break:break-word">' + esc(v) + '</td></tr>').join('');
      layer.bindPopup('<div style="font:12px Arial,sans-serif"><div style="font-weight:bold;margin-bottom:4px">' + esc(lay.name) + '</div>'
        + (rows ? '<table>' + rows + '</table>' : '<i>No attributes</i>') + '</div>', {maxWidth: 340, maxHeight: 260});
    }}).addTo(siteG));
});
if (all && !keep) map.fitBounds(all.pad(0.1));
if (C.georef.length && !keep) map.fitBounds(L.latLngBounds(C.georef.flat(1)).pad(0.1));

// vantage analysis (Batch 1): black WTG buffer (white halo so it reads on satellite), WTG markers, optional raster, vantage points
let vbB = null;
if (C.vbuf) {
  L.geoJSON(C.vbuf, {interactive: false, style: {color: '#ffffff', weight: 6, opacity: 0.9, fill: false}}).addTo(vsG);
  const vb = L.geoJSON(C.vbuf, {interactive: false, style: {color: C.vbc, weight: 2.5, dashArray: '8,5', fill: false}}).addTo(vsG);
  vbB = vb.getBounds();
}
(C.vr || []).forEach(r => L.imageOverlay(r.url, r.b, {pane: 'vsr_pane', opacity: 1, interactive: false, className: 'vsr'}).addTo(vsG));
(C.wtg || []).forEach(w => L.marker([w[0], w[1]], {icon: wmIcon}).bindTooltip(esc(w[2]), {direction: 'top'}).addTo(vsG));
(C.vp || []).forEach(b => b.pts.forEach(q => L.circleMarker([q[0], q[1]], {radius: 8, color: '#fff', weight: 2, fillColor: b.color, fillOpacity: 1})
  .bindTooltip(q[2], {sticky: true})
  .bindPopup('<div style="font:12px Arial,sans-serif"><b>' + esc(q[2]) + '</b><br>Elevation: ' + q[4] + ' m<br>WTGs visible (' + q[5] + '): ' + esc(q[3]) + '</div>', {maxWidth: 300})
  .addTo(vsG)));

// Tab-aware content. Modes: 'vs' = Vantage tab open (vantage layers only), 'main' = another tab open (original vectors + buffers only),
// 'both' = the open tab cannot be determined (parent page unreadable / Vantage tab not found) -> show everything so nothing is ever hidden.
// A layer switch is always present for manual control. The tab is polled every 300 ms; the legend groups follow the same mode.
if (C.vbuf) {
  let mode = null;
  const apply = m => {
    if (m === mode) return;
    const first = mode === null;
    mode = m;
    if (m === 'main') map.removeLayer(vsG); else map.addLayer(vsG);
    if (m === 'vs') map.removeLayer(siteG); else map.addLayer(siteG);
    if (m === 'vs' && vbB && !(first && keep)) map.fitBounds(vbB.pad(0.1));
    else if (m === 'both' && first && !keep && !all && vbB) map.fitBounds(vbB.pad(0.1));
    else if (m === 'main' && !first && all) map.fitBounds(all.pad(0.1));
  };
  const probe = () => {
    try {
      const D = window.parent.document;
      const t = [...D.querySelectorAll('[role="tab"]')].find(x => /vantage/i.test(x.textContent || ''));
      if (!t) return 'both';
      const s = t.getAttribute('aria-selected');
      return s === null ? 'both' : (s === 'true' ? 'vs' : 'main');
    } catch (e) { return 'both'; }
  };
  const legend = m => {
    try {
      const D = window.parent.document;
      D.querySelectorAll('.lgg-main').forEach(e => { e.style.display = m === 'vs' ? 'none' : ''; });
      D.querySelectorAll('.lgg-vs').forEach(e => { e.style.display = m === 'main' ? 'none' : ''; });
    } catch (e) {}
  };
  L.control.layers(null, {'Site vectors & buffers': siteG, 'Vantage analysis (Batch 1)': vsG}, {collapsed: true, position: 'topright'}).addTo(map);
  const tick = () => { const m = probe(); apply(m); legend(m); };
  tick();
  setInterval(tick, 300);
}

// PMTiles source that caches every byte range in the browser Cache API
class CSrc {
  constructor(url, v) { this.url = url; this.v = v; }
  getKey() { return this.url; }
  async getBytes(o, l, sig) {
    const k = this.url + '?v=' + this.v + '&r=' + o + '-' + (o + l - 1);
    let c = null;
    try { c = await caches.open('pm1'); const h = await c.match(k); if (h) return {data: await h.arrayBuffer()}; } catch (e) {}
    const r = await fetch(this.url, {signal: sig, headers: {Range: 'bytes=' + o + '-' + (o + l - 1)}});
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const b = await r.arrayBuffer();
    if (r.status === 200 && b.byteLength !== l) throw new Error('server ignored Range');
    try { if (c) await c.put(k, new Response(b.slice(0))); } catch (e) {}
    return {data: b};
  }
}
const addPM = async (url, v) => {
  const src = new pmtiles.PMTiles(new CSrc(url, v)), h = await src.getHeader();
  const detail = h.minZoom >= 13;   // SOI detail strips: show from z13 and overzoom past z14
  pmtiles.leafletRasterLayer(src, {pane: 'topo', minZoom: h.minZoom, maxNativeZoom: h.maxZoom,
    maxZoom: detail ? 20 : h.maxZoom, zIndex: h.minZoom, opacity: 0.85, keepBuffer: 4}).addTo(map);
};
(async () => {
  for (const g of (C.geo_img || [])) {
    try { await addPM(abs(g.alt), g.v); }
    catch (e) { log(g.name + ': PMTiles preview failed -> image preview'); L.imageOverlay(g.url, g.b, {pane: 'topo', opacity: 0.85}).addTo(map); }
  }
  for (const p of C.pm) {
    let ok = false;
    for (const [label, url] of [['R2', p.r2], ['Hugging Face direct', p.alt]]) {
      if (!url) continue;
      try { await addPM(url, p.v); ok = true; break; } catch (e) { log(p.name + ': ' + label + ' failed -> ' + (e && e.message || e)); }
    }
    if (!ok) log(p.name + ': SKIPPED (all sources failed)');
  }
})();

// one-time warm-up per browser and data version: zoomed-out tiles of every sheet
setTimeout(async () => {
  try { if (!C.all.length || localStorage.getItem('pmWarm') === C.wv) return; } catch (e) { return; }
  const q = C.all.slice();
  const run = async () => {
    for (let p; (p = q.shift());) {
      try {
        const s = new pmtiles.PMTiles(new CSrc(p.r2, p.v)), h = await s.getHeader();
        if (h.minZoom > 8) continue;
        for (let z = h.minZoom; z <= Math.min(h.minZoom + 2, h.maxZoom); z++) {
          const n = 2 ** z, cl = v => Math.min(n - 1, Math.max(0, Math.floor(v)));
          const tx = lo => cl((lo + 180) / 360 * n), ty = la => cl((1 - Math.asinh(Math.tan(la * Math.PI / 180)) / Math.PI) / 2 * n);
          for (let x = tx(h.minLon); x <= tx(h.maxLon); x++)
            for (let y = ty(h.maxLat); y <= ty(h.minLat); y++) await s.getZxy(z, x, y);
        }
      } catch (e) {}
    }
  };
  await Promise.all([run(), run(), run(), run()]);
  try { localStorage.setItem('pmWarm', C.wv); } catch (e) {}
}, 3000);

if (C.osm_index) {
  L.geoJSON(C.osm_index, {
    pane: 'osm_pane',
    style: () => ({color: '#ff4400', weight: 1, fill: false, fillOpacity: 0}),
    onEachFeature: function(feature, layer) {
      const p = feature.properties || {}, sheetName = p.EVEREST_SH || Object.values(p).find(v => typeof v === 'string') || 'N/A';
      layer.bindTooltip(sheetName, {sticky: true, direction: 'auto'});
      layer.on({
        mouseover: e => { e.target.setStyle({color: '#000000', weight: 2.5}); e.target.bringToFront(); },
        mouseout:  e => { e.target.setStyle({color: '#ff4400', weight: 1}); }
      });
    }
  }).addTo(map);
}
</script>
"""

col_map, col_leg = st.columns([3.2, 1.1])
with col_map:
    show_html(HTML.replace("__CFG__", json.dumps(cfg).replace("</", "<\\/")), 490)
    c1, c2 = st.columns([1, 2])
    c1.markdown("**Source : Survey of India**")
    c2.markdown(f"**Topo sheet No. {sheets_str}**")
with col_leg:
    with st.container(height=480, border=True):
        st.markdown("### **Legends**")
        # main group: original vectors + their buffers (colours identical to the map) and the other layers
        lg = "".join(
            (f'<div class="lg">{WM_SVG}<b>{lab}</b></div>' if icon else
             f'<div class="lg"><i style="border:{w}px {"dashed" if dash else "solid"} {c}"></i><b>{lab}</b></div>')
            for lab, c, w, dash, icon in legend_rows(layers))
        if on_dem:
            lg += '<div class="lg"><b>Elevation in (m)</b></div>' + "".join(
                f'<div class="lg"><i style="background:{c};border:1px solid #777"></i><b>{lb}</b></div>'
                for c, lb in zip(DEM_PAL, dem_labels(terr[2])))
        for lab, c, w, sty in line_rows(on_rd, on_con, water):
            if sty == "fill":
                lg += f'<div class="lg"><i style="background:{tint(c)};border:{w}px solid {c}"></i><b>{lab}</b></div>'
            else:
                lg += f'<div class="lg"><i style="height:0;border-top:{w}px solid {c}"></i><b>{lab}</b></div>'
        # vantage group: shown only while the Vantage tab is open (or when the open tab cannot be detected)
        lv = ""
        if vs_b:
            lv = f'<div class="lg"><b>Vantage analysis (Batch 1)</b></div><div class="lg">{WM_SVG}<b>WTG (site)</b></div>'
            for n, b in enumerate(vs_b, 1):
                lv += (f'<div class="lg"><i style="width:12px;height:12px;margin:0 5px;border-radius:50%;background:{VS_COL[(n - 1) % 3]};'
                       f'border:2px solid #fff;box-shadow:0 0 0 1px #777"></i><b>Vantage Batch {n} ({len(b["pts"])} points)</b></div>')
            lv += (f'<div class="lg"><i style="height:0;border-top:3px dashed {VS_BUF_COL}"></i>'
                   f'<b>WTG buffer {fmt_m(vs_args[3])}</b></div>')
            if vr:
                lv += (f'<div class="lg"><i style="background:linear-gradient(90deg,{tint(VS_RAS, .55)},{VS_RAS},#015a85);border:1px solid #777"></i>'
                       '<b>Visibility raster (darker = more WTGs)</b></div>')
        if lg or lv:
            st.markdown(f'<div class="lgg lgg-main">{lg}</div>' + (f'<div class="lgg lgg-vs">{lv}</div>' if lv else ""),
                        unsafe_allow_html=True)
        else:
            st.markdown("Upload a vector to see its legend.")
        if leg_bytes:
            st.image(leg_bytes, caption="CONVENTIONAL SYMBOLS", width="stretch")


# ---------------- georeferencing status ----------------
@st.fragment(run_every=2 if geo_busy else None)
def geo_ui():
    j = GEO.get(st.session_state.get("geo_id"))
    refresh_geo(j)
    if not j:
        return

    def end():
        gid = st.session_state.pop("geo_id", None)
        kill_job(j)
        drop_static(gid)
        GEO.pop(gid, None)
        st.session_state.pop("geo_poll", None)

    def clear_btn(label="🗑️ Clear"):
        if st.button(label, key="geo_x", width="stretch"):
            end()
            st.rerun()

    if j["status"] != "running" and (j.get("push") or {}).get("status") != "running" and st.session_state.pop("geo_poll", False):
        st.rerun()
    for nm in j["skipped"]:
        st.warning(f"{nm}: no matching sheet - skipped.")
    el = int((j["end"] or time.time()) - j["start"])
    if j["status"] == "running":
        st.progress(min(j["progress"], .99), text=f"Processing: {j['msg']}  |  {el}s  (background - keep working)")
        return clear_btn("⛔ Cancel")
    if j["status"] == "error":
        st.error(f"Processing failed: {j['error']}")
        return clear_btn()
    ok = [r for r in j["results"] if "error" not in r]
    for r in j["results"]:
        if "error" in r:
            st.error(f"{r['name']}: {r['error']}")
    pu = j.get("push") or {}
    if pu.get("status") == "done":
        st.session_state["geo_select"] = list(set(st.session_state.get("geo_select", [])) | {r["name"] + ".pmtiles" for r in ok})
        release_assets.clear()
        end()
        st.rerun()
    if not ok:
        return clear_btn()
    st.markdown("### 🧭 New sheets")
    st.caption(f"Done in {el}s. Check that they line up on the map, then download or add them to the library.")
    st.markdown("| Sheet | Zoom | Tiles | Size | Already in library |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {r['name']} | {r['zoom'][0]}-{r['zoom'][1]} | {r['tiles']} | {r['pm_mb']} MB | "
        f"{'yes (skipped)' if r['name'] + '.pmtiles' in pm_assets else 'no'} |" for r in ok))
    cols = st.columns(4)
    for i, r in enumerate([r for r in ok if r.get("tif_mb")]):
        with open(os.path.join(j["dir"], r["name"] + ".tif"), "rb") as fh:
            cols[i % 4].download_button(f"⬇️ {r['name']} (GeoTIFF, {r['tif_mb']} MB)", fh.read(), file_name=r["name"] + ".tif",
                                        mime="image/tiff", width="stretch", key=f"geo_dl{i}")
    new = [r for r in ok if r["name"] + ".pmtiles" not in pm_assets]
    if pu.get("status") == "running":
        st.info("☁️ Adding to the library in the background...")
    elif not new:
        st.info("All sheets are already in the library.")
    else:
        if pu.get("status") == "error":
            st.error(f"Upload failed: {pu['msg']}")
        sure = st.checkbox("✅ I checked the map - the sheet(s) line up correctly", key="geo_ok")
        if st.button("☁️ Add to library", type="primary", disabled=not sure, key="geo_push", width="stretch"):
            wt = write_token()
            if not wt:
                st.error("No write token: set HF_WRITE_TOKEN (env or Streamlit secret).")
            else:
                dirs = [os.path.dirname(p) for p in pm_assets.values() if not p.startswith("contributions/")]
                folder = max(set(dirs), key=dirs.count) if dirs else ""
                push_bg(j, [(os.path.join(j["dir"], r["name"] + ".pmtiles"), f"{folder}/{r['name']}.pmtiles".lstrip("/"))
                            for r in new], wt)
                st.session_state["geo_poll"] = True
                st.rerun()
    clear_btn("🗑️ Clear")


geo_ui()

for jid, j in list(JOBS.items()):
    if j["status"] != "running" and j.get("end") and time.time() - j["end"] > JOB_TTL:
        JOBS.pop(jid, None)
job = JOBS.get(st.session_state.get("job_id"))
refresh_job(job)

opts = [k for k, ok in (("Toposheet Map", show_topo and (chosen or geo_ov)), ("DEM - Elevation Map", on_dem),
                        ("DEM - Contour Map", on_dem and on_con), ("Roads & Transportation Map", on_rd),
                        ("Drainage Map", on_wt), ("Satellite Map", bool(layers))) if ok] or ["Toposheet Map"]


@st.fragment(run_every=1.5 if (job and job["status"] == "running") else None)
def export_ui():
    job = JOBS.get(st.session_state.get("job_id"))
    refresh_job(job)
    running = bool(job) and job["status"] == "running"
    if job and not running and st.session_state.pop("polling", False):
        st.rerun()

    cl, ck, cd, cb = st.columns([1.1, 1.9, 1.2, 2.6])
    cl.markdown("### 📥 Export Map")
    kind = ck.selectbox("Map type", opts, label_visibility="collapsed", disabled=running)
    dpi = cd.selectbox("Resolution (DPI)", DPI_OPTIONS, index=0, label_visibility="collapsed", disabled=running)

    if running:
        if cb.button("⛔ Cancel export", width="stretch"):
            kill_job(job, "cancelled")
            st.session_state.pop("polling", None)
            st.rerun()
    elif cb.button("🚀 Generate High-Res Map", type="primary", width="stretch"):
        hd, hc, hr, hw, hs = KINDS[kind]
        topo = not any(KINDS[kind])
        if topo and not (chosen or geo_ov):
            st.warning("Select at least one toposheet in the Toposheet viewer tab first.")
        elif hs and not layers:
            st.warning("Upload at least one vector first (the satellite map shows the vectors).")
        else:
            if job:
                JOBS.pop(st.session_state.get("job_id"), None)
            jid = uuid.uuid4().hex
            extra = {"dem": terr[:3] if hd else None, "contours": terr[3] if hc else None,
                     "roads": road_fc if hr else None, "water": water if hw else None, "logo": logo_bytes,
                     "base": None if topo else SAT_BASE if hs else BASEMAPS[basemap]}
            JOBS[jid] = start_export([pm_assets[n] for n in chosen] + geo_local if topo else [], layers, project_title, sheets_str,
                                     dpi, leg_bytes if topo else None, kind, extra)
            st.session_state["job_id"] = jid
            st.session_state["polling"] = True
            st.rerun()

    if job:
        elapsed = int((job["end"] or time.time()) - job["start"])
        if job["status"] == "running":
            st.progress(min(job["progress"], 0.99), text=f"{job['msg']}  |  elapsed {elapsed}s")
            st.caption("You can change basemap / layers while this runs. The export keeps going.")
        elif job["status"] == "done":
            st.success(f"{job['kind']} ready in {elapsed}s ({job['dpi']} DPI).")
            if job.get("warn"):
                st.warning(f"Failed to load (white areas): {job['warn']}. Generate again.")
            st.download_button("⬇️ Download Ready High-Res Map (PNG)", job["result"],
                               file_name=re.sub(r"\W+", "_", job["kind"]) + "_HighRes.png", mime="image/png", width="stretch")
        elif job["status"] == "cancelled":
            st.warning("Export cancelled.")
        elif job["status"] == "error":
            st.error(f"Export failed: {job['error']}")


# ---------------- deliverables ----------------
slug = lambda s: re.sub(r"\W+", "_", s).strip("_")
fc_geoms = lambda fc: [shape(f["geometry"]) for f in fc["features"]]
dls = []
for l in layers:
    k0 = l["key"]
    dls.append((f"⬇️ {l['name']} (KML)",
                memo(("site", k0), lambda l=l: to_kml(
                    l["name"], [(feat_name(f["props"], l["name"]), shape(f["geom"]), f["props"]) for f in l["feats"]], COLORS[0])),
                slug(l["name"]) + ".kml", KML))
    for i, b in enumerate(l["bufs"]):
        dls.append((f"⬇️ {b['label']} (KML)",
                    memo(("buf", k0, i), lambda b=b, i=i, l=l: to_kml(
                        b["label"], [(b["label"], g) for g in fc_geoms(b["gj"])], buf_color(i, len(l["bufs"]), l["cols"]))),
                    slug(b["label"]) + ".kml", KML))
if on_dem:
    dls.append(("⬇️ DEM (GeoTIFF)", dem_tif_bytes(ou_wkb), "dem_30m.tif", "image/tiff"))
if on_con:
    dls.append(("⬇️ Contours (KML)",
                memo(("con", okey), lambda: to_kml(
                    f"Contour {CONTOUR_STEP} meter",
                    [(f"{int(f['properties']['elev'])} m", shape(f["geometry"])) for f in terr[3]["features"]], CON_COL, 1)),
                f"contours_{CONTOUR_STEP}m_min_length_{CONTOUR_MIN_LEN}m.kml", KML))
if on_rd:
    dls.append(("⬇️ Roads (KML)",
                memo(("rd", okey, rv), lambda: to_kml("Roads", [("Roads", GeometryCollection(fc_geoms(road_fc)))], ROAD_COL)),
                "roads.kml", KML))
for k, fc in water.items():
    lab, stem, col = DRAIN[k]
    dls.append((f"⬇️ {lab} (KML)",
                memo(("wt", k, okey, wv.get(k)), lambda lab=lab, fc=fc, col=col: to_kml(
                    lab, [(lab, GeometryCollection(fc_geoms(fc)))], col, 1)),
                stem + ".kml", KML))
for n, b in enumerate(vs_res["batches"] if vs_res else [], 1):   # all 3 batches are exported
    tag, vk = f"Batch_{n}_Target_{int(vs_args[2])}m", hash(vs_args)
    dls.append((f"⬇️ Vantage Batch {n} (KML)", memo(("vk", vk, n), lambda n=n, b=b: vp_kml(n, b)),
                f"Vantage_Points_{tag}.kml", KML))
    dls.append((f"⬇️ Visibility Batch {n} (TIF)", memo(("vt", vk, n), lambda b=b: geotiff_bytes(b["cov"], vs_res["g"], 0)),
                f"Visibility_Coverage_{tag}.tif", "image/tiff"))
with st.container(border=True):
    st.markdown('<div class="sec">Deliverables</div>', unsafe_allow_html=True)
    for i, (lab, data, nm, mime) in enumerate(dls):
        if i % 5 == 0:
            cols = st.columns(5)
        cols[i % 5].download_button(lab, data, nm, mime, width="stretch", key=f"dl{i}")
    export_ui()

st.markdown("""
<div class="custom-footer">Developed by <b>Himanshu Gupta</b> &nbsp;|&nbsp;
📧 <a href="mailto:himanshu.guptta.311@gmail.com">himanshu.guptta.311@gmail.com</a> &nbsp;|&nbsp;
🔗 <a href="https://www.linkedin.com/in/himanshu-gupta-4a7942224" target="_blank">LinkedIn Profile</a></div>
""", unsafe_allow_html=True)
