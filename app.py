"""
Geospatial Site Analysis & Mapping Dashboard - single file, Pillow-only export.
Run:  streamlit run app.py      (login: LOGIN_USER / LOGIN_PASS below, or env APP_USER / APP_PASS)
Needs: streamlit, requests, geopandas, shapely>=2, pyproj, pillow>=10.1, pmtiles, numpy, contourpy,
       mapbox-vector-tile (vector PMTiles), huggingface_hub + hf_xet (uploads) - the last two auto-install when needed.
Data: Hugging Face dataset himanshuguptta/geotiff-hub (HF_TOKEN env/secret recommended).

 * After login a one-time "setting up" screen downloads the data files to the cache folder (D:\\topo_cache by default,
   override with env TOPO_CACHE); afterwards everything is read from disk -> fast.
 * Roads / drainage are vector PMTiles: only the tiles over the outer buffer are read, decoded, merged and clipped.
 * The PNG export runs in a separate worker process ("--export <dir>") so a crash/OOM never kills the app.
"""
import os as _os
LOGIN_USER, LOGIN_PASS = _os.getenv("APP_USER", "Kosher"), _os.getenv("APP_PASS", "Himanshugis")

import os, re, io, sys, gc, json, math, time, uuid, zlib, base64, shutil, zipfile, tempfile, textwrap, threading, \
    subprocess, types, hmac, importlib, faulthandler
from functools import lru_cache
from urllib.parse import quote, unquote
from xml.sax.saxutils import escape
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:  # Fix "Cannot find proj.db"
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
from shapely.geometry import box, shape, mapping, LineString, GeometryCollection
from shapely.ops import unary_union
from PIL import Image, ImageDraw, ImageFont, TiffImagePlugin
from pmtiles.reader import Reader

Image.MAX_IMAGE_PIXELS = None

# disk cache: hidden folder "geospatial_dashboard_cache" on D: when it exists (else the system temp folder); module-level (the export worker uses it too)
CACHE = os.environ.get("TOPO_CACHE") or os.path.join(
    "D:\\" if os.path.isdir("D:\\") else tempfile.gettempdir(), "geospatial_dashboard_cache")
os.makedirs(CACHE, exist_ok=True)
try:  # hide the folder in Explorer (Hidden + System attributes, same as Windows' own protected folders)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetFileAttributesW(CACHE, 0x2 | 0x4)
except Exception:
    pass
PRELOAD_TOPO = True  # True: the setup screen also downloads ALL toposheets | False: only roads/water/legend/logo

APP_TITLE = "Geospatial Site Analysis & Mapping Dashboard"
HF_REPO = "himanshuguptta/geotiff-hub"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}"
HF_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main?recursive=true"
PROXY_PORT, TILE_WORKERS, JOB_TTL = 8765, 12, 1800
LIST_TTL, WATCH_EVERY = 20, 30
R = 6378137.0
COLORS = ["#000000", "#d00000", "#0057ff"]
DPI_OPTIONS = [300, 500, 800]
LOGO_SCALE, LOGO_H_PT = 0.6, 100

BASEMAPS = {"White": None,
            "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "Google Hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            "Google Roadmap": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            "Esri Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"}
SAT_BASE = BASEMAPS["Google Satellite"]
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
DEM_PAL = ["#ffffd4", "#fed98e", "#fe9929", "#d95f0e", "#993404"]
DEM_SRC = "NASA SRTM 30 meter Digital Elevation"
ROAD_SRC, ROAD_COL = "Survey of India", "#8b4513"
WATER_SRC = ("Bhuvan LULC-10k:SIS-DP Phase 2 (2018-23) datasets rectified with current "
             "Google Earth Imagery (2025-26)")
SAT_SRC = "Google Earth Imagery"
# drainage layers: key -> (legend text, vector PMTiles name, colour)
DRAIN = {"canal": ("Canal / Drain", "canal_drain", "#00897b"),
         "lake": ("Lake / Pond", "lake_pond", "#29b6f6"),
         "river": ("River / Stream", "river_stream", "#1f4fd8"),
         "reservoir": ("Reservoir / Tank", "reservoir_tank", "#8e44ad")}
tint = lambda c, a=.45: "#" + "".join(f"{int(255 - (255 - int(c[i:i + 2], 16)) * a):02x}" for i in (1, 3, 5))
norm = lambda s: re.sub(r"[\s_\-]+", "", s.lower())
stem_of = lambda k: norm(os.path.splitext(k)[0])
is_vec = lambda k: stem_of(k).startswith("roads") or stem_of(k) in {norm(v[1]) for v in DRAIN.values()}
WM_SVG = ('<svg width="34" height="22" viewBox="-17 -11 34 22" style="flex:none"><path d="M0 -2L0 -11M0 -2L-7.8 2.5M0 -2L7.8 2.5M0 -2L0 10" '
          'stroke="#000" stroke-width="1.8" stroke-linecap="round" fill="none"/><circle cx="0" cy="-2" r="2" fill="#000"/></svg>')
CON_COL = "#707070"
CONTOUR_STEP, LABEL_STEP, CONTOUR_MIN_LEN = 10, 20, 500
KML = "application/vnd.google-earth.kml+xml"
# map type -> (DEM, contours, roads, water, satellite); all-zero = toposheet map
KINDS = {"Toposheet Map": (0, 0, 0, 0, 0), "DEM - Elevation Map": (1, 0, 0, 0, 0), "DEM - Contour Map": (1, 1, 0, 0, 0),
         "Roads & Transportation Map": (0, 0, 1, 0, 0), "Drainage Map": (0, 0, 0, 1, 0), "Satellite Map": (0, 0, 0, 0, 1)}

TOKEN = os.getenv("HF_TOKEN")
session = requests.Session()
if TOKEN:
    session.headers.update({"Authorization": f"Bearer {TOKEN}"})
web = requests.Session()  # other hosts: never sends the HF token
web.headers["User-Agent"] = "Mozilla/5.0 (toposheet-app)"
for _s in (session, web):
    _s.mount("https://", HTTPAdapter(pool_connections=8, pool_maxsize=32))

# shared state (lives in sys.modules so Streamlit reruns never reset locks / signatures)
_S = sys.modules.get("_topo_state")
if _S is None:
    _S = sys.modules["_topo_state"] = types.SimpleNamespace()
for _k, _v in dict(locks={}, bounds={}, sig={}, ok={}, glock=threading.Lock(),
                   pf=set(), pool=ThreadPoolExecutor(3)).items():
    if not hasattr(_S, _k):
        setattr(_S, _k, _v)


def ensure(mod, pkg):
    """Import `mod`, pip-installing `pkg` first if it is missing."""
    try:
        return importlib.import_module(mod)
    except ImportError:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkg.split()], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"Could not install {pkg} automatically - run: pip install -U {pkg}\n" + r.stderr[-300:])
        importlib.invalidate_caches()
        return importlib.import_module(mod)


def list_assets():
    """{file name: repo path}; also records each file's id+size in _S.sig."""
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


_pth = lambda path: os.path.join(CACHE, path.replace("/", "__"))


def _rw(p, txt=None):
    try:
        if txt is None:
            with open(p, encoding="utf-8") as f:
                return f.read()
        with open(p, "w", encoding="utf-8") as f:
            f.write(txt)
    except OSError:
        return None


def cached_path(path):
    """Local path of an up-to-date cached copy, else None (a file replaced on HF counts as not cached)."""
    p, want = _pth(path), _S.sig.get(path)
    if not os.path.exists(p):
        return None
    if want is None or _S.ok.get(path) == want:
        return p
    have = _rw(p + ".sig")
    if have is None:
        _rw(p + ".sig", want)
        have = want
    if have == want:
        _S.ok[path] = want
        return p
    return None


def local_file(path):
    """Download a dataset file once (single-flight, unique temp name, 6 retries), then read it from disk."""
    p = cached_path(path)
    if p:
        return p
    p = _pth(path)
    with _S.glock:
        lk = _S.locks.setdefault(path, threading.Lock())
    with lk:
        if cached_path(path):
            return p
        want, tmp, last = _S.sig.get(path), f"{p}.{uuid.uuid4().hex}.part", None
        for attempt in range(6):
            try:
                with session.get(f"{HF_BASE}/resolve/main/{quote(path)}", stream=True, timeout=60,
                                 headers={"Accept-Encoding": "identity"}) as r:
                    if r.status_code == 429:
                        time.sleep(min(float(r.headers.get("Retry-After", 5)), 30))
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for c in r.iter_content(1 << 20):
                            f.write(c)
                os.replace(tmp, p)
                if want:
                    _rw(p + ".sig", want)
                    _S.ok[path] = want
                return p
            except Exception as e:
                last = e
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                time.sleep(3 * (attempt + 1))
        raise last


def fetch_range(path, rng=None, full=False):
    """(status, Content-Range, bytes). From the disk cache when present (full=True downloads first); else straight from HF."""
    p = local_file(path) if full else cached_path(path)
    if p:
        size = os.path.getsize(p)
        m = re.match(r"bytes=(\d+)-(\d*)", rng or "")
        with open(p, "rb") as f:
            if not m:
                return 200, None, f.read()
            a = int(m.group(1))
            b = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            f.seek(a)
            return 206, f"bytes {a}-{b}/{size}", f.read(max(b - a + 1, 0))
    h = {"Accept-Encoding": "identity", **({"Range": rng} if rng else {})}
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


def web_img(url, key):
    """Tile from any web host, cached on disk. None on failure."""
    p = os.path.join(CACHE, f"tile__{key}.png")
    if not os.path.exists(p):
        for a in range(3):
            try:
                r = web.get(url, timeout=20)
                if r.status_code == 200:
                    tmp = f"{p}.{uuid.uuid4().hex}.part"
                    with open(tmp, "wb") as f:
                        f.write(r.content)
                    os.replace(tmp, p)
                    break
                if r.status_code in (400, 403, 404):
                    return None
            except Exception:
                pass
            time.sleep(.5 * (a + 1))
        else:
            return None
    return Image.open(p)


def reader(aid, full=False):
    """PMTiles Reader whose small range reads (header, directories) are memoised -> far fewer requests per tile."""
    @lru_cache(maxsize=256)
    def gb(o, l):
        return fetch_range(aid, f"bytes={o}-{o + l - 1}", full)[2]
    return Reader(gb)


def sheet_bounds(aid):
    """(min_lon, min_lat, max_lon, max_lat) from the PMTiles header; failures return None (not cached)."""
    k = (aid, _S.sig.get(aid))
    if k not in _S.bounds:
        try:
            h = reader(aid).header()
            _S.bounds[k] = tuple(h[c] / 1e7 for c in ("min_lon_e7", "min_lat_e7", "max_lon_e7", "max_lat_e7"))
        except Exception:
            return None
    return _S.bounds[k]


def prefetch(paths):
    """Background download of selected sheets, so the export finds them on disk."""
    def bg(p, k):
        try:
            local_file(p)
        except Exception:
            _S.pf.discard(k)
    for p in paths:
        k = (p, _S.sig.get(p))
        if k not in _S.pf:
            _S.pf.add(k)
            _S.pool.submit(bg, p, k)


# ---------------- vectors + buffers ----------------
def parse_buffers(txt):
    v = [float(n) * (1000 if u == "km" else 1)
         for n, u in re.findall(r"(\d+(?:\.\d+)?)\s*(km|m)(?:eters?)?(?![a-z])", txt.lower())]
    return sorted({m for m in v if 0 < m <= 20000})


fmt_m = lambda m: f"{m / 1000:g}km" if m >= 1000 else f"{m:g}m"
buf_color = lambda i, n: COLORS[0 if i == 0 else 2 if i == n - 1 else 1]


def default_labels(i, bufs):
    if i == 1:
        return ["Project Site"] + [("Core Zone " if k == 0 else "Buffer Zone ") + fmt_m(m) for k, m in enumerate(bufs)]
    return ["Transmission Line"] + [f"Transmission Line Buffer {fmt_m(m)}" for m in bufs]


def make_layer(g, labels, buf_m, dash):
    """True-distance buffers via UTM, dissolved, GeoJSON out. 'feats' keeps original features + attributes (KML / popups)."""
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
    """(label, colour, width, dashed, windmill-icon?)"""
    return [r for l in layers for r in [(l["name"], COLORS[0], 1, l["dash"], l.get("pts", False))] +
            [(b["label"], buf_color(i, len(l["bufs"])), 2, l["dash"], False) for i, b in enumerate(l["bufs"])]]


def line_rows(rd, con, water=None):
    rows = []
    if rd:
        rows.append(("Roads & Transportation", ROAD_COL, 2, "line"))
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


# ---------------- KML export ----------------
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
    """[(placemark name, geometry[, attribute dict])] -> KML text ('description' -> <description>, rest -> <ExtendedData>)."""
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


# ---------------- roads / water (vector PMTiles) / DEM / contours: inside the outermost buffer only ----------------
def to_fc(g, cols=()):
    """GeoDataFrame -> compact GeoJSON dict (coords rounded to 5 decimals)."""
    rnd = lambda c: [round(c[0], 5), round(c[1], 5)] if isinstance(c[0], float) else [rnd(p) for p in c]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": dict(zip(cols, v)),
         "geometry": {"type": m.geom_type, "coordinates": rnd(mapping(m)["coordinates"])}}
        for m, *v in zip(g.geometry, *[g[c] for c in cols])]}


def clip_geoms(g, ou, tol, kinds=("LineString",)):
    """Repair, clip to the buffer, simplify, explode; keep only the given geometry types."""
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
    """Vector PMTiles (read from the disk copy) -> tiles over the outer buffer (<= 300) decoded, seams dissolved, clipped -> GeoJSON dict or None."""
    mvt = ensure("mapbox_vector_tile", "mapbox-vector-tile")
    ou, rd = swkb.loads(ou_wkb), reader(path, True)
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
        if data and data[:2] == b"\x1f\x8b":  # gzip-compressed tile (as stored in the PMTiles file)
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
    geoms = [unary_union(gr) for gr in groups if gr]  # dissolves the duplicate / split pieces at tile seams
    if not geoms:
        return None
    g = clip_geoms(gpd.GeoDataFrame(geometry=geoms, crs=4326), ou, 1e-5, kinds)
    return to_fc(g) if len(g) else None


build_roads = lambda path, ou_wkb, v=None: read_vt(path, ou_wkb, ("LineString",))  # v = file version (cache key only)
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


def build_terrain(ou_wkb):
    """Terrarium DEM clipped to the outer buffer -> (z float32 NaN outside, (x0, y1, pixel) in 3857, 5 class edges,
    contour GeoJSON (lines >= CONTOUR_MIN_LEN m), total contour km, label anchors)."""
    from contourpy import contour_generator
    ou = swkb.loads(ou_wkb)
    if ou.area == 0:
        raise RuntimeError("add a buffer distance first (the DEM is clipped to the outermost buffer)")
    b, z = ou.bounds, 13
    while True:
        _, x0, x1, y0, y1 = tile_range(b, z)
        if z <= 8 or max(x1 - x0, y1 - y0) < 12:
            break
        z -= 1
    res = mosaic((z, z, lambda zz, x, y: web_img(TERRAIN_URL.format(z=zz, x=x, y=y), f"dem_{zz}_{x}_{y}"), None), b, 99)
    if not res:
        raise RuntimeError("no elevation tiles for this area")
    cv, ext = res
    a = np.asarray(cv, dtype="float32")
    e = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    e[a[..., 3] == 0] = np.nan
    del a
    p = (ext[1] - ext[0]) / cv.width
    om = gpd.GeoSeries([ou], crs=4326).to_crs(3857).iloc[0]
    bx0, by0, bx1, by1 = om.bounds
    c0, c1 = max(int((bx0 - ext[0]) // p), 0), min(int((bx1 - ext[0]) // p) + 1, e.shape[1])
    r0, r1 = max(int((ext[3] - by1) // p), 0), min(int((ext[3] - by0) // p) + 1, e.shape[0])
    e, X0, Y1 = e[r0:r1, c0:c1], ext[0] + c0 * p, ext[3] - r0 * p
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

    f = np.where(np.isnan(e), np.nanmin(e), e)  # light 3x3 smoothing -> cleaner contours
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
    """Classify the DEM (equal interval) -> (RGBA image, [xmin, xmax, ymin, ymax] in 3857)."""
    X0, Y1, p = g
    h, w = z.shape
    ok = ~np.isnan(z)
    rgba = np.zeros((h, w, 4), np.uint8)
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in DEM_PAL], np.uint8)
    rgba[ok, :3] = pal[np.digitize(z[ok], edges[1:-1])]
    rgba[ok, 3] = 255
    return Image.fromarray(rgba), [X0, X0 + w * p, Y1 - h * p, Y1]


# =====================================================================================
# EXPORT WORKER
# =====================================================================================
def pm_src(aid):
    """Raster PMTiles sheet as a tile source; the file is read from disk (downloaded once if needed)."""
    rd = reader(aid, True)
    h = rd.header()

    def get(z, x, y):
        data = rd.get(z, x, y)
        return Image.open(io.BytesIO(data)) if data else None
    return h["min_zoom"], h["max_zoom"], get, sheet_bounds(aid)


def xyz_src(url):
    k = zlib.crc32(url.encode())
    return 0, 19, lambda z, x, y: web_img(url.format(z=z, x=x, y=y), f"bm{k}_{z}_{x}_{y}"), None


def mosaic(src, bounds, max_tiles, on_tile=None):
    """Stitch tiles covering `bounds` (lon/lat) into one RGBA canvas -> (canvas, [xmin,xmax,ymin,ymax] in 3857)."""
    zmin, z, get, sb = src
    if sb:
        bounds = [max(bounds[0], sb[0]), max(bounds[1], sb[1]), min(bounds[2], sb[2]), min(bounds[3], sb[3])]
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            return None
    n, x0, x1, y0, y1 = tile_range(bounds, z)
    while z > zmin and (x1 - x0 + 1 > max_tiles or y1 - y0 + 1 > max_tiles):
        z -= 1
        n, x0, x1, y0, y1 = tile_range(bounds, z)
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

    def load(xy):  # missing tile -> upscale parent tile (up to 3 zooms up)
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


# ---------------- drawing helpers ----------------
merc = lambda lon, lat: (R * math.radians(lon), R * math.asinh(math.tan(math.radians(lat))))
unmerc = lambda x, y: (math.degrees(x / R), math.degrees(math.atan(math.sinh(y / R))))


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
    except ValueError:
        d.text(xy, s, font=font, fill="black")


def center_text(d, s, cx, cy, w, h, size, bold=False):
    """Wrapped text, auto-sized to fit the w x h box centred on (cx, cy)."""
    while size > 6:
        f = get_font(size, bold)
        t = "\n".join(textwrap.wrap(s, max(8, int(w / (size * 0.52)))))
        bb = d.multiline_textbbox((0, 0), t, font=f, spacing=size * .2, align="center")
        if bb[2] <= w and bb[3] <= h:
            break
        size -= 1
    else:
        f, t = get_font(6), s
    sp = size * .2
    try:
        d.multiline_text((cx, cy), t, font=f, fill="black", anchor="mm", align="center", spacing=sp)
    except ValueError:
        d.multiline_text((cx - w / 2, cy - h / 2), t, font=f, fill="black", align="center", spacing=sp)


def prepared_by(img, d, cx, ya, yb, w, size, logo_bytes, logo_h):
    """'Prepared by' text followed by the logo, centred in the box (plain text when no logo)."""
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


def draw_windmill(d, x, y, s, color="black"):
    """3-blade wind turbine: tower, tapered blades, hub. (x, y) = hub centre, s = blade length (px)."""
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
    """Drainage layers: translucent polygon fill in the layer colour (holes respected) + outlines; lines as streams."""
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
    """Black elevation labels (white halo), rotated along the contour, on every `step` m contour; no overlaps."""
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


def whiten_outside(map_img, geoms, tp):
    """Satellite map: everything outside the outermost buffer polygons becomes white."""
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

    # ---- view extent ----
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

    sheets_pm = [a for a in sheets_pm if overlaps(a)]

    def pre(a):  # download all needed sheets in parallel before drawing (no-op when already cached)
        try:
            local_file(a)
        except Exception:
            pass

    progress(0.02, "Checking toposheet files...")
    with ThreadPoolExecutor(max_workers=4) as exr:
        list(exr.map(pre, sheets_pm))
    items = [(os.path.basename(a), lambda a=a: pm_src(a)) for a in sheets_pm] + \
            ([("basemap", lambda: xyz_src(ex["base"]))] if ex.get("base") else [])
    n = max(len(items), 1)
    max_tiles = min(16, math.ceil(max(mw, mh) / 256))

    # ---- base: toposheet tiles and/or web basemap (each source retried 3x, then skipped) ----
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

    # ---- overlay ----
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
        dash, n_ = ((pt(6), pt(4)) if l["dash"] else None), len(l["bufs"])
        for geoms, c, lw in [(b["gj"], buf_color(i, n_), 2) for i, b in enumerate(l["bufs"])] + [(l["site"], COLORS[0], 1)]:
            for g in geoms:
                for path in geom_paths(g):
                    pts_ = [tp(x, y) for x, y in path]
                    if len(pts_) == 1:
                        wm.append(pts_[0])
                    else:
                        vec.append((pts_, c, pt(lw), dash))
    if has_sat:  # white casing keeps dark lines visible on imagery
        for pts, c, lw, dash in vec:
            draw_path(md, pts, "white", lw + pt(1))
    for pts, c, lw, dash in vec:
        draw_path(md, pts, c, lw, dash)
    for x_, y_ in wm:
        draw_windmill(md, x_, y_, pt(12))
    if has_con:
        draw_contour_labels(map_img, ex.get("contours"), tp, pt, LABEL_STEP)

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
    put_text(d, ((mx0 + mx1) / 2, H * .03), "Topographic MAP" if topo else kind, get_font(pt(22), True), "mm")

    # ---- side panel ----
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
    put_text(d, (PX(.5), PY(.88)), "Title :", get_font(pt(11.5), True), "mm")
    center_text(d, title.replace("\n", " "), PX(.5), PY(.8225), pw * .94, (PY(.74) - PY(.905)) * .9, pt(15))

    put_text(d, (PX(.5), PY(.715)), "Legends", get_font(pt(13), True), "mm")
    f_leg, y = get_font(pt(11)), .672
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
        if legend_bytes:  # conventional symbols stretched to fill the box
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
    else:  # bottom box: [DEM classes] / Source / Prepared by (+ logo)
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


def worker_main(job_dir):
    def progress(frac, msg):
        p = os.path.join(job_dir, "progress.json")
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"progress": frac, "msg": msg}, f)
        os.replace(p + ".tmp", p)

    def read(name, mode="rb"):
        p = os.path.join(job_dir, name)
        if os.path.exists(p):
            with open(p, mode, **({"encoding": "utf-8"} if "b" not in mode else {})) as f:
                return f.read()

    cfg = json.loads(read("input.json", "r"))
    _S.sig.update(cfg.get("sig") or {})  # so the worker re-downloads files replaced on Hugging Face
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


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--export":
    faulthandler.enable()
    worker_main(sys.argv[2])
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

st.set_page_config(page_title=APP_TITLE, page_icon="🛰️", layout="wide")

# ---------------- login gate ----------------
if not st.session_state.get("auth"):
    st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 18% 12%,rgba(0,170,255,.30),transparent 45%),
  radial-gradient(circle at 85% 88%,rgba(0,230,170,.20),transparent 42%),
  linear-gradient(rgba(255,255,255,.05) 1px,transparent 1px) 0 0/46px 46px,
  linear-gradient(90deg,rgba(255,255,255,.05) 1px,transparent 1px) 0 0/46px 46px,#07162b!important}
.block-container{max-width:480px!important;margin:0 auto!important;padding-top:17vh!important}
[data-testid="stForm"]{background:rgba(255,255,255,.09);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,.28);border-radius:22px;padding:32px 30px 24px;box-shadow:0 24px 70px rgba(0,0,0,.5)}
.lg-t{text-align:center;color:#fff;font-size:28px;font-weight:800;letter-spacing:.3px}
.lg-s{text-align:center;color:#b9cde8;font-size:12px;margin:4px 0 16px;line-height:1.5}
[data-testid="stForm"] label p{color:#dbe8fb!important;font-weight:600}
[data-testid="stForm"] div[data-baseweb="input"],[data-testid="stForm"] div[data-baseweb="base-input"]{
  background:rgba(255,255,255,.96)!important;border-radius:10px}
[data-testid="stForm"] input{color:#10223d!important;-webkit-text-fill-color:#10223d!important}
[data-testid="stForm"] button[kind]{background:linear-gradient(90deg,#0a84ff,#00c2a8);color:#fff;border:0;border-radius:10px;font-weight:700}
</style>""", unsafe_allow_html=True)
    with st.form("login"):
        st.markdown(f'<div class="lg-t">🛰️ Welcome</div><div class="lg-s">{APP_TITLE}<br>Sign in to continue</div>',
                    unsafe_allow_html=True)
        u_in = st.text_input("Username")
        p_in = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", width="stretch"):
            if hmac.compare_digest(u_in.encode(), LOGIN_USER.encode()) and hmac.compare_digest(p_in.encode(), LOGIN_PASS.encode()):
                st.session_state["auth"] = True
                st.rerun()
            st.error("Invalid username or password")
    st.stop()

# ---------------- caches ----------------
release_assets = st.cache_data(ttl=LIST_TTL, show_spinner=False)(list_assets)
terrain = st.cache_resource(show_spinner="Loading elevation tiles...", max_entries=3)(build_terrain)
road_data = st.cache_resource(show_spinner="Loading roads...", max_entries=4)(build_roads)
water_data = st.cache_resource(show_spinner="Loading surface water features...", max_entries=8)(build_water)

# ---------------- one-time setup screen (prepares the data files in the cache folder) ----------------
def _setup_paths():
    return list(dict.fromkeys(
        p for k, p in release_assets().items()
        if (k.lower().endswith(".pmtiles") and (PRELOAD_TOPO or is_vec(k))) or "legend" in k.lower()
        or k.lower().endswith(("logo.jpg", "logo.jpeg")) or k.startswith("OSM_SHEET_INDEX.")))


def _needs_setup():
    if not st.session_state.get("ready"):
        return True
    try:  # already set up in this session: still make sure every file is really on disk
        return any(not cached_path(p) for p in _setup_paths())
    except Exception:
        return False


if _needs_setup():
    st.markdown("""<style>
header[data-testid="stHeader"],[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
.stApp{background:radial-gradient(circle at 18% 12%,rgba(0,170,255,.30),transparent 45%),
  radial-gradient(circle at 85% 88%,rgba(0,230,170,.20),transparent 42%),
  linear-gradient(rgba(255,255,255,.05) 1px,transparent 1px) 0 0/46px 46px,
  linear-gradient(90deg,rgba(255,255,255,.05) 1px,transparent 1px) 0 0/46px 46px,#07162b!important}
.block-container{max-width:560px!important;margin:0 auto!important;padding-top:24vh!important}
.su-card{background:rgba(255,255,255,.09);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid rgba(255,255,255,.28);border-radius:22px;padding:30px 32px 26px;box-shadow:0 24px 70px rgba(0,0,0,.5);text-align:center}
.su-t{color:#fff;font-size:26px;font-weight:800;letter-spacing:.3px}
.su-s{color:#b9cde8;font-size:13px!important;margin:6px 0 20px;line-height:1.5}
.su-spin{width:34px;height:34px;margin:0 auto 14px;border:3px solid rgba(255,255,255,.25);border-top-color:#00c2a8;
  border-radius:50%;animation:su 1s linear infinite}
@keyframes su{to{transform:rotate(360deg)}}
.su-track{position:relative;height:14px;border-radius:99px;background:rgba(255,255,255,.14);
  border:1px solid rgba(255,255,255,.25);overflow:hidden;box-shadow:inset 0 2px 6px rgba(0,0,0,.35)}
.su-bar{position:absolute;left:0;top:0;bottom:0;width:var(--to);border-radius:99px;
  background:linear-gradient(90deg,#0a84ff,#00c2a8);box-shadow:0 0 14px rgba(0,194,168,.7);animation:su-grow 1s ease-out}
@keyframes su-grow{from{width:var(--from)}to{width:var(--to)}}
.su-bar::after{content:"";position:absolute;inset:0;
  background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.55) 50%,transparent 70%);
  background-size:200% 100%;animation:su-shine 1.6s linear infinite}
@keyframes su-shine{from{background-position:200% 0}to{background-position:-200% 0}}
.su-pct{color:#dbe8fb;font-size:15px!important;font-weight:700;margin-top:12px;letter-spacing:.5px}
</style>""", unsafe_allow_html=True)
    _ph, _prev = st.empty(), [0]

    def _setup_ui(pct):
        pct = max(0, min(100, int(pct)))
        _ph.markdown(
            '<div class="su-card"><div class="su-spin"></div><div class="su-t">🛰️ Setting up your workspace</div>'
            '<div class="su-s">Getting everything ready. This only happens once and can take a few minutes.</div>'
            f'<div class="su-track"><div class="su-bar" style="--from:{_prev[0]}%;--to:{pct}%"></div></div>'
            f'<div class="su-pct">{pct}%</div></div>',
            unsafe_allow_html=True)
        _prev[0] = pct

    _setup_ui(0)
    failed, _dl = True, False
    try:
        _paths = _setup_paths()
        _missing = lambda: [p for p in _paths if not cached_path(p)]
        for _pass in range(2):  # a second pass retries whatever failed in the first
            _todo = _missing()
            if not _todo:
                break
            _dl = True
            with ThreadPoolExecutor(max_workers=4) as _ex:
                for _f in as_completed([_ex.submit(local_file, p) for p in _todo]):
                    try:
                        _f.result()
                    except Exception:
                        pass
                    _setup_ui((1 - len(_missing()) / max(len(_paths), 1)) * 100)
        failed = bool(_missing())
    except Exception:
        failed = True
    if failed:
        st.error("Couldn't finish preparing your workspace. Please check your internet connection and try again.")
        if st.button("🔄 Try again", width="stretch"):
            st.rerun()
        st.stop()
    _setup_ui(100)
    if _dl:
        time.sleep(1.0)  # let the line finish filling before the dashboard opens
    st.session_state["ready"] = True
    st.rerun()

# ---------------- dashboard header + CSS (runs only after setup is done) ----------------
st.markdown(f"""
<style>
.block-container{{padding:2.5rem 1.5rem .5rem!important}}
.big-main-title{{font-size:34px;font-weight:800;line-height:1.25;margin-bottom:.1rem}}
.sub-title{{color:#666;font-size:13px!important;margin-bottom:.6rem}}
html,body,[class*="css"],.stMarkdown,p,span,label{{font-size:12px!important}}
[data-testid="stStatusWidget"]{{display:none!important}}
.stButton>button{{font-size:12px!important;font-weight:600!important}}
hr{{margin:.4rem 0!important}}
.custom-footer{{text-align:center;color:#888;font-size:10px;padding:4px 0;border-top:1px solid #e0e0e0;margin-top:10px}}
.custom-footer a{{color:#06c;text-decoration:none}}
.lg{{display:flex;align-items:center;gap:10px;margin:7px 0}}
.lg i{{display:inline-block;width:34px;height:14px;box-sizing:border-box;flex:none}}
@media (min-width:992px){{
header[data-testid="stHeader"]{{display:none!important}}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"]{{display:none!important}}
section[data-testid="stSidebar"]{{transform:none!important;margin-left:0!important;visibility:visible!important;min-width:21rem!important;width:21rem!important}}
}}
@media (max-width:991px){{
.block-container{{padding:3rem .7rem .5rem!important}}
.big-main-title{{font-size:22px!important}}
.sub-title{{font-size:11px!important}}
}}
</style>
<div class="big-main-title">🛰️ {APP_TITLE}</div>
<div class="sub-title">Site boundaries &amp; buffer zones &middot; Toposheets &middot; DEM &amp; contours &middot; Roads &middot; Surface water &middot; Satellite &mdash; with print-ready map export</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Reading vector and building buffers...", max_entries=6)
def layer_for(name, data, bm, labels, dash):
    g = load_vector(name, data)
    if g is None:
        raise ValueError("No .shp found inside the zip.")
    l = make_layer(g, list(labels), list(bm), dash)
    l["key"] = f"{name}|{len(data)}|{bm}|{labels}|{dash}"
    return l


@st.cache_resource(show_spinner=False, max_entries=2)
def dem_web(wkb):
    """DEM as a static PNG data-URL + lat/lon bounds for the Leaflet image overlay."""
    z, g, edges = terrain(wkb)[:3]
    im, e = dem_rgba(z, g, edges)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    (lo0, la0), (lo1, la1) = unmerc(e[0], e[2]), unmerc(e[1], e[3])
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), [[la0, lo0], [la1, lo1]]


@st.cache_resource(show_spinner=False, max_entries=2)
def dem_tif_bytes(wkb):
    """Clipped DEM as a GeoTIFF (EPSG:3857, float32, nodata -9999), written with Pillow."""
    z, (X0, Y1, p) = terrain(wkb)[:2]
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550], ifd[33922] = (p, p, 0.0), (0.0, 0.0, 0.0, float(X0), float(Y1), 0.0)
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3857)
    ifd[42113] = "-9999"
    buf = io.BytesIO()
    Image.fromarray(np.where(np.isnan(z), -9999, z).astype("float32")).save(buf, "TIFF", tiffinfo=ifd, compression="tiff_deflate")
    return buf.getvalue()


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
    """Like memo, but a failed (None) result is never remembered, so it is retried on the next rerun."""
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


class Proxy(BaseHTTPRequestHandler):
    """Local range proxy for the live map: disk cache when present, otherwise the byte range is forwarded to HF."""

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges")

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def do_GET(self):
        try:
            code, cr, body = fetch_range(unquote(self.path.split("?")[0].lstrip("/")), self.headers.get("Range"))
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            if cr:
                self.send_header("Content-Range", cr)
            self._cors(); self.end_headers(); self.wfile.write(body)
        except (ConnectionError, TimeoutError):
            pass  # the browser cancelled the request (pan / zoom): harmless
        except Exception:
            try:
                self.send_response(500); self._cors(); self.end_headers()
            except Exception:
                pass

    def log_message(self, *a):
        pass


@st.cache_resource
def start_proxy():
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PROXY_PORT), Proxy)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    except OSError:
        return None  # port busy: another copy of the app is already serving it


start_proxy()


@st.cache_resource
def job_store():
    return {}


JOBS = job_store()


def first_bytes(assets, keys, local):
    """Bytes of the first readable HF asset in `keys` (repo root before contributions/), else a local file."""
    for k in keys:
        try:
            code, _, body = fetch_range(assets[k], None)
            if code == 200 and body:
                return body
        except Exception:
            pass
    here = os.path.dirname(os.path.abspath(__file__))
    for path in [os.path.join(here, local), local, "assets/" + local]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()


def legend_image(assets):
    return first_bytes(assets, [k for k in assets if "legend_toposheet" in k.lower() or "legend_topsheet" in k.lower()],
                       "legend_toposheet.png")


def logo_image(assets):
    keys = [k for k in assets if k.lower().endswith(("logo.jpg", "logo.jpeg"))]
    root = [k for k in keys if not assets[k].startswith("contributions/")]
    contrib = sorted((k for k in keys if assets[k].startswith("contributions/")), reverse=True)
    return first_bytes(assets, root + contrib, "logo.jpg")


@st.cache_resource(show_spinner="Loading OSM sheet index...", max_entries=1)
def osm_geojson(v):  # v = file version (cache key only)
    a = release_assets()
    if "OSM_SHEET_INDEX.shp" not in a:
        raise FileNotFoundError("OSM_SHEET_INDEX.shp not found in the HF repo listing")
    for ext in ("shx", "dbf", "prj", "cpg"):
        if f"OSM_SHEET_INDEX.{ext}" in a:
            local_file(a[f"OSM_SHEET_INDEX.{ext}"])
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    g = gpd.read_file(local_file(a["OSM_SHEET_INDEX.shp"]))
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
    g["geometry"] = g.geometry.simplify(0.0005)
    return json.loads(g.to_json(default=str))


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
    known = {0xC0000005: "access violation (native crash)", 0xC0000017: "not enough memory",
             0xC0000135: "DLL not found", 0xC00000FD: "stack overflow",
             0xC0000409: "native stack buffer overrun", 0xC06D007F: "DLL function not found (library version conflict)"}
    return f"{known[code]} [0x{code:08X}]" if code in known else f"exit code {rc}"


def start_export(sheets_pm, layers, title, sheets_str, dpi, legend_bytes, kind, extra):
    d = tempfile.mkdtemp(prefix="topo_export_")
    with open(os.path.join(d, "input.json"), "w", encoding="utf-8") as f:
        json.dump({"sheets_pm": sheets_pm, "layers": [light(l) for l in layers], "title": title, "sheets_str": sheets_str,
                   "dpi": dpi, "kind": kind, "base": extra.get("base"), "sig": dict(_S.sig)}, f)
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
    env = dict(os.environ, **({"HF_TOKEN": TOKEN} if TOKEN else {}))
    with open(os.path.join(d, "log.txt"), "wb") as lg:
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--export", d],
                                env=env, stdout=lg, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"status": "running", "progress": 0.0, "msg": "Starting export process...", "start": time.time(),
            "end": None, "result": None, "error": None, "dpi": dpi, "kind": kind, "dir": d, "proc": proc}


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
        try:
            with open(os.path.join(d, "warn.txt")) as f:
                job["warn"] = f.read()
        except Exception:
            pass
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


def hf_upload(files, wt):
    """Upload files to contributions/ in the HF dataset, one commit per file."""
    HfApi = ensure("huggingface_hub", "huggingface_hub hf_xet").HfApi
    api, ts, done = HfApi(token=wt), int(time.time()), []
    for f in files:
        dest = f"contributions/{ts}_{re.sub(r'[^\w.-]', '_', f.name)}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".upload") as t:
            t.write(f.getbuffer())
            tmp = t.name
        try:
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=tmp, path_in_repo=dest, repo_id=HF_REPO, repo_type="dataset",
                                    commit_message=f"Contribution {ts}: {f.name}")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        done.append(dest)
    return done


# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Data Inputs")
if st.sidebar.button("🚪 Log out", width="stretch"):
    st.session_state["auth"] = False
    st.rerun()
assets = {}
try:
    assets = release_assets()
except Exception as e:
    st.sidebar.error(f"Hugging Face error: {e}")
st.session_state["hf_sig"] = hash(tuple(sorted(_S.sig.items())))


@st.fragment(run_every=WATCH_EVERY)
def watch_hf():
    """Polls the HF listing; a new / replaced / deleted file triggers one rerun so it shows up on its own."""
    try:
        release_assets()
    except Exception:
        return
    if hash(tuple(sorted(_S.sig.items()))) != st.session_state.get("hf_sig"):
        st.rerun()


with st.sidebar:
    watch_hf()

pm_assets = {n: i for n, i in assets.items() if n.lower().endswith(".pmtiles") and not is_vec(n)}  # raster toposheets only

basemap = st.sidebar.selectbox("🗺️ Choose Base Map (also used in DEM / contour / roads / water exports)", list(BASEMAPS), index=2)
show_topo = st.sidebar.toggle("🗾 Show Topo Layer", value=True)
show_roads = st.sidebar.toggle("🛣️ Show Roads & Transportation", value=False)
show_dr = {k: st.sidebar.toggle(f"💧 {v[0]}", value=False) for k, v in DRAIN.items()}
show_dem = st.sidebar.toggle("⛰️ Show DEM (Elevation)", value=False)
show_con = st.sidebar.toggle(f"〰️ Show Contours ({CONTOUR_STEP} m, lines ≥ {CONTOUR_MIN_LEN} m long)", value=False)
st.sidebar.caption("Roads, water, DEM and contours appear only after a vector + buffer is added, and only inside the outermost buffer.")
show_osm_index = st.sidebar.toggle("📐 Show OSM Sheet Index", value=False)
chosen = st.sidebar.multiselect("Toposheets (PMTiles)", list(pm_assets))
prefetch([pm_assets[n] for n in chosen])  # background download -> export starts instantly

layers, sig = [], []
for i in (1, 2):
    vf = st.sidebar.file_uploader(f"Vector {i} (.geojson, .kml, .zip)", type=["geojson", "kml", "gpx", "zip"], key=f"vec{i}")
    bt = st.sidebar.text_input(f"Vector {i} buffers (e.g. 2km, 5km, 10km)", "2km, 5km, 10km" if i == 1 else "2km", key=f"buf{i}")
    bm = parse_buffers(bt)
    dl = default_labels(i, bm)
    lt = st.sidebar.text_input(f"Vector {i} legend text (comma separated: site, then each buffer)", ", ".join(dl), key=f"lab{i}_{bt}")
    lab = [x.strip() for x in lt.split(",")]
    labels = [lab[k] if k < len(lab) and lab[k] else d for k, d in enumerate(dl)]
    if vf:
        try:
            layers.append(layer_for(vf.name, vf.getvalue(), tuple(bm), tuple(labels), i == 2))
            sig.append(vf.name)
        except Exception as e:
            st.sidebar.error(f"Vector {i} error: {e}")

with st.sidebar.expander("🤝 Contribute files"):
    cfs = st.file_uploader("Any file: PMTiles, GeoTIFF, vector, PNG, PDF, ...", accept_multiple_files=True, key="contrib")
    if cfs and st.button("Submit contribution"):
        try:
            wt = os.getenv("HF_WRITE_TOKEN") or st.secrets.get("HF_WRITE_TOKEN") or TOKEN
        except Exception:
            wt = TOKEN
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
        st.sidebar.error(f"OSM index: {e}")

# Sheet numbers: all selected sheets, or - once vectors are loaded - only those touching the outermost buffers
outer = [shape(f["geometry"]) for l in layers
         for f in (l["bufs"][-1]["gj"] if l["bufs"] else l["site"])["features"]]
if outer and chosen:
    with ThreadPoolExecutor(max_workers=8) as _ex:
        list(_ex.map(sheet_bounds, [pm_assets[n] for n in chosen]))


def touches_buffer(name):
    b = sheet_bounds(pm_assets[name])
    return True if b is None else any(g.intersects(box(*b)) for g in outer)


names = [n for n in chosen if not outer or touches_buffer(n)]
sheets = [s for s in dict.fromkeys(sheet_no(n) for n in names) if s]
sheets_str = ", ".join(sheets) or ("None within buffers" if chosen else "56F/12, 56F/8, 56G/5, 56G/9")

# Roads / water / DEM / contours: only with a vector, only inside the outermost buffer (union over both vectors)
ou_wkb = unary_union(outer).wkb if outer else None
okey = f"{zlib.crc32(ou_wkb)}:{len(ou_wkb)}" if ou_wkb else ""
terr = road_fc = dem_ov = None
water, rv, wv = {}, None, {}
if not ou_wkb and (show_roads or show_dem or show_con or any(show_dr.values())):
    st.sidebar.warning("Roads / water / DEM / contours need a vector (Vector 1 or 2) with a buffer. Upload one first.")
if ou_wkb and (show_dem or show_con):
    try:
        terr = terrain(ou_wkb)
        dem_ov = dem_web(ou_wkb) if show_dem else None
        if show_con and not terr[3]["features"]:
            st.sidebar.warning(f"No contour lines of {CONTOUR_MIN_LEN} m or longer inside the buffer.")
    except Exception as e:
        st.sidebar.error(f"DEM error: {e}")
if ou_wkb and show_roads:
    rk = next((v for k, v in assets.items() if k.lower().endswith(".pmtiles") and stem_of(k).startswith("roads")), None)
    try:
        if not rk:
            raise FileNotFoundError("roads.pmtiles not found in the HF repo listing")
        rv = _S.sig.get(rk)
        road_fc = road_data(rk, ou_wkb, rv)
        if not road_fc:
            st.sidebar.warning("No roads inside the buffer.")
    except Exception as e:
        st.sidebar.error(f"Roads error: {e}")
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
                st.sidebar.warning(f"No {lab} inside the buffer.")
        except Exception as e:
            st.sidebar.error(f"{lab} error: {e}")
on_dem, on_con, on_rd, on_wt = bool(dem_ov), bool(show_con and terr and terr[3]["features"]), bool(road_fc), bool(water)

cfg = {
    "basemap": basemap, "base": BASEMAPS[basemap], "colors": COLORS, "layers": [dict(light(l), fc=site_fc(l)) for l in layers],
    "sig": "|".join(sig),
    "pm": [{"name": n, "url": f"http://127.0.0.1:{PROXY_PORT}/{quote(pm_assets[n])}",
            "alt": f"{HF_BASE}/resolve/main/{quote(pm_assets[n])}"} for n in chosen] if show_topo else [],
    "osm_index": osm_json,
    "dem": {"url": dem_ov[0], "b": dem_ov[1]} if on_dem else None,
    "contours": terr[3] if on_con else None, "con_labels": terr[5] if on_con else None,
    "roads": road_fc, "water": water or None, "wcol": {k: v[2] for k, v in DRAIN.items()},
    "cc": CON_COL, "rc": ROAD_COL,
}

# ---------------- Page ----------------
project_title = st.text_input(
    "Project Title",
    value="Preliminary Assessment Study for the 14MW Solar Project at Gadgi, Bidar District, Karnataka, India")
st.markdown("---")

HTML = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/pmtiles@3.0.7/dist/pmtiles.js"></script>
<style>
html,body{margin:0}
.cl{display:block;width:30px;text-align:center;font:bold 10px Arial,sans-serif;color:#000;
    text-shadow:-1px -1px 0 #fff,1px -1px 0 #fff,-1px 1px 0 #fff,1px 1px 0 #fff;pointer-events:none}
</style>
<div id="m" style="height:480px;border:1px solid #ccc;background:#fff"></div>
<script>
const C = __CFG__;
const log = m => console.error(m);
let keep = null;  // last view of this tab, restored unless the uploaded vectors changed
try { keep = JSON.parse(sessionStorage.getItem('topoView')); } catch (e) {}
if (keep && keep.sig !== C.sig) keep = null;
const map = L.map('m', {preferCanvas: true, zoomSnap: 0.5}).setView(keep ? keep.c : [21, 79], keep ? keep.z : 5.5);
map.on('moveend', () => { try { sessionStorage.setItem('topoView', JSON.stringify({sig: C.sig, c: map.getCenter(), z: map.getZoom()})); } catch (e) {} });
L.control.scale().addTo(map);
if (C.base) L.tileLayer(C.base, {maxZoom: 20, attribution: C.basemap}).addTo(map);
[['topo', 300], ['dem_pane', 310], ['water_pane', 315], ['con_pane', 320], ['conlab_pane', 325], ['road_pane', 330], ['osm_pane', 400]]
  .forEach(([n, z]) => { map.createPane(n).style.zIndex = z; });

if (C.dem) L.imageOverlay(C.dem.url, C.dem.b, {pane: 'dem_pane', opacity: 0.75}).addTo(map);
if (C.water) Object.entries(C.water).forEach(([k, fc]) => L.geoJSON(fc, {pane: 'water_pane', interactive: false,
  style: f => f.geometry.type.indexOf('Line') >= 0
    ? {color: C.wcol[k], weight: 1.6, fill: false}
    : {color: C.wcol[k], weight: 0.8, fillColor: C.wcol[k], fillOpacity: 0.55}}).addTo(map));
if (C.contours) L.geoJSON(C.contours, {pane: 'con_pane', interactive: false, style: {color: C.cc, weight: 0.8}}).addTo(map);
if (C.con_labels) {  // black elevation labels, shown from zoom 12 up
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

let all = null;  // bounds of all vectors + buffers
C.layers.forEach(lay => {
  const n = lay.bufs.length, dash = lay.dash ? '8,6' : null;
  const grow = g => { all = all ? all.extend(g.getBounds()) : g.getBounds(); };
  lay.bufs.forEach((b, i) => grow(L.geoJSON(b.gj, {style: {color: C.colors[i == 0 ? 0 : i == n - 1 ? 2 : 1], weight: 2.5, dashArray: dash, fill: false},
    pointToLayer: (f, ll) => L.marker(ll, {icon: wmIcon})}).bindTooltip(b.label, {sticky: true}).addTo(map)));
  grow(L.geoJSON(lay.fc, {
    style: {color: C.colors[0], weight: 2, dashArray: dash, fill: true, fillOpacity: 0},  // clickable inside, looks hollow
    pointToLayer: (f, ll) => L.marker(ll, {icon: wmIcon}),
    onEachFeature: (f, layer) => {
      const p = f.properties || {};
      const nk = Object.keys(p).find(k => k.toLowerCase() === 'name' && p[k] !== null && p[k] !== '');
      layer.bindTooltip(nk ? String(p[nk]) : lay.name, {sticky: true});
      const rows = Object.entries(p).filter(([k, v]) => v !== null && v !== '')
        .map(([k, v]) => '<tr><td style="padding:2px 8px 2px 0;color:#555;vertical-align:top"><b>' + esc(k) + '</b></td><td style="padding:2px 0;word-break:break-word">' + esc(v) + '</td></tr>').join('');
      layer.bindPopup('<div style="font:12px Arial,sans-serif"><div style="font-weight:bold;margin-bottom:4px">' + esc(lay.name) + '</div>'
        + (rows ? '<table>' + rows + '</table>' : '<i>No attributes</i>') + '</div>', {maxWidth: 340, maxHeight: 260});
    }}).addTo(map));
});
if (all && !keep) map.fitBounds(all.pad(0.1));

(async () => {
  for (const p of C.pm) {  // each sheet is independent: local proxy first, then direct HF
    let ok = false;
    for (const [label, url] of [['local proxy', p.url], ['Hugging Face direct', p.alt]]) {
      try {
        const src = new pmtiles.PMTiles(url);
        const h = await src.getHeader();
        pmtiles.leafletRasterLayer(src, {pane: 'topo', minZoom: h.minZoom, maxZoom: h.maxZoom, opacity: 0.85, keepBuffer: 4}).addTo(map);
        ok = true; break;
      } catch (e) { log(p.name + ': ' + label + ' failed -> ' + (e && e.message || e)); }
    }
    if (!ok) log(p.name + ': SKIPPED (all sources failed)');
  }
})();

if (C.osm_index) {
  L.geoJSON(C.osm_index, {
    pane: 'osm_pane',
    style: () => ({color: '#ff4400', weight: 1, fill: false, fillOpacity: 0}),
    onEachFeature: function(feature, layer) {
      const p = feature.properties || {}, sheetName = p.EVEREST_SH || Object.values(p).find(v => typeof v === 'string') || 'N/A';
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

_ver = lambda *keys: tuple(sorted((k, _S.sig.get(p)) for k, p in assets.items() if any(w in k.lower() for w in keys)))
leg_bytes = sticky(("legend", _ver("legend_t")), lambda: legend_image(assets))
logo_bytes = sticky(("logo", _ver("logo")), lambda: logo_image(assets))
col_map, col_leg = st.columns([3.2, 1.1])
with col_map:
    show_html(HTML.replace("__CFG__", json.dumps(cfg).replace("</", "<\\/")), 490)
    c1, c2 = st.columns([1, 2])
    c1.markdown("**Source : Survey of India**")
    c2.markdown(f"**Topo sheet No. {sheets_str}**")
with col_leg:
    with st.container(height=480, border=True):
        st.markdown("### **Legends**")
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
        st.markdown(lg or "Upload a vector to see its legend.", unsafe_allow_html=True)
        if leg_bytes:
            st.image(leg_bytes, caption="CONVENTIONAL SYMBOLS", width="stretch")

st.markdown("---")

for jid, j in list(JOBS.items()):  # drop finished jobs (and their PNG bytes) after JOB_TTL
    if j["status"] != "running" and j.get("end") and time.time() - j["end"] > JOB_TTL:
        JOBS.pop(jid, None)
job = JOBS.get(st.session_state.get("job_id"))
refresh_job(job)

# map types offered = layers currently open (contours are exported together with the DEM only)
opts = [k for k, ok in (("Toposheet Map", show_topo and chosen), ("DEM - Elevation Map", on_dem),
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
            cancel_job(job)
            st.session_state.pop("polling", None)
            st.rerun()
    elif cb.button("🚀 Generate High-Res Map", width="stretch"):
        hd, hc, hr, hw, hs = KINDS[kind]
        topo = not any(KINDS[kind])
        if topo and not chosen:
            st.warning("Select at least one toposheet in the sidebar first.")
        elif hs and not layers:
            st.warning("Upload at least one vector first (the satellite map shows the vectors).")
        else:
            if job:
                JOBS.pop(st.session_state.get("job_id"), None)
            jid = uuid.uuid4().hex
            extra = {"dem": terr[:3] if hd else None, "contours": terr[3] if hc else None,
                     "roads": road_fc if hr else None, "water": water if hw else None,
                     "logo": None if topo else logo_bytes,
                     # toposheet map keeps its white base; satellite map is always Google Satellite; others follow the base map
                     "base": None if topo else SAT_BASE if hs else BASEMAPS[basemap]}
            JOBS[jid] = start_export([pm_assets[n] for n in chosen] if topo else [], layers, project_title, sheets_str,
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


# data exports above the map export: KML (site keeps its original features + attributes; one file per site / buffer)
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
                        b["label"], [(b["label"], g) for g in fc_geoms(b["gj"])], buf_color(i, len(l["bufs"])))),
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
