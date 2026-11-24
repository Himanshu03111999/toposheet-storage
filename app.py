"""DREAM dMRV Engine v6.1 - farmer-reported vs Sentinel-1 verified AWD (GEE + CHIRPS + ESA acquisition plan)
Run:  streamlit run dmrv_awd.py        (needs Streamlit >= 1.39)
  * Landing page only before sign-in (side panel closed until you click the CTA); smaller fonts
  * After sign-in: one horizontal bar  [6 tabs] ... [◀ ▶] [💾 Save analysis]
  * "Plots to audit" sits beside "Inspect plot" again
  * Run analysis runs in a background thread; progress is the vertical bar on the right (top to bottom)
  * Saved analyses live in Hugging Face folder  dmrv_analysis/<id>/  (needs HF_TOKEN with write access)
"""
import os, io, re, json, base64, tempfile, time, hashlib, hmac, shutil, math, uuid, gzip, pickle, threading, datetime as dt
import numpy as np, pandas as pd, geopandas as gpd
import streamlit as st, folium, ee
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import mapping
from folium.plugins import Fullscreen, MousePosition
from streamlit_folium import st_folium

st.set_page_config(page_title="AWD dMRV Platform", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")
HF_REPO = "himanshuguptta/geotiff-hub"
HF_FOLDER = "dmrv_analysis"
DISPLAY_TZ = "Asia/Kolkata"
GEE_PROJECT = os.getenv("EE_PROJECT", "virtual-strata-442411-p6")
SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT", "himanshu-guptta1@virtual-strata-442411-p6.iam.gserviceaccount.com")
KEY_PATH = os.getenv("EE_PRIVATE_KEY_PATH", "C:/Users/Syste/Downloads/virtual-strata-442411-p6-d7fa3aa0b1cf.json")
AUTH_USER = os.getenv("DMRV_USER", "kosher"); AUTH_PASS = os.getenv("DMRV_PASS", "Himanshugis")   # move to env / st.secrets
AHEAD = 20                                                                                       # default satellite-path look-ahead (days)
NAV = [("🗺 Spatial Audit", "Spatial Audit"), ("🌿 Graph CH4 & Stackable Export", "Export"), ("🛰 Multispectral Analysis", "Multispectral Analysis"),
       ("🧪 Ground Truth Verification", "Ground truth"), ("📍 Satellite Path", "Satellite Path"), ("🌾 Paddy Identification 🔒", "Paddy")]

# logout link (full reload = new session = signed out)
if "logout" in st.query_params:
    st.session_state.clear(); st.query_params.clear()
# keep the "plots to audit" widget value alive even on runs where the widget is not drawn (e.g. Saved view)
if "n_rows_w" in st.session_state: st.session_state["n_rows_w"] = st.session_state["n_rows_w"]

# ============================================================ SENTINEL-1 ACQUISITION PLAN (ESA KML)
S1_PAGE = "https://sentinels.copernicus.eu/copernicus/sentinel-1/acquisition-plans"
_KML_RX = re.compile(r'href=["\']([^"\']*?/documents/d/sentinel/(s1[abcd])_mp_user_(\d{8}t\d{6})_(\d{8}t\d{6})[^"\']*)', re.I)
def _fname_t(s): return pd.Timestamp(dt.datetime.strptime(s.upper(), "%Y%m%dT%H%M%S"))

def list_plan_links(html):
    out = {}
    for href, sat, a, b in _KML_RX.findall(html):
        path = "/documents/d/sentinel/" + href.split("/documents/d/sentinel/")[1].split("?")[0]
        out[path] = ("https://sentinels.copernicus.eu" + path, sat.upper(), _fname_t(a), _fname_t(b))
    return list(out.values())

@st.cache_data(show_spinner=False, ttl=3 * 3600)
def fetch_plan_kmls(ahead=25):
    import requests
    from concurrent.futures import ThreadPoolExecutor
    H = {"User-Agent": "Mozilla/5.0"}
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    allk = list_plan_links(requests.get(S1_PAGE, headers=H, timeout=30).text)
    if not allk: raise RuntimeError("ESA page loaded but no plan links were found (layout may have changed).")
    links = [l for l in allk if l[3] >= now and l[2] <= now + pd.Timedelta(days=ahead)]
    if not links: raise RuntimeError(f"ESA lists {len(allk)} plans but none cover the next {ahead} days yet.")
    def get(l):
        for u in (l[0], l[0] + "?download=true"):
            try:
                r = requests.get(u, headers=H, timeout=90)
                if r.ok and b"<kml" in r.content[:2000].lower():
                    return (f"{l[1]}_{l[2]:%Y%m%dT%H%M%S}_{l[3]:%Y%m%dT%H%M%S}.kml", r.content)
            except Exception: pass
    with ThreadPoolExecutor(max_workers=6) as ex: got = tuple(x for x in ex.map(get, links) if x)
    if not got: raise RuntimeError(f"Found {len(links)} plan link(s) but none downloaded.")
    return got

@st.cache_data(show_spinner=False)
def read_footprints(kmls, bbox):
    """IW + VV/VH datatake footprints. Where files overlap, the file with the latest start wins."""
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon, box
    bb, rows, covered = box(*bbox), [], []
    def span(n):
        m = re.search(r"(\d{8}T\d{6})_(\d{8}T\d{6})", n.upper())
        return (_fname_t(m.group(1)), _fname_t(m.group(2))) if m else (pd.Timestamp.min, pd.Timestamp.max)
    for name, raw in sorted(kmls, key=lambda k: span(k[0])[0], reverse=True):
        new = []
        for pm in ET.parse(io.BytesIO(raw)).getroot().iter():
            if not pm.tag.endswith("Placemark"): continue
            d = {x.get("name"): (x.findtext("{*}value") or "").strip() for x in pm.iter() if x.tag.endswith("Data")}
            co = next((k.text for k in pm.iter() if k.tag.endswith("coordinates") and k.text), None)
            if not co or not d.get("ObservationTimeStart") or not (d.get("Mode") == "IW" and d.get("Polarisation") == "DV"): continue
            pts = [tuple(map(float, q.split(",")[:2])) for q in co.split() if "," in q]
            if len(pts) < 4: continue
            poly, t = Polygon(pts), pd.Timestamp(d["ObservationTimeStart"])
            if poly.is_valid and poly.intersects(bb) and not any(a <= t <= b for a, b in covered):
                new.append(dict(sat=d.get("SatelliteId", "S1"), time=t, geometry=poly))
        rows += new; covered.append(span(name))
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else gpd.GeoDataFrame(columns=["sat", "time", "geometry"], geometry="geometry", crs=4326)

_TZF = []
def tz_for(lon, lat):
    """Best IANA time zone for a location (timezonefinder if installed, else India box, else nearest whole-hour zone)."""
    try:
        from timezonefinder import TimezoneFinder
        if not _TZF: _TZF.append(TimezoneFinder())
        z = _TZF[0].timezone_at(lng=float(lon), lat=float(lat))
        if z: return z
    except Exception: pass
    if 68 <= lon <= 98 and 6 <= lat <= 37: return "Asia/Kolkata"
    h = int(round(lon / 15)); return f"Etc/GMT{-h:+d}" if h else "UTC"

def to_local(t, tz):
    try: return pd.Timestamp(t).tz_localize("UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception: return ""

def sat_plan(gdf, ahead=AHEAD):
    """All planned IW VV/VH passes over every plot in the next `ahead` days, with local time per plot."""
    try:
        kmls = fetch_plan_kmls(ahead + 5)
        fp = read_footprints(kmls, tuple(gdf.total_bounds + np.array([-1, -1, 1, 1])))
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        L = fp[(fp.time >= now) & (fp.time <= now + pd.Timedelta(days=ahead))].drop_duplicates(["sat", "time"]).reset_index(drop=True)
        P = gpd.GeoDataFrame({"pid": gdf.pid.to_numpy()}, geometry=gdf.geometry.to_numpy(), crs=4326)
        ps = (pd.DataFrame(gpd.sjoin(P, L[["sat", "time", "geometry"]], predicate="intersects")[["pid", "sat", "time"]])
              if not L.empty else pd.DataFrame(columns=["pid", "sat", "time"])).sort_values(["pid", "time"])
        grp = {k: v for k, v in ps.groupby("pid")}
        rows, tips = [], {}
        for p in gdf.itertuples():
            tz, sub = tz_for(p.lon_c, p.lat_c), grp.get(p.pid)
            if sub is None or sub.empty:
                rows.append(dict(Plot=p.pid, Satellite="", Pass_UTC=f"No pass in next {ahead} days", Pass_local="", Time_zone=tz, Days_ahead=None)); tips[p.pid] = "No pass"
                continue
            lst = []
            for s in sub.itertuples():
                loc = to_local(s.time, tz)
                rows.append(dict(Plot=p.pid, Satellite=s.sat, Pass_UTC=f"{s.time:%Y-%m-%d %H:%M:%S}", Pass_local=loc, Time_zone=tz,
                                 Days_ahead=round((s.time - now).total_seconds() / 86400, 1)))
                lst.append(f"{s.sat} {loc}")
            tips[p.pid] = "<br>".join(lst[:15]) + (f"<br>…+{len(lst) - 15} more" if len(lst) > 15 else "") + f"<br>({tz})"
        if not L.empty:
            c = L.geometry.centroid
            L["utc"] = L.time.dt.strftime("%Y-%m-%d %H:%M:%S")
            L["local"] = [to_local(t, tz_for(x, y)) for t, x, y in zip(L.time, c.x, c.y)]
        return dict(tbl=pd.DataFrame(rows), lines=L, tips=tips, kmls=[n for n, _ in kmls], n_passes=len(ps), ahead=ahead)
    except Exception as e:
        return dict(err=str(e))

# ============================================================ STYLE
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html{font-size:12.5px;} html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(180deg,#E4F0FE 0%,#F3F8FF 45%,#EAF3FC 100%);}
[data-testid=stSidebar]{background:linear-gradient(180deg,#0A2A57 0%,#103A78 100%);border-right:none;position:relative;}
[data-testid=stSidebarContent],[data-testid=stSidebarHeader]{position:static !important;}
[data-testid=stSidebar] [data-testid=stSidebarUserContent]{padding-top:1rem;}
[data-testid=stSidebarCollapseButton] *{color:#fff !important;}
[data-testid=stSidebarCollapseButton]{position:absolute !important;top:50%;right:6px;transform:translateY(-50%);z-index:50;background:rgba(255,255,255,.14);border-radius:8px;}
[data-testid=stSidebar] [data-testid=stMarkdownContainer] p,[data-testid=stSidebar] label p{color:#DDE8FA !important;font-size:.78rem;}
[data-testid=stSidebar] [data-testid=stCaptionContainer] *{color:#8FB0E3 !important;}
[data-testid=stSidebar] [data-testid=stExpander]{border:1px solid rgba(255,255,255,.18) !important;border-radius:10px;background:rgba(255,255,255,.06) !important;}
[data-testid=stSidebar] [data-testid=stExpander] details,[data-testid=stSidebar] [data-testid=stExpander] details>div,[data-testid=stSidebar] [data-testid=stExpanderDetails],[data-testid=stSidebar] [data-testid=stExpander] summary{background:transparent !important;}
[data-testid=stSidebar] [data-testid=stExpander] summary,[data-testid=stSidebar] [data-testid=stExpander] summary *{color:#fff !important;font-weight:600;font-size:.8rem;}
[data-testid=stSidebar] [data-testid=stForm]{border:none !important;padding:0 !important;background:transparent !important;}
[data-testid=stSidebar] [data-testid=stSliderThumbValue],[data-testid=stSidebar] [data-testid=stTickBarMin],[data-testid=stSidebar] [data-testid=stTickBarMax]{color:#fff !important;}
[data-stale="true"],div[data-stale="true"]{opacity:1 !important;transition:none !important;}
.sb-brand{color:#fff;font-weight:800;font-size:1.05rem;} .sb-brand span{color:#7DB8F5;}
.sb-h{color:#7DB8F5;font-size:.62rem;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;margin:14px 0 4px 0;}
.block-container{padding-top:.9rem;padding-bottom:1rem;max-width:1600px;}
#MainMenu,footer,[data-testid=stMainMenu],[data-testid=stToolbarActions],[data-testid=stDecoration],[data-testid=stStatusWidget],.stAppDeployButton{display:none !important;}
header[data-testid=stHeader]{background:transparent;height:0;}
[data-testid=stSidebarCollapsedControl],[data-testid=collapsedControl],[data-testid=stExpandSidebarButton]{position:fixed !important;top:50%;left:0;transform:translateY(-50%);z-index:99999;
 display:flex !important;visibility:visible !important;background:#0A2342;border-radius:0 10px 10px 0;padding:10px 4px;box-shadow:2px 2px 8px rgba(10,35,66,.35);}
[data-testid=stSidebarCollapsedControl] *,[data-testid=collapsedControl] *,[data-testid=stExpandSidebarButton] *{color:#fff !important;fill:#fff !important;}
.hero{background:#fff;border:1px solid #D6E2F3;border-radius:14px;overflow:hidden;box-shadow:0 3px 10px rgba(10,35,66,.10);margin-bottom:10px;}
.hero .strip{background:linear-gradient(90deg,#071B3A,#0F3A7D 55%,#1D5FD1);color:#DCE9FF;font-size:.56rem;letter-spacing:.8px;text-transform:uppercase;font-weight:600;padding:3px 16px;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;}
.hero .body{display:flex;align-items:center;gap:10px;padding:5px 16px;flex-wrap:wrap;}
.hero img{height:28px;} .hero h2{margin:0;font-weight:800;font-size:1.2rem;line-height:1.05;color:#0A2342;}
.hero small{display:block;color:#5E6E80;font-size:.72rem;} .sp{flex:1;}
.chip{display:inline-flex;align-items:center;border:1px solid #D0DCEE;background:#fff;color:#0A2342;border-radius:7px;padding:3px 9px;font-size:.68rem;font-weight:500;text-decoration:none;}
.chip.ok{background:#0A2342;color:#fff !important;border-color:#0A2342;font-weight:600;} a.chip.ok:hover{background:#B42318;border-color:#B42318;}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;} .dot.on{background:#22C55E;box-shadow:0 0 0 3px rgba(34,197,94,.28);} .dot.off{background:#9CA3AF;}
/* ---- one horizontal bar: 6 tabs + (◀ ▶) + Save ---- */
.st-key-nav button{border-radius:10px;border:1px solid #C9D8EE;background:#fff;min-height:0;padding:7px 6px;box-shadow:none;}
.st-key-nav button p{color:#0A2342;font-size:.72rem;font-weight:600;line-height:1.2;white-space:normal;}
.st-key-nav button[kind=primary]{background:#0A2342 !important;border-color:#0A2342 !important;} .st-key-nav button[kind=primary] p{color:#fff !important;}
.st-key-nav button:disabled{opacity:.45;}
.st-key-nav .st-key-arrows [data-testid=stHorizontalBlock]{gap:3px !important;}
.st-key-nav .st-key-arrows button{background:#EAF1FC;border-color:#B5CBEA;padding:7px 0;}
.st-key-nav .st-key-savebox button{background:linear-gradient(90deg,#1D5FD1,#0FB5A6) !important;border:none !important;}
.st-key-nav .st-key-savebox button p{color:#fff !important;font-weight:700;}
.kpi{background:#fff;border:1px solid #DCE8F7;border-radius:14px;padding:12px 16px;box-shadow:0 4px 14px rgba(47,128,237,.07);display:flex;gap:14px;align-items:center;}
.kpi .ic{width:40px;height:40px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;}
.kpi .v{font-size:1.15rem;font-weight:800;color:#12325C;line-height:1.1;} .kpi .l{color:#6B84A3;font-size:.7rem;margin-top:2px;}
.card{background:#fff;border:1px solid #DCE8F7;border-radius:14px;padding:14px 18px;box-shadow:0 4px 14px rgba(47,128,237,.07);}
.card h4{margin:0 0 8px 0;color:#12325C;font-size:1rem;}
.row{display:flex;justify-content:space-between;gap:14px;padding:4px 0;border-bottom:1px dashed #E6EEF9;font-size:.78rem;}
.row span:first-child{color:#6B84A3;} .row span:last-child{color:#12325C;font-weight:600;}
.badge{padding:3px 10px;border-radius:999px;font-size:.75rem;font-weight:700;}
.b-ok{background:#DFF7E8;color:#16874A;} .b-warn{background:#FFF1D6;color:#B7791F;} .b-bad{background:#FDE5E5;color:#C53030;} .b-gray{background:#EDF1F7;color:#5A6B82;}
.ev{background:#fff;border:1px solid #DCE8F7;border-left:6px solid var(--c);border-radius:14px;padding:12px 16px;}
.ev b{color:#12325C;} .ev .m{color:#6B84A3;font-size:.82rem;margin-top:4px;line-height:1.5;}
.vbar{position:fixed;right:10px;top:64px;bottom:18px;width:12px;background:#D6E4F7;border-radius:8px;z-index:99998;overflow:hidden;}
.vfill{width:100%;background:linear-gradient(180deg,#1D5FD1,#0FB5A6);border-radius:8px;}
.vlbl{position:fixed;right:30px;top:64px;z-index:99998;background:#0A2342;color:#fff;border-radius:8px;padding:5px 10px;font-size:.72rem;font-weight:600;}
.sv{background:#fff;border:1px solid #DCE8F7;border-radius:14px;overflow:hidden;box-shadow:0 4px 14px rgba(47,128,237,.07);margin-bottom:6px;}
.sv img{width:100%;height:150px;object-fit:cover;display:block;background:#0A2342;}
.sv .b{padding:10px 12px;} .sv .b b{color:#12325C;font-size:.88rem;display:block;margin-bottom:4px;}
.sv .m{color:#6B84A3;font-size:.72rem;line-height:1.65;} .sv .m em{font-style:normal;color:#12325C;font-weight:600;}
[data-testid=stMain] [data-testid=stExpander]{background:#fff;border:1px solid #DCE8F7;border-radius:14px;}
[data-testid=stMain] label[data-baseweb=radio]{background:#fff;border:1px solid #C9D8EE;border-radius:10px;padding:5px 13px;margin:0 !important;}
[data-testid=stMain] label[data-baseweb=radio]>div:first-child{display:none;}
[data-testid=stMain] label[data-baseweb=radio]:has(input:checked){background:#0A2342;border-color:#0A2342;}
[data-testid=stMain] label[data-baseweb=radio]:has(input:checked) p{color:#fff !important;}
[data-testid=stMain] div[role=radiogroup]{gap:6px;flex-wrap:wrap;}
.lg-card{display:flex;gap:11px;align-items:center;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.2);border-radius:12px;padding:10px 12px;margin-bottom:8px;}
.lg-card b{display:block;color:#fff;font-size:.85rem;} .lg-card span{display:block;color:#9DB9E6;font-size:.68rem;}
.lg-ic{width:34px;height:34px;flex:none;border-radius:10px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#1D5FD1,#0FB5A6);}
.lg-ic.ok{background:#16A34A;color:#fff;font-weight:800;}
[data-testid=stSidebar] [data-testid=stTextInput] input,[data-testid=stSidebar] textarea{background:rgba(255,255,255,.96) !important;color:#0A2342 !important;border-radius:9px;font-size:.8rem;}
[data-testid=stSidebar] button[kind=primary],[data-testid=stSidebar] button[kind=primaryFormSubmit]{background:linear-gradient(90deg,#1D5FD1,#0FB5A6) !important;border:none !important;color:#fff !important;font-weight:700;border-radius:10px;}
[data-testid=stSidebar] button[kind=secondary]{background:rgba(255,255,255,.08) !important;border:1px solid rgba(255,255,255,.28) !important;color:#fff !important;border-radius:10px;}
[data-testid=stSidebar] button p{color:#fff !important;}
[data-testid=stSidebar] [data-testid=stFileUploaderDropzone]{background:rgba(255,255,255,.08);border:1px dashed rgba(255,255,255,.4);border-radius:12px;}
[data-testid=stSidebar] [data-testid=stFileUploader] *{color:#fff !important;} [data-testid=stSidebar] [data-testid=stFileUploader] svg{fill:#fff !important;}
[data-testid=stSidebar] [data-testid=stFileUploaderFile]{background:rgba(255,255,255,.14) !important;border-radius:10px;padding:4px 8px;}
</style>""", unsafe_allow_html=True)

LANDING = r"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<style>
:root{--n:#0A2342;--b:#1D5FD1;--t:#0FB5A6;--l:#DCE8F7;--m:#6B84A3}
*{box-sizing:border-box}body{margin:0;font-family:Inter,sans-serif;color:var(--n);background:transparent}
.w{max-width:1400px;margin:auto;padding:2px 2px 16px}
.card{background:#fff;border:1px solid var(--l);border-radius:14px;box-shadow:0 4px 14px rgba(47,128,237,.08)}
.badge{display:inline-block;background:#fff;border:1px solid #D0DCEE;border-radius:99px;padding:4px 11px;font:600 9.5px 'JetBrains Mono';color:#0F3A7D;letter-spacing:.5px}
.badge i{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--t);margin-right:6px;animation:p 1.6s infinite}
@keyframes p{50%{opacity:.25}}
h1{font-size:27px;line-height:1.15;margin:10px 0;letter-spacing:-.5px;font-weight:800}
h2{font-size:18px;margin:0 0 3px;font-weight:800}.eyebrow{font:600 9.5px 'JetBrains Mono';color:var(--b);text-transform:uppercase;letter-spacing:1px}
p{color:#4A5F7A;line-height:1.55;font-size:12px;margin:5px 0}
.hero{display:grid;grid-template-columns:1.05fr 1fr;gap:22px;align-items:center;padding:22px 26px;border-radius:18px;color:#fff;
 background:linear-gradient(120deg,#071B3A,#0F3A7D 60%,#1D5FD1);position:relative;overflow:hidden}
.hero h1{color:#fff}.hero h1 span{background:linear-gradient(90deg,#7DB8F5,#5EEAD4);-webkit-background-clip:text;color:transparent}.hero p{color:#CFE0FA;font-size:12.5px}
.hero .badge{background:rgba(255,255,255,.1);border-color:rgba(255,255,255,.25);color:#DCE9FF}
.chips{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}
.chip{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.2);border-radius:10px;padding:8px}
.chip b{display:block;font-size:14px;color:#fff}.chip span{font-size:9.5px;color:#A9C4EE}
.mini{background:#fff;color:var(--n);padding:12px;border-radius:14px}.mini h4{margin:0;font-size:12px}
.sec{margin-top:26px}.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
.tab{border:1px solid #C9D8EE;background:#fff;color:var(--n);border-radius:9px;padding:5px 11px;font:600 10.5px Inter;cursor:pointer;transition:.2s}
.tab:hover{border-color:var(--b)}.tab.on{background:var(--n);color:#fff;border-color:var(--n)}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.cmp{padding:13px}.cmp .h{display:flex;justify-content:space-between;font-size:11px;font-weight:700;margin-bottom:8px}.cmp .h em{font:500 9.5px 'JetBrains Mono';color:var(--t);font-style:normal}
.cmp .s{display:grid;grid-template-columns:1fr 1fr;gap:8px}.cmp .s div{border-radius:10px;padding:8px;font-size:11px;line-height:1.5}
.bad{background:#FDECEC;border:1px solid #F8D0D0}.ok{background:#E6F1FF;border:1px solid #BFD7F7}.s b{display:block;font-size:9px;letter-spacing:.8px;text-transform:uppercase;margin-bottom:3px}
.bad b{color:#C53030}.ok b{color:var(--b)}
.pil{padding:14px;transition:.25s}.pil:hover{transform:translateY(-4px);box-shadow:0 12px 24px rgba(29,95,209,.18)}
.pil .ic{width:38px;height:38px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:8px;color:#fff}
.pil small{font:600 9.5px 'JetBrains Mono';color:var(--b)}.pil h3{margin:3px 0;font-size:13px}.pil p{font-size:11.5px}
.pipe{padding:18px;background:linear-gradient(135deg,#EEF5FF,#fff 60%,#E4F8F5)}
.steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}
.st{position:relative;overflow:hidden;text-align:left;background:#fff;border:1px solid #C9D8EE;border-radius:12px;padding:10px;cursor:pointer;font-family:Inter;transition:.2s}
.st b{display:block;font-size:12.5px;color:var(--n)}.st span{font:600 9.5px 'JetBrains Mono';color:var(--m)}
.st.on{border-color:var(--b);box-shadow:0 6px 16px rgba(29,95,209,.2)}.st.on span{color:var(--b)}
.st .bar{position:absolute;left:0;bottom:0;height:3px;width:0;background:linear-gradient(90deg,var(--b),var(--t))}.st.on .bar{animation:f 7s linear forwards}
@keyframes f{to{width:100%}}
.det{display:grid;grid-template-columns:1.2fr 1fr;gap:18px;background:#fff;border:1px solid var(--l);border-radius:12px;padding:16px;min-height:200px}
.tag{display:inline-block;background:#EDF1F7;border-radius:7px;padding:3px 9px;font-size:10.5px;font-weight:600;margin:4px 5px 0 0}
pre{margin:0;background:#0A2342;color:#CFE0FA;border-radius:10px;padding:12px;font:500 10.5px/1.6 'JetBrains Mono';overflow:auto}
.cta{margin-top:26px;text-align:center;padding:20px;border-radius:14px;background:linear-gradient(90deg,#0A2342,#1D5FD1);color:#fff;cursor:pointer;transition:.2s}
.cta:hover{filter:brightness(1.12);transform:translateY(-2px)}
.cta h3{margin:0 0 4px;font-size:16px}.cta p{color:#CFE0FA;margin:0;font-size:12px}
.cta button{margin-top:10px;background:#fff;color:var(--n);border:none;border-radius:9px;padding:7px 16px;font:700 11.5px Inter;cursor:pointer}
@media(max-width:900px){.hero,.det,.g2{grid-template-columns:1fr}.g4,.steps,.chips{grid-template-columns:1fr 1fr}}
</style>
<div class="w">
<div class="hero">
 <div><span class="badge"><i></i>AUDIT-GRADE RICE dMRV • VM0051 &amp; DREAM READY</span>
  <h1>Turn Paddy Water Cycles into <span>Bankable, Audit-Proof</span> Carbon Credits.</h1>
  <p>Compare farmer-reported aeration with Sentinel-1 radar and CHIRPS rainfall, plot by plot. Cloud-penetrating SAR, buffer gating and a flowering lock replace paper trails and manual spot checks.</p>
  <div class="chips"><div class="chip"><b>6–12 d</b><span>Sentinel-1 revisit</span></div><div class="chip"><b>−10 m</b><span>Edge-buffer gate</span></div>
  <div class="chip"><b>≥ 3 days</b><span>Creditable dry spell</span></div><div class="chip"><b>CHIRPS V3</b><span>Rain gap-fill</span></div></div></div>
 <div class="mini"><h4>🛰 Try it: drying trigger = dB_wet + Δ</h4>
  <svg id="ch" viewBox="0 0 330 150" style="width:100%"></svg>
  <div style="display:flex;align-items:center;gap:10px;font-size:11px"><span>Δ</span><input id="dl" type="range" min="0.5" max="8" step="0.5" value="3" style="flex:1"><b id="dv">3.0 dB</b></div>
  <div id="hit" style="font-size:11px;color:var(--m);margin-top:4px"></div></div>
</div>

<div class="sec"><div class="eyebrow">Eliminate audit discounts</div><h2>Why Manual AWD Programmes Fail Verification</h2>
 <p>Disconnected spreadsheets, recycled photos and missing drainage logs force verifiers to haircut credit issuance.</p>
 <div class="tabs" id="ft"></div><div class="g2" id="cg"></div></div>

<div class="sec"><div class="eyebrow">Enterprise architecture</div><h2>Built for Smallholder Scale and VVB Scrutiny</h2>
 <p>Each module of this workspace is designed to be defensible in front of an auditor.</p><div class="g4" id="pl" style="margin-top:12px"></div></div>

<div class="sec"><div class="card pipe"><div class="eyebrow">End-to-end operational flow</div><h2>The 4-Stage Digital Pipeline</h2>
 <p>Click a stage, or let it play. Each stage shows what the engine does and a sample output.</p>
 <div class="steps" id="sp"></div><div class="det"><div><span class="badge" id="pb"></span><h3 id="pt" style="font-size:16px;margin:8px 0 4px"></h3><p id="pd"></p><div id="pg"></div></div>
 <div><div style="font:600 10px 'JetBrains Mono';color:var(--m);margin-bottom:6px" id="ph"></div><pre id="pc"></pre></div></div></div></div>

<div class="cta" onclick="openPanel()"><h3>⬅ Ready when you are</h3><p>Open the side panel, sign in, upload your plots, set the parameters, and press ▶ Run analysis.</p><button onclick="openPanel();event.stopPropagation()">Open side panel</button></div>
</div>
<script>
const $=i=>document.getElementById(i);
const V=[-18.5,-19.8,-20.2,-19,-15.5,-13.8,-14.9,-19.5,-20,-16.2,-12.9,-18.8],wet=Math.min(...V),X=i=>18+i*28.5,Y=v=>10+(-8-v)/14*125;
function draw(){const d=+$('dl').value,tr=wet+d;$('dv').textContent=d.toFixed(1)+' dB';let n=0;
 let s=`<line x1="14" x2="326" y1="${Y(wet)}" y2="${Y(wet)}" stroke="#1D5FD1" stroke-dasharray="4 3"/><text x="16" y="${Y(wet)-4}" font-size="9" fill="#1D5FD1">dB_wet ${wet}</text>
 <line x1="14" x2="326" y1="${Y(tr)}" y2="${Y(tr)}" stroke="#E5484D" stroke-dasharray="4 3"/><text x="318" text-anchor="end" y="${Y(tr)-4}" font-size="9" fill="#E5484D">trigger ${tr.toFixed(1)}</text>
 <polyline fill="none" stroke="#0FB5A6" stroke-width="2.4" points="${V.map((v,i)=>X(i)+','+Y(v)).join(' ')}"/>`;
 V.forEach((v,i)=>{const h=v>=tr;n+=h;s+=`<circle cx="${X(i)}" cy="${Y(v)}" r="${h?5.5:3.5}" fill="${h?'#E5484D':'#0FB5A6'}" stroke="#fff" stroke-width="1.5"/>`});
 $('ch').innerHTML=s;$('hit').textContent=`${n} of ${V.length} passes flagged as drying. Raise Δ and fewer passes qualify.`}
$('dl').oninput=draw;draw();
const C=[['registry','01 · Farmer & plot registry','Buffer + clusters','Handwritten rosters double-count neighbouring plots and overlapping claims.','<b>Spatial gate:</b> every plot is shrunk by an edge buffer; small plots are merged into Cluster Polygons (DREAM A 2.3.3b).'],
['evidence','02 · Water-level evidence','Ground truth','Late manual logs and recycled gallery photos that fail audit scrutiny.','<b>Accuracy check:</b> tube observations are scored against satellite status; below 80% raises the deduction to 20%.'],
['sar','03 · Monsoon cloud cover','Sentinel-1 SAR','Optical satellites go blind in Kharif, exactly when methane peaks.','<b>All-weather C-band radar:</b> VV/VH backscatter shows flooded vs dry soil through the clouds.'],
['yield','04 · Flowering & rain risk','Lock + CHIRPS','Drying during flowering hurts yield, and rain between passes goes unnoticed.','<b>Flowering lock</b> keeps the window flooded; <b>CHIRPS rain</b> fills the gaps between satellite passes.']];
const F=[['all','All 4 guardrails'],...C.map(c=>[c[0],c[1].slice(5)])];
function filt(k){$('ft').innerHTML=F.map(f=>`<button class="tab ${f[0]==k?'on':''}" onclick="filt('${f[0]}')">${f[1]}</button>`).join('');
 $('cg').innerHTML=C.filter(c=>k=='all'||c[0]==k).map(c=>`<div class="card cmp"><div class="h"><span>${c[1]}</span><em>${c[2]}</em></div><div class="s"><div class="bad"><b>Legacy paper MRV</b>${c[3]}</div><div class="ok"><b>AWD dMRV</b>${c[4]}</div></div></div>`).join('');fit()}
$('pl').innerHTML=[['🗺','#1D5FD1','01 · SPATIAL AUDIT','Buffer gate &amp; clusters','Plots without a core pixel are merged into cluster polygons, then mapped over satellite imagery.'],
['🛰','#0FB5A6','02 · SAR &amp; LOCK','dB_wet and flowering lock','Flooded baseline, drying trigger and a lock window around heading, from Sentinel-1.'],
['🌧','#0F3A7D','03 · CH₄ &amp; EXPORT','Rain-aware credit maths','Gap-filling, the 3-day rule and CHIRPS rainfall turn dry days into tCO₂e, with CSV export.'],
['🧪','#0A2342','04 · GROUND TRUTH','Accuracy against the field','Confusion matrix of tube vs satellite, with the uncertainty deduction set automatically.']]
.map(p=>`<div class="card pil"><div class="ic" style="background:${p[1]}">${p[0]}</div><small>${p[2]}</small><h3>${p[3]}</h3><p>${p[4]}</p></div>`).join('');
const S=[['Spatial GIS','Map &amp; Gate','STAGE 01 · PLOTS IN','Load plots and apply the core-zone buffer','Upload GeoJSON, KML, SHP, GPKG or a WKT table. Dates are read day-first, small plots are clustered, and unusable boundaries are rejected early.',['✓ Any vector format','✓ Cluster polygons','✓ Date check table'],'gate_report.json','{\n "plots_loaded": 1420,\n "buffer_m": 10,\n "passed": 1388,\n "clusters_formed": 12\n}'],
['Sentinel-1','Measure','STAGE 02 · RADAR','Extract VV/VH backscatter per plot','GRD scenes are speckle-filtered and averaged over each core zone, in parallel on the high-volume Earth Engine endpoint.',['✓ Multi-temporal filter','✓ VV, VH or VV+VH','✓ Parallel chunks'],'s1_timeseries.csv','pid,date,VV,VH\nP01,2026-07-04,-14.2,-20.8\nP01,2026-07-16,-12.9,-19.6\nP01,2026-07-28,-17.5,-23.1'],
['Rules engine','Audit','STAGE 03 · VERIFY','Compare farmer reports with satellite status','dB_wet sets the flooded baseline, the trigger flags drying, and CHIRPS rain fills the gaps. Only spells of 3 or more days are credited.',['✓ dB_wet + Δ trigger','✓ Flowering lock','✓ 3-day rule'],'plot_verdict.json','{\n "plot": "P01",\n "AE1": "CONFIRMED",\n "AE2": "DRY_NOT_CREDITED",\n "lock_violations": 0\n}'],
['Export','Quantify','STAGE 04 · CREDITS OUT','Convert verified dry days into net tCO₂e','Baseline CH₄ × scaling factor × GWP, minus the uncertainty deduction, only for VALID plots. Download audit CSVs and the parameter JSON.',['✓ Editable factors','✓ Audit-report CSV','✓ Parameters JSON'],'emission_ledger.json','{\n "avoided_kgCH4": 61.4,\n "gross_tCO2e": 1.72,\n "uncertainty_pct": 15,\n "net_tCO2e": 1.46\n}']];
let cur=0,tm;
function go(i){cur=i;$('sp').innerHTML=S.map((s,k)=>`<button class="st ${k==i?'on':''}" onclick="go(${k});play()"><span>${s[0]}</span><b>${s[1]}</b><div class="bar"></div></button>`).join('');
 const s=S[i];$('pb').innerHTML=s[2];$('pt').innerHTML=s[3];$('pd').innerHTML=s[4];$('pg').innerHTML=s[5].map(t=>`<span class="tag">${t}</span>`).join('');$('ph').textContent=s[6];$('pc').textContent=s[7]}
function play(){clearInterval(tm);tm=setInterval(()=>go((cur+1)%4),7000)}
go(0);play();filt('all');

/* ---- open the Streamlit side panel (CTA) ---- */
function openPanel(){try{const d=window.parent.document;
 const b=d.querySelector('[data-testid="stExpandSidebarButton"]')||d.querySelector('[data-testid="stSidebarCollapsedControl"] button')||d.querySelector('[data-testid="collapsedControl"] button')||d.querySelector('[data-testid="stSidebarCollapsedControl"]');
 if(b)b.click();}catch(e){}}
/* ---- keep the side panel closed when the site is first opened ---- */
setTimeout(()=>{try{const d=window.parent.document,sb=d.querySelector('[data-testid="stSidebar"]');
 if(sb&&sb.getAttribute('aria-expanded')==='true'){const c=d.querySelector('[data-testid="stSidebarCollapseButton"] button');if(c)c.click();}}catch(e){}},350);
/* ---- auto-fit iframe height to content ---- */
function fit(){try{const h=document.querySelector('.w').offsetHeight+30;window.frameElement.style.height=h+'px';}catch(e){}}
window.addEventListener('load',fit);window.addEventListener('resize',fit);try{new ResizeObserver(fit).observe(document.querySelector('.w'))}catch(e){}fit();
</script>
"""

def landing():
    components.html(LANDING, height=1500, scrolling=False)

# ============================================================ GEE + LOGO
@st.cache_resource
def init_gee():
    try:
        if "gee" in st.secrets:
            info = dict(st.secrets["gee"]); creds = ee.ServiceAccountCredentials(info["client_email"], key_data=json.dumps(info))
        else:
            creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
        try: ee.Initialize(creds, project=GEE_PROJECT, opt_url="https://earthengine-highvolume.googleapis.com")
        except Exception: ee.Initialize(creds, project=GEE_PROJECT)
        return True, ""
    except Exception as e:
        return False, str(e)

def hf_token():
    try: t = st.secrets.get("HF_TOKEN", None)
    except Exception: t = None
    return t or os.getenv("HF_TOKEN")

@st.cache_resource
def get_logo():
    try:
        from PIL import Image, ImageChops
        names = ("logo.png", "logo.jpg", "logo.jpeg")
        dirs = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
        p = next((os.path.join(d, n) for d in dirs for n in names if os.path.exists(os.path.join(d, n))), None)
        if p is None:
            from huggingface_hub import hf_hub_download
            tok = hf_token()
            for n in names[:2]:
                try: p = hf_hub_download(HF_REPO, n, repo_type="dataset", token=tok); break
                except Exception: pass
        im = Image.open(p).convert("RGB")
        bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).convert("L").point(lambda v: 255 if v > 25 else 0).getbbox()
        im = im.crop(bb) if bb else im
        im.thumbnail((600, 200)); buf = io.BytesIO(); im.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None

# process-wide store: survives Streamlit reruns, so background jobs keep running while the user clicks around
@st.cache_resource
def job_store(): return {"jobs": {}, "cache": {}}
STORE = job_store()

gee_ok, gee_err = init_gee()
logo = get_logo()

# ============================================================ DATA HELPERS
def _table_to_gdf(df):
    import shapely
    gcol = next((c for c in df.columns if c.strip().lower() in ("wkt_geom", "wkt", "geometry", "geom", "the_geom")), None)
    if gcol is None: raise ValueError("No WKT geometry column found (expected wkt_geom / wkt / geometry).")
    df = df.copy()
    return gpd.GeoDataFrame(df, geometry=shapely.force_2d(shapely.from_wkt(df.pop(gcol).astype(str).to_numpy())), crs=4326)

def load_vector(name, raw, pasted=""):
    import shapely
    if not name: return _table_to_gdf(pd.read_csv(io.StringIO(pasted.strip()), sep=None, engine="python", dtype=str))
    low = name.lower()
    if low.endswith((".csv", ".tsv", ".txt")): return _table_to_gdf(pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str))
    if low.endswith((".xlsx", ".xls")): return _table_to_gdf(pd.read_excel(io.BytesIO(raw), dtype=str))
    import fiona
    fiona.drvsupport.supported_drivers["KML"] = "rw"; fiona.drvsupport.supported_drivers["LIBKML"] = "rw"
    path = os.path.join(tempfile.mkdtemp(), name); open(path, "wb").write(raw)
    gdf = (gpd.read_file("zip://" + path) if low.endswith(".zip") else gpd.read_file(path)).to_crs(4326)
    gdf["geometry"] = gpd.GeoSeries(shapely.force_2d(np.asarray(gdf.geometry.values)), index=gdf.index, crs=4326)
    return gdf

@st.cache_data(show_spinner="Reading plots…")
def load_cached(name, raw, pasted): return load_vector(name, raw, pasted)

def parse_day(x):
    if x is None or (not isinstance(x, (pd.Timestamp, dt.date)) and (pd.isna(x) or not str(x).strip())): return pd.NaT
    if isinstance(x, (pd.Timestamp, dt.date)): return pd.Timestamp(x)
    t = str(x).strip()
    if len(t) >= 10 and t[:4].isdigit() and t[4] == "-": return pd.to_datetime(t[:10], errors="coerce")
    return pd.to_datetime(t, dayfirst=True, errors="coerce")

def pdate(s):
    """Vectorised day-first date parser (ISO and Excel dates stay year-month-day)."""
    if pd.api.types.is_datetime64_any_dtype(s): return s
    t = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso = t.str.match(r"^\d{4}-\d{2}-\d{2}").fillna(False).astype(bool)
    if iso.any(): out[iso] = pd.to_datetime(t[iso].str[:10], errors="coerce")
    rest = (~iso) & t.notna() & (t != "")
    if rest.any():
        try: out[rest] = pd.to_datetime(t[rest], dayfirst=True, errors="coerce", format="mixed")
        except (ValueError, TypeError): out[rest] = t[rest].apply(parse_day)
    return out

def auto_range(g, cols):
    """Default window from the file: earliest date - 10 d ... latest date (or anchor + 120 d) + 12 d, capped at today."""
    try:
        pc = lambda h: next((c for c in cols if h in c.lower() and "iso" not in c.lower()), None)
        d = pd.concat([pdate(g[c]) for c in map(pc, ("sowing", "transplant", "pipe", "aeration_1", "aeration_2")) if c]).dropna()
        a = pd.concat([pdate(g[c]) for c in map(pc, ("sowing", "transplant")) if c]).dropna()
        lo = d.min() - pd.Timedelta(days=10)
        hi = min(max(d.max(), a.max() + pd.Timedelta(days=120)) + pd.Timedelta(days=12), pd.Timestamp.today())
        if lo < hi: return lo.date(), hi.date()
    except Exception: pass
    return dt.date.today() - dt.timedelta(days=150), dt.date.today()

def _cached(key_parts, fn, on_progress=None):
    key = hashlib.md5(repr(key_parts).encode()).hexdigest(); store = STORE["cache"]; hit = store.get(key)
    if hit and time.time() - hit[0] < 3600:
        if on_progress: on_progress(1.0)
        return hit[1]
    val = fn(); store[key] = (time.time(), val)
    return val

def fetch_s1(feats_json, start, end, orbit, on_progress=None):
    return _cached(("s1", feats_json, start, end, orbit), lambda: _fetch_s1_raw(feats_json, start, end, orbit, on_progress), on_progress)

def fetch_chirps_all(pts_json, start, end, on_progress=None):
    return _cached(("rain", pts_json, start, end), lambda: _fetch_chirps_raw(pts_json, start, end, on_progress), on_progress)

def _fetch_s1_raw(feats_json, start, end, orbit, on_progress=None):
    """Mean VV/VH (dB, speckle-filtered) per plot per date, chunked and queried in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from shapely.geometry import shape
    feats = json.loads(feats_json)
    def _valid(f):
        try: return bool(np.isfinite(shape(f["geom"]).bounds).all())
        except Exception: return False
    feats = [f for f in feats if _valid(f)]
    if not feats: return pd.DataFrame()
    def key(f):
        b = shape(f["geom"]).bounds
        return (int((b[1] + b[3]) * 5), (b[0] + b[2]) / 2)
    feats.sort(key=key)
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).days, 1)
    size = int(max(20, min(300, 60000 // (2 * (days // (6 if orbit == "BOTH" else 12) + 1)))))
    def run(chunk, depth=0):
        try:
            bs = [shape(f["geom"]).bounds for f in chunk]
            region = ee.Geometry.Rectangle([min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)])
            fc = ee.FeatureCollection([ee.Feature(ee.Geometry(f["geom"]), {"pid": f["pid"]}) for f in chunk])
            col = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(start, end)
                   .filter(ee.Filter.eq("instrumentMode", "IW"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                   .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
            if orbit != "BOTH": col = col.filter(ee.Filter.eq("orbitProperties_pass", orbit))
            lin = col.select(["VV", "VH"]).map(lambda im: ee.Image(10).pow(im.divide(10)).copyProperties(im, im.propertyNames()))
            tm = lin.mean(); box = tm.focal_mean(3, "square", "pixels")
            flt = lin.map(lambda im: ee.Image(10).multiply(im.multiply(box.divide(tm)).log10())
                          .copyProperties(im, ["system:time_start", "system:index"])).sort("system:time_start")
            stats = flt.toBands().reduceRegions(fc, ee.Reducer.mean(), 10, tileScale=4)
            meta = ee.Dictionary({"idx": flt.aggregate_array("system:index"), "t": flt.aggregate_array("system:time_start")}).getInfo()
            if not meta["idx"]: return []
            rows = []
            for ft in stats.getInfo()["features"]:
                pr = ft["properties"]
                for i, t in zip(meta["idx"], meta["t"]):
                    vv, vh = pr.get(f"{i}_VV"), pr.get(f"{i}_VH")
                    if vv is not None and vh is not None: rows.append({"pid": pr["pid"], "VV": vv, "VH": vh, "t": t})
            return rows
        except Exception as e:
            print(f"S1 chunk of {len(chunk)} failed at depth {depth}: {e}", flush=True)
            if len(chunk) > 5 and depth < 2:
                h = len(chunk) // 2
                return run(chunk[:h], depth + 1) + run(chunk[h:], depth + 1)
            raise
    chunks = [feats[i:i + size] for i in range(0, len(feats), size)]
    out = []
    with ThreadPoolExecutor(max_workers=min(10, len(chunks))) as ex:
        for k_, fut in enumerate(as_completed([ex.submit(run, c) for c in chunks]), 1):
            out += fut.result()
            if on_progress: on_progress(k_ / len(chunks))
    df = pd.DataFrame(out)
    if df.empty: return df
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    return df.groupby(["pid", "date"], as_index=False)[["VV", "VH"]].mean().sort_values(["pid", "date"])

def _fetch_chirps_raw(pts_json, start, end, on_progress=None):
    """Daily CHIRPS V3 rain, one query per 0.05 deg pixel. Returns ({cell: DataFrame(date, rain)}, {plot id: cell})."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    pts = json.loads(pts_json); step = 0.05
    cell_of = {p: f"{int(np.floor(x / step))}_{int(np.floor(y / step))}" for p, x, y in pts}
    cells = sorted(set(cell_of.values()))
    per = max(1, min(40, 4500 // max((pd.Timestamp(end) - pd.Timestamp(start)).days, 1)))
    col = ee.ImageCollection("UCSB-CHC/CHIRPS/V3/DAILY_SAT").filterDate(start, end).select("precipitation")
    def run(chunk):
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([(int(c.split("_")[0]) + .5) * step, (int(c.split("_")[1]) + .5) * step]), {"cell": c}) for c in chunk])
        def per_img(im):
            t = im.get("system:time_start")
            return im.reduceRegions(fc, ee.Reducer.mean(), 5566).map(lambda f: ee.Feature(None, {"cell": f.get("cell"), "rain": f.get("mean"), "t": t}))
        return [f["properties"] for f in col.map(per_img).flatten().filter(ee.Filter.notNull(["rain"])).getInfo()["features"]]
    out, chunks = [], [cells[i:i + per] for i in range(0, len(cells), per)]
    with ThreadPoolExecutor(max_workers=min(6, len(chunks))) as ex:
        for k_, fut in enumerate(as_completed([ex.submit(run, c) for c in chunks]), 1):
            out += fut.result()
            if on_progress: on_progress(k_ / len(chunks))
    df = pd.DataFrame(out)
    if df.empty: return {}, cell_of
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    return {c: d[["date", "rain"]].sort_values("date") for c, d in df.groupby("cell")}, cell_of

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_s2(geom_json, start, end, cloud):
    """Sentinel-2 SR indices (mean inside the plot) per date, cloud/shadow masked with SCL."""
    geom = ee.Geometry(json.loads(geom_json))
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(start, end).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)))
    def proc(im):
        scl = im.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        r = im.select(["B2", "B3", "B4", "B8", "B11"]).divide(10000).updateMask(ok)
        nir, red, blue = r.select("B8"), r.select("B4"), r.select("B2")
        idx = ee.Image.cat([
            r.normalizedDifference(["B8", "B4"]).rename("NDVI"), r.normalizedDifference(["B8", "B11"]).rename("NDMI"),
            r.normalizedDifference(["B3", "B8"]).rename("NDWI"), r.normalizedDifference(["B3", "B11"]).rename("MNDWI"),
            nir.subtract(red).multiply(2.5).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).rename("EVI"),
            nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename("SAVI")])
        stt = idx.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9, bestEffort=True)
        clear = ok.rename("clear").reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9, bestEffort=True).get("clear")
        return ee.Feature(None, stt.set("t", im.get("system:time_start")).set("clear", clear).set("cloud", im.get("CLOUDY_PIXEL_PERCENTAGE")))
    df = pd.DataFrame([f["properties"] for f in col.map(proc).filter(ee.Filter.notNull(["NDVI"])).getInfo()["features"]])
    if df.empty: return df
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.normalize()
    return df.drop(columns="t").groupby("date", as_index=False).mean().sort_values("date")

# ---------------- cluster polygons
def build_units(g, utm, mode, id_col, incl_micro, buf, fm_col):
    """DREAM A 2.3.3b: plots with no core pixel after the buffer are aggregated into Cluster Polygons."""
    from shapely.ops import unary_union
    g = g.copy().reset_index(drop=True)
    pj = g.to_crs(utm)
    g["core_m2"] = pj.geometry.buffer(-buf).area; g["area_ha"] = pj.area / 1e4
    g["members"] = g.pid; g["n_members"] = 1; g["date_spread"] = np.nan
    if mode == "Off": return g, pd.DataFrame()
    n = len(g); parent = list(range(n))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    if mode == "By column" and id_col:
        key = g[id_col].astype(str).str.strip()
        ok = g[id_col].notna() & (key != "") & (key.str.lower() != "nan")
        for _, idxs in g[ok].groupby(key[ok]).groups.items():
            idxs = list(idxs)
            for j in idxs[1:]: parent[find(j)] = find(idxs[0])
    else:
        cm, ah = g.core_m2.to_numpy(), g.area_ha.to_numpy()
        cand = set(np.flatnonzero((cm <= 0) | ((ah < 0.5) if incl_micro else False)).tolist())
        bg = pj.geometry.buffer(1); sidx = bg.sindex
        for i in cand:
            for j in sidx.query(bg.iloc[i], predicate="intersects"):
                if j in cand and j != i: parent[find(i)] = find(j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    out, info, cnt, single, core_arr = [], [], 0, [], g.core_m2.to_numpy()
    for idxs in groups.values():
        if len(idxs) < 2 or (mode != "By column" and not (core_arr[idxs] <= 0).any()):
            single.extend(idxs); continue
        sub_ = g.loc[idxs]; cnt += 1
        geom = unary_union(list(pj.geometry.loc[idxs].buffer(1))).buffer(-1)
        rec = sub_.iloc[0].to_dict()
        rec.update(pid=f"CL{cnt:02d}", geometry=gpd.GeoSeries([geom], crs=utm).to_crs(4326).iloc[0], members=", ".join(sub_.pid),
                   n_members=len(sub_), core_m2=geom.buffer(-buf).area, area_ha=geom.area / 1e4)
        spread = []
        for k_ in ["trans", "sow", "pipe_dt", "ae1", "ae2"]:
            v = sub_[k_].dropna(); rec[k_] = v.median() if len(v) else pd.NaT
            if len(v) > 1: spread.append((v.max() - v.min()).days)
        rec["date_spread"] = max(spread) if spread else 0
        if fm_col: rec[fm_col] = f"{len(sub_)} plots (cluster)"
        out.append(rec)
        info.append(dict(Cluster=rec["pid"], Members=rec["members"], Plots=len(sub_), Core_ha=round(rec["core_m2"] / 1e4, 4), Max_date_spread_days=rec["date_spread"]))
    keep = g.loc[sorted(single)]
    if out: keep = pd.concat([keep, gpd.GeoDataFrame(out, geometry="geometry", crs=4326)], ignore_index=True)
    return gpd.GeoDataFrame(keep, geometry="geometry", crs=4326).reset_index(drop=True), pd.DataFrame(info)

# ============================================================ AUDIT ENGINE
def audit_plot(ts, p, P, rain=None):
    """Farmer records vs Sentinel-1 following DREAM Annex A 2.3 (dB_wet, delta), A 2.3.6-7 (daily status, 3-day rule), A 2.4 (lock)."""
    R = dict(status="NO_DATA", ts=ts.iloc[0:0], wet={}, aer={}, wet_date={}, ls=pd.NaT, le=pd.NaT, ev=[], extra=[], lock_hits=0,
             n_obs=0, aer_hits=0, daily=pd.DataFrame(columns=["date", "dry", "credited"]), runs=[], dry_days=0, dry_events=0)
    if pd.isna(p.anchor) or ts.empty: return R
    D = pd.Timedelta
    last_ev = max([d for d in [p.ae1, p.ae2] if pd.notna(d)] + [p.anchor + D(days=P["cycle"])])
    ts = ts[(ts.date >= p.anchor - D(days=5)) & (ts.date <= last_ev + D(days=P["tol"]))].copy()
    if ts.empty: return R
    ts["VV+VH"] = 10 * np.log10(10 ** (ts.VV / 10) + 10 ** (ts.VH / 10))
    w = ts[(ts.date >= p.anchor + D(days=P["w0"])) & (ts.date <= p.anchor + D(days=P["w1"]))]
    w = w if len(w) else ts
    for b in ["VV", "VH", "VV+VH"]:
        R["wet"][b] = w[b].min(); R["wet_date"][b] = w.loc[w[b].idxmin(), "date"]; R["aer"][b] = R["wet"][b] + P["delta"]
    det = P["det"]
    if P["lock"]:
        hd = p.anchor + D(days=P["heading"]); R["ls"], R["le"] = hd - D(days=10), hd + D(days=10)
    ts["inlock"] = (ts.date >= R["ls"]) & (ts.date <= R["le"])
    ts["aerated"] = ts[det] >= R["aer"][det]
    ts["audited"] = np.where(ts.inlock, R["wet"][det] - 1, ts[det])
    R.update(ts=ts, n_obs=len(ts), lock_hits=int((ts.aerated & ts.inlock).sum()), aer_hits=int((ts.aerated & ~ts.inlock).sum()))

    o = ts.sort_values("date").reset_index(drop=True)
    dry_obs = (o.aerated & ~o.inlock).to_numpy()
    days = pd.date_range(o.date.iloc[0], o.date.iloc[-1], freq="D")
    dry = pd.Series(False, index=days)
    rs = rain.set_index("date").rain if rain is not None and len(rain) else pd.Series(dtype=float)
    for i in range(len(o)):
        d0 = o.date.iloc[i]; dry.loc[d0] = bool(dry_obs[i])
        if i + 1 == len(o): break
        d1 = o.date.iloc[i + 1]; gap = pd.date_range(d0 + D(days=1), d1 - D(days=1), freq="D")
        if len(gap) == 0: continue
        if dry_obs[i] and dry_obs[i + 1]: dry.loc[gap] = not (rs.reindex(gap).max() >= P["rain"])
        elif dry_obs[i + 1]: dry.loc[gap[gap >= d0 + (d1 - d0) / 2]] = True
    if P["lock"]: dry[(dry.index >= R["ls"]) & (dry.index <= R["le"])] = False
    vals, runs, start = dry.to_numpy(), [], None
    for k, v in enumerate(vals):
        if v and start is None: start = k
        if start is not None and (not v or k == len(vals) - 1):
            e_ = k if v else k - 1
            runs.append((days[start], days[e_], e_ - start + 1)); start = None
    runs = [r for r in runs if r[2] >= 3]
    cred = pd.Series(False, index=days)
    for a_, b_, _ in runs: cred.loc[a_:b_] = True
    R.update(daily=pd.DataFrame({"date": days, "dry": vals, "credited": cred.to_numpy()}), runs=runs, dry_days=int(cred.sum()), dry_events=len(runs))

    tol = D(days=P["tol"]); matched = set(); dd = R["daily"]
    for name, d in [("Aeration 1", p.ae1), ("Aeration 2", p.ae2)]:
        if pd.isna(d): continue
        near = ts[(ts.date >= d - tol) & (ts.date <= d + tol)]
        if near.empty:
            R["ev"].append(dict(name=name, date=d, verdict="NO_PASS", obs=None, val=None, hit=False)); continue
        hit = near[near.aerated & ~near.inlock]
        credited = dd[dd.credited & (dd.date >= d - tol) & (dd.date <= d + tol)]
        pool = hit if len(hit) else near
        o_ = pool.iloc[(pool.date - d).abs().argsort().iloc[0]]
        verdict = "CONFIRMED" if len(credited) else ("DRY_NOT_CREDITED" if len(hit) else "NOT_CONFIRMED")
        R["ev"].append(dict(name=name, date=d, verdict=verdict, obs=o_.date, val=o_[det], hit=len(hit) > 0))
        if verdict == "CONFIRMED" and len(hit): matched.add(o_.date)
    R["extra"] = [r.date for r in ts[ts.aerated & ~ts.inlock].itertuples()
                  if r.date not in matched and all(abs((r.date - e["date"]).days) > P["tol"] for e in R["ev"])]
    verd = [e["verdict"] for e in R["ev"]]
    if R["lock_hits"] > 0: R["status"] = "LOCK_VIOLATION"
    elif verd and all(v == "CONFIRMED" for v in verd): R["status"] = "VALID"
    elif "CONFIRMED" in verd: R["status"] = "PARTIAL"
    elif verd: R["status"] = "NOT_CONFIRMED"
    else: R["status"] = "VALID" if R["dry_events"] > 0 else "NO_AERATION"
    return R

# ---------------- exports + extracted ranges (built once)
def make_exports(gdf, rep):
    g = gpd.GeoDataFrame(gdf[["pid", "geometry"]].merge(rep, left_on="pid", right_on="Plot_ID").drop(columns="pid"), geometry="geometry", crs=4326)
    for c in g.columns:
        if c != "geometry" and pd.api.types.is_datetime64_any_dtype(g[c]): g[c] = g[c].dt.strftime("%Y-%m-%d")
    out = {"csv": rep.to_csv(index=False).encode(), "geojson": g.to_json().encode(), "kml": None, "shp": None}
    d = tempfile.mkdtemp()
    try:
        g.to_file(os.path.join(d, "awd.kml"), driver="KML"); out["kml"] = open(os.path.join(d, "awd.kml"), "rb").read()
    except Exception: pass
    try:
        os.makedirs(os.path.join(d, "shp")); g.to_file(os.path.join(d, "shp", "awd_audit.shp"))
        out["shp"] = open(shutil.make_archive(os.path.join(d, "awd_shp"), "zip", os.path.join(d, "shp")), "rb").read()
    except Exception: pass
    return out

def ring_json(geom):
    """Polygon boundary as a JSON-style list of {"lat": .., "lng": ..} dictionaries (largest part for multi-polygons)."""
    try:
        if geom is None or geom.is_empty: return "[]"
        if hasattr(geom, "geoms"): geom = max(geom.geoms, key=lambda x: x.area)
        return json.dumps([{"lat": round(y, 7), "lng": round(x, 7)} for x, y, *_ in geom.exterior.coords])
    except Exception:
        return "[]"

def ranges_table(g, Q):
    """All uploaded rows: original attributes + satellite date range per plot + boundary as lat/lng list in column `range`."""
    d = pd.DataFrame(g.drop(columns="geometry")).copy()
    cs = [Q[k] for k in ("c_sw", "c_tr", "c_pp", "c_a1", "c_a2") if Q[k]]
    dd = pd.concat([pdate(d[c]) for c in cs], axis=1)
    a1, a2 = (Q["c_sw"], Q["c_tr"]) if Q["anchor_mode"] == "Sowing" else (Q["c_tr"], Q["c_sw"])
    anc = pdate(d[a1]) if a1 else pd.Series(pd.NaT, index=d.index)
    if a2: anc = anc.fillna(pdate(d[a2]))
    last = pd.concat([dd.max(axis=1), anc + pd.Timedelta(days=Q["cycle"])], axis=1).max(axis=1)
    frm = dd.min(axis=1) - pd.Timedelta(days=10)
    to = (last + pd.Timedelta(days=Q["tol"] + 6)).clip(upper=pd.Timestamp.today().normalize())
    if Q["dr"]:                                       # the range chosen in the side panel wins
        frm = frm.clip(lower=pd.Timestamp(Q["dr"][0])); to = to.clip(upper=pd.Timestamp(Q["dr"][1]))
    d["Range_from"], d["Range_to"], d["Range_days"] = frm.dt.strftime("%Y-%m-%d"), to.dt.strftime("%Y-%m-%d"), (to - frm).dt.days
    d["range"] = [ring_json(x) for x in g.geometry]
    return d

def run_pipeline(gdf, Q, prog, gdf_all):
    """Everything heavy. Runs in a background thread (no Streamlit calls in here!)."""
    from shapely import set_precision
    prog(0.02, "Reading dates")
    gdf = gdf.copy(); gdf["pid"] = gdf[Q["c_id"]].astype(str); gdf["trans"] = pdate(gdf[Q["c_tr"]])
    for k, c in [("sow", Q["c_sw"]), ("pipe_dt", Q["c_pp"]), ("ae1", Q["c_a1"]), ("ae2", Q["c_a2"])]:
        gdf[k] = pdate(gdf[c]) if c else pd.NaT
    utm = gdf.estimate_utm_crs(); gdf_plots = gdf.copy()
    prog(0.05, "Grouping small plots")
    gdf, cl_info = build_units(gdf, utm, Q["cl_mode"], Q["cl_col"], Q["cl_micro"], Q["buffer_m"], Q["c_fm"])
    gdf["gate"] = np.where(gdf.core_m2 > 0, "PASSED", "REJECTED")
    a1, a2 = ("sow", "trans") if Q["anchor_mode"] == "Sowing" else ("trans", "sow")
    gdf["anchor"] = gdf[a1].fillna(gdf[a2])
    bnd = gdf.geometry.bounds
    gdf["lon_c"], gdf["lat_c"] = (bnd.minx + bnd.maxx) / 2, (bnd.miny + bnd.maxy) / 2
    out = dict(gdf=gdf, gdf_plots=gdf_plots, cl_info=cl_info, utm=utm, rain_cells={}, cell_of={})
    try: out["ranges"] = ranges_table(gdf_all, Q)
    except Exception as e: out["ranges"] = pd.DataFrame(gdf_all.drop(columns="geometry")); out["ranges_err"] = str(e)
    alld = pd.concat([gdf[k] for k in ["sow", "trans", "pipe_dt", "ae1", "ae2"]]).dropna()
    if alld.empty: out["err"] = "No valid dates found in the mapped date columns."; return out
    tol, cycle = Q["tol"], Q["cycle"]
    w_start = (alld.min() - pd.Timedelta(days=10)).date()
    w_end = (max(gdf.anchor.max() + pd.Timedelta(days=cycle), alld.max()) + pd.Timedelta(days=tol + 6)).date()
    if Q["dr"]: w_start, w_end = Q["dr"]
    w_end = min(w_end, dt.date.today())
    if w_start >= w_end: out["err"] = f"Date range {w_start} → {w_end} is empty (end is capped at today)."; return out
    vg = gdf[gdf.gate == "PASSED"]
    core = vg.geometry.to_crs(utm).buffer(-Q["buffer_m"]).to_crs(4326)
    def _prep(g_):
        if g_ is None or g_.is_empty: return None
        r = set_precision(g_, 1e-6); r = r if (not r.is_empty and r.area > 0) else g_
        return None if (r.is_empty or r.area <= 0) else r
    prepped = [(p, _prep(g_)) for p, g_ in zip(vg.pid, core)]
    out["skipped_empty"] = [p for p, g_ in prepped if g_ is None]
    feats = [{"pid": p, "geom": mapping(g_)} for p, g_ in prepped if g_ is not None]
    prog(0.10, "Sentinel-1")
    s1 = fetch_s1(json.dumps(feats), str(w_start), str(w_end + dt.timedelta(days=1)), Q["orbit"], on_progress=lambda f: prog(0.10 + 0.55 * f, "Sentinel-1"))
    out.update(w_start=w_start, w_end=w_end, n_gate=len(vg), custom_dr=bool(Q["dr"]))
    if s1.empty: out["s1_empty"] = True; return out
    try:
        out["rain_cells"], out["cell_of"] = fetch_chirps_all(json.dumps(list(zip(gdf["pid"], gdf["lon_c"], gdf["lat_c"]))), str(w_start),
                                                             str(w_end + dt.timedelta(days=1)), on_progress=lambda f: prog(0.65 + 0.10 * f, "CHIRPS rain"))
    except Exception as e: out["rain_err"] = str(e)
    s1g = {k: v for k, v in s1.groupby("pid")}
    empty = pd.DataFrame(columns=["pid", "date", "VV", "VH"])
    P = dict(w0=Q["w0"], w1=Q["w1"], delta=Q["delta"], heading=Q["heading"], cycle=cycle, det=Q["det"], tol=tol, lock=Q["lock_on"], rain=Q["rain_thr"])
    det, c_fm, res, rows = Q["det"], Q["c_fm"], {}, []
    step = max(1, len(gdf) // 50)
    for i_, (_, p) in enumerate(gdf.iterrows()):
        if i_ % step == 0: prog(0.75 + 0.18 * i_ / len(gdf), "Auditing plots")
        R = audit_plot(s1g.get(p.pid, empty), p, P, out["rain_cells"].get(out["cell_of"].get(p.pid)))
        if p.gate == "REJECTED": R["status"] = "REJECTED_SMALL"
        res[p.pid] = R
        ha = p.core_m2 / 1e4; base = Q["ef"] * cycle * ha
        av = base * (1 - Q["sf"]) if R["status"] == "VALID" else 0.0
        ev = {e["name"]: e["verdict"] for e in R["ev"]}
        rows.append({"Plot_ID": p.pid, "Farmer": p[c_fm] if c_fm else "", "Transplant": p.trans, "Core_ha": round(ha, 4),
                     "Gate": p.gate, f"dB_wet_{det}": R["wet"].get(det), f"dB_aer_{det}": R["aer"].get(det),
                     "AE1_farmer": p.ae1, "AE1_check": ev.get("Aeration 1", "—"), "AE2_farmer": p.ae2, "AE2_check": ev.get("Aeration 2", "—"),
                     "Lock_Violations": R["lock_hits"], "S1_obs": R["n_obs"], "Dry_days_credited": R["dry_days"], "Dry_events": R["dry_events"],
                     "Members": int(p.n_members), "Status": R["status"], "Baseline_kgCH4": base, "Avoided_kgCH4": av,
                     "Avoided_tCO2e": av * Q["gwp"] / 1000, "Net_tCO2e": av * Q["gwp"] / 1000 * (1 - Q["unc"] / 100)})
    rep = pd.DataFrame(rows); out.update(res=res, rep=rep)
    prog(0.95, "Building exports"); out["exp"] = make_exports(gdf, rep)
    prog(1.0, "Done")
    return out

# ============================================================ BACKGROUND JOB
def start_job(gdf_head, Q, gdf_all):
    """Start run_pipeline in a daemon thread. The UI only reads job['frac'/'text'/'status'] so it never blocks."""
    jid = uuid.uuid4().hex[:8]
    job = dict(id=jid, status="running", frac=0.0, text="Starting", t0=time.time(), err=None, result=None)
    now = time.time()
    for k in list(STORE["jobs"]):                      # purge old finished jobs
        j = STORE["jobs"][k]
        if j["status"] != "running" and now - j["t0"] > 1800: STORE["jobs"].pop(k, None)
    STORE["jobs"][jid] = job
    def prog(f, text=""): job["frac"] = max(0.0, min(1.0, f)); job["text"] = text
    def work():
        try:
            res = run_pipeline(gdf_head, Q, prog, gdf_all)
            res["Q"] = Q; res["uid"] = jid; res["created"] = pd.Timestamp.now(tz="UTC").isoformat()
            job["result"] = res; job["status"] = "done"
        except Exception as e:
            job["err"] = f"{type(e).__name__}: {e}"; job["status"] = "error"
    threading.Thread(target=work, daemon=True, name=f"dmrv-{jid}").start()
    return jid

@st.fragment(run_every=2)
def job_bar(jid):
    """Vertical progress bar on the right edge (top to bottom). Only this fragment reruns, the page stays usable."""
    j = STORE["jobs"].get(jid)
    if not j: return
    if j["status"] != "running":
        st.rerun(); return
    pct = int(j["frac"] * 100)
    st.markdown(f'<div class="vbar"><div class="vfill" style="height:{pct}%"></div></div><div class="vlbl">{j["text"]} · {pct}%</div>', unsafe_allow_html=True)

# ============================================================ SAVED ANALYSES (Hugging Face)
def _hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=hf_token())

def _sign(blob): return hmac.new((AUTH_PASS + "|dmrv-analysis").encode(), blob, hashlib.sha256).hexdigest()

def make_thumb(pipe):
    """Small PNG (base64) drawing the audited polygons coloured by status."""
    from PIL import Image, ImageDraw
    W, H, pad = 360, 220, 14
    im = Image.new("RGB", (W, H), (10, 35, 66)); dr = ImageDraw.Draw(im)
    gdf, rep = pipe["gdf"], pipe["rep"]; stat = dict(zip(rep.Plot_ID, rep.Status))
    colr = {"VALID": "#22C55E", "LOCK_VIOLATION": "#EF4444", "REJECTED_SMALL": "#EF4444", "PARTIAL": "#F59E0B", "NOT_CONFIRMED": "#F59E0B", "NO_AERATION": "#F59E0B"}
    rgb = lambda h: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    minx, miny, maxx, maxy = gdf.total_bounds
    k = math.cos(math.radians((miny + maxy) / 2)); wx, wy = max((maxx - minx) * k, 1e-9), max(maxy - miny, 1e-9)
    s = min((W - 2 * pad) / wx, (H - 2 * pad) / wy); ox, oy = (W - wx * s) / 2, (H - wy * s) / 2
    pt = lambda x, y: (ox + (x - minx) * k * s, H - (oy + (y - miny) * s))
    for pid, geom in zip(gdf.pid, gdf.geometry):
        if geom is None or geom.is_empty: continue
        for pg in (geom.geoms if hasattr(geom, "geoms") else [geom]):
            if pg.geom_type != "Polygon" or pg.is_empty: continue
            dr.polygon([pt(x, y) for x, y, *_ in pg.exterior.coords], fill=rgb(colr.get(stat.get(pid), "#94A3B8")), outline=(255, 255, 255))
    buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def save_analysis(pipe, name, project):
    from huggingface_hub import CommitOperationAdd
    if not hf_token(): raise RuntimeError("HF_TOKEN is missing (add it to st.secrets or the environment).")
    aid = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    blob = gzip.compress(pickle.dumps({k: v for k, v in pipe.items() if k != "sp"}, protocol=4))
    rep, Qs = pipe["rep"], pipe["Q"]
    meta = dict(id=aid, name=name, project=project, created=pd.Timestamp.now(tz="UTC").isoformat(),
                range_from=str(pipe["w_start"]), range_to=str(pipe["w_end"]), n_plots=int(len(pipe["gdf_plots"])), n_units=int(len(rep)),
                n_valid=int((rep.Status == "VALID").sum()), net_tco2e=round(float(rep.Net_tCO2e.sum()), 3), core_ha=round(float(rep.Core_ha.sum()), 2),
                orbit=Qs["orbit"], size_kb=round(len(blob) / 1024), sig=_sign(blob), thumb=make_thumb(pipe))
    ops = [CommitOperationAdd(f"{HF_FOLDER}/{aid}/meta.json", json.dumps(meta).encode()),
           CommitOperationAdd(f"{HF_FOLDER}/{aid}/analysis.pkl.gz", blob)]
    _hf_api().create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops, commit_message=f"Save dMRV analysis {aid}")
    list_saved.clear()
    return aid

@st.cache_data(show_spinner=False, ttl=120)
def list_saved(_v=0):
    from huggingface_hub import hf_hub_download
    from concurrent.futures import ThreadPoolExecutor
    try:
        folders = [e.path for e in _hf_api().list_repo_tree(HF_REPO, path_in_repo=HF_FOLDER, repo_type="dataset") if e.__class__.__name__ == "RepoFolder"]
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower(): return dict(items=[], err="")
        return dict(items=[], err=str(e))
    tok = hf_token()
    def one(p):
        try: return json.load(open(hf_hub_download(HF_REPO, f"{p}/meta.json", repo_type="dataset", token=tok), encoding="utf-8"))
        except Exception: return None
    with ThreadPoolExecutor(max_workers=6) as ex: items = [m for m in ex.map(one, folders) if m]
    items.sort(key=lambda m: m.get("created", ""), reverse=True)
    return dict(items=items, err="")

def load_saved(aid, sig):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(HF_REPO, f"{HF_FOLDER}/{aid}/analysis.pkl.gz", repo_type="dataset", token=hf_token())
    blob = open(p, "rb").read()
    if not hmac.compare_digest(_sign(blob), sig): raise ValueError("Signature mismatch: this file was not saved by this app, so it was not opened.")
    return pickle.loads(gzip.decompress(blob))

def delete_saved(aid):
    _hf_api().delete_folder(path_in_repo=f"{HF_FOLDER}/{aid}", repo_id=HF_REPO, repo_type="dataset", commit_message=f"Delete dMRV analysis {aid}")
    list_saved.clear()

def fmt_created(s):
    try: return pd.Timestamp(s).tz_convert(DISPLAY_TZ).strftime("%d %b %Y, %H:%M")
    except Exception: return str(s)

def saved_workspace():
    if not hf_token():
        st.warning("Saving / opening analyses needs a Hugging Face token with write access to the dataset repo. Add `HF_TOKEN` to `st.secrets` or the environment."); return
    data = list_saved(); items = data["items"]
    if data["err"]: st.error(f"Could not list saved analyses: {data['err']}")
    if not items:
        st.info("No saved analyses yet. Run an analysis, then press 💾 Save analysis."); return
    for i in range(0, len(items), 3):
        for col, it in zip(st.columns(3), items[i:i + 3]):
            with col:
                poly = f'{it.get("n_plots", "?")}' + (f' → {it["n_units"]} units' if it.get("n_units") not in (None, it.get("n_plots")) else "")
                st.markdown(f'<div class="sv"><img src="data:image/png;base64,{it.get("thumb", "")}"><div class="b"><b>{it.get("name", "Analysis")}</b>'
                            f'<div class="m">📅 <em>{it.get("range_from", "?")} → {it.get("range_to", "?")}</em><br>'
                            f'🔷 Polygons: <em>{poly}</em> · {it.get("core_ha", "?")} ha<br>'
                            f'✅ Valid: <em>{it.get("n_valid", "?")}</em> · ☁️ Net <em>{it.get("net_tco2e", "?")}</em> tCO₂e<br>'
                            f'🛰 Orbit {it.get("orbit", "?")} · 🕒 Created <em>{fmt_created(it.get("created"))}</em><br>'
                            f'<span style="color:#9AAEC7">{it.get("project", "")} · {it.get("size_kb", "?")} KB</span></div></div></div>', unsafe_allow_html=True)
                a, b = st.columns([3, 1])
                if a.button("Open analysis", key=f"op_{it['id']}", type="primary", use_container_width=True):
                    try:
                        with st.spinner("Opening saved analysis…"): P_ = load_saved(it["id"], it.get("sig", ""))
                        P_["uid"] = "saved-" + it["id"]
                        st.session_state.update(pipe=P_, pipe_name=it.get("name", ""), ws="run", page="Spatial Audit")
                        st.session_state.pop("sel", None); st.rerun()
                    except Exception as e: st.error(f"Could not open: {e}")
                with b.popover("🗑", use_container_width=True):
                    st.caption("Delete this saved analysis permanently?")
                    if st.button("Yes, delete", key=f"sdel_{it['id']}", type="primary"):
                        try: delete_saved(it["id"]); st.rerun()
                        except Exception as e: st.error(f"Delete failed: {e}")

# ============================================================ SHARED UI HELPERS
def kpi_row(items):
    for col, (ic, bg, v, l) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="ic" style="background:{bg}">{ic}</div><div><div class="v">{v}</div><div class="l">{l}</div></div></div>', unsafe_allow_html=True)

def esri(m):
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri", name="Esri Satellite").add_to(m)
    Fullscreen().add_to(m)

def preview_gdf(g_all, c_id, n_rows):
    """Light version of the plots for browsing while the analysis is still running."""
    pv = g_all.head(int(n_rows)).copy().reset_index(drop=True)
    pv = pv[pv.geometry.notna() & ~pv.geometry.is_empty].reset_index(drop=True)
    pv["pid"] = pv[c_id].astype(str); b = pv.geometry.bounds
    pv["lon_c"], pv["lat_c"] = (b.minx + b.maxx) / 2, (b.miny + b.maxy) / 2
    return pv

def preview_map(pv):
    m = folium.Map([pv.lat_c.mean(), pv.lon_c.mean()], zoom_start=14, tiles=None); esri(m); MousePosition().add_to(m)
    near = pv if len(pv) <= 500 else pv.iloc[:500]
    folium.GeoJson(near[["pid", "geometry"]], name="Plots", style_function=lambda f: {"color": "#22C55E", "weight": 2, "fillOpacity": .12},
                   tooltip=folium.GeoJsonTooltip(["pid"])).add_to(m)
    x0, y0, x1, y1 = pv.total_bounds; m.fit_bounds([[y0, x0], [y1, x1]]); folium.LayerControl().add_to(m)
    components.html(m._repr_html_(), height=470)            # static: no reruns while panning
    if len(pv) > 500: st.caption(f"Map shows the first 500 of {len(pv):,} plots.")

def satellite_page(spg, lat0, lon0):
    st.markdown("#### 📍 Sentinel-1 passes over your plots (ESA acquisition plan)")
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 3], vertical_alignment="bottom")
    days = int(c1.number_input("Show passes for the next … days", 1, 45, AHEAD, key="sp_days",
                               help="Choose how far ahead to look. Passes can only be listed as far ahead as ESA has published its plans."))
    go_run = c2.button("▶ Run passes", type="primary", use_container_width=True, key="sp_run")
    go_ref = c3.button("🔄 Refresh plans", use_container_width=True, key="sp_ref", help="Re-download the ESA KML files and recompute")
    c4.caption(f"Window: now → **{(pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=days)):%d %b %Y}** (UTC)")
    sig = hashlib.md5("|".join(spg.pid.astype(str)).encode()).hexdigest()
    if go_run or go_ref:                                    # computed here only, never during 'Run analysis'
        if go_ref: fetch_plan_kmls.clear()
        with st.spinner("Downloading / reading ESA acquisition plans…"):
            st.session_state["sp"] = dict(sat_plan(spg, days), sig=sig)
    SP = st.session_state.get("sp")
    if SP and SP.get("sig") != sig: SP = None
    if not SP: st.info("Choose the number of days and press ▶ Run passes to download the ESA plan and match it with your plots."); return
    if SP.get("err"): st.error(f"Could not get the ESA acquisition plans: {SP['err']}"); return
    ahead, tbl, L = SP["ahead"], SP["tbl"], SP["lines"]
    if ahead != days: st.caption(f"Showing results for the next {ahead} days. Press ▶ Run passes to update to {days}.")
    has = tbl[tbl.Satellite != ""]
    kpi_row([("📍", "#E3F0FF", f"{has.Plot.nunique()} / {len(spg)}", "Plots with a pass"), ("🛰", "#E8F7E3", f"{len(has)}", f"Pass rows in next {ahead} days"),
             ("⏱", "#FFF1D6", f"{has.Days_ahead.min():.1f} d" if len(has) else "—", "Soonest pass")])
    st.write("")
    sc = spg.assign(d=(spg.lon_c - lon0) ** 2 + (spg.lat_c - lat0) ** 2).nsmallest(300, "d")
    sc = sc.assign(passes=sc.pid.map(SP["tips"]).fillna("No pass"))
    m = folium.Map([lat0, lon0], zoom_start=11, tiles=None); esri(m)
    if not L.empty:
        folium.GeoJson(L[["sat", "utc", "local", "geometry"]], name="All coming pass footprints",
                       style_function=lambda f: {"color": "#FFD60A", "weight": 1.5, "fillOpacity": .05},
                       highlight_function=lambda f: {"weight": 3, "fillOpacity": .2},
                       tooltip=folium.GeoJsonTooltip(["sat", "utc", "local"], aliases=["Satellite", "UTC", "Local"], sticky=True)).add_to(m)
    folium.GeoJson(sc[["pid", "passes", "geometry"]], name="Plots", style_function=lambda f: {"color": "#22C55E", "weight": 2, "fillOpacity": .15},
                   tooltip=folium.GeoJsonTooltip(["pid", "passes"], aliases=["Plot", "Passes"], sticky=True)).add_to(m)
    folium.LayerControl().add_to(m)
    components.html(m._repr_html_(), height=480)            # static map: no reruns while panning / hovering
    st.caption("Hover a footprint for its pass details, or a plot for all its coming passes (local time of the plot's time zone). Plans: " + ", ".join(n.rsplit(".", 1)[0] for n in SP["kmls"]))
    st.download_button(f"⬇ All passes next {ahead} days (CSV)", tbl.to_csv(index=False), f"S1_passes_next_{ahead}_days.csv", "text/csv")

# ============================================================ SIDEBAR 1 (login + upload)
ids_prev = st.session_state.get("ids", [])
if "pending" in st.session_state: st.session_state["sel"] = st.session_state.pop("pending")
if st.session_state.get("sel") not in ids_prev: st.session_state.pop("sel", None)

def _ok(u, p): return hmac.compare_digest(u.strip().lower().encode(), AUTH_USER.lower().encode()) and hmac.compare_digest(p.encode(), AUTH_PASS.encode())
authed = st.session_state.get("auth", False)
upl, paste, project = None, "", "AWD West Bengal"
with st.sidebar:
    st.markdown('<div class="sb-brand">🌾 DREAM <span>dMRV</span></div><div class="sb-h">Access</div>', unsafe_allow_html=True)
    if authed:
        st.markdown('<div class="lg-card"><div class="lg-ic ok">✓</div><div><b>Signed in</b><span>Upload your plots below</span></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="lg-card"><div class="lg-ic">🔐</div><div><b>Unlock Studio</b><span>Verify your contributor credentials first</span></div></div>', unsafe_allow_html=True)
        with st.form("login", border=False):
            em = st.text_input("Contributor Email", placeholder="name@organisation")
            pw = st.text_input("Passcode", type="password", placeholder="••••••••")
            sub = st.form_submit_button("Authenticate", type="primary", use_container_width=True)
        demo = st.button("⚡ Quick Demo Sign-In", use_container_width=True, key="demo")
        if (sub and _ok(em, pw)) or demo: st.session_state["auth"] = True; st.rerun()
        elif sub: st.error("Invalid email or passcode")
    upl = st.file_uploader("Plots file", type=["geojson", "json", "kml", "zip", "gpkg", "csv", "tsv", "txt", "xlsx", "xls"], key="upl", disabled=not authed,
                           help="GeoJSON / KML / SHP.zip / GPKG / CSV-TSV with wkt_geom / XLSX")
    if not authed: st.caption("🔒 Locked until you sign in")
    with st.expander("📋 Paste table instead"):
        paste = st.text_area("Tab-separated, with a wkt_geom column", height=90, key="paste", disabled=not authed)
    if authed:
        st.markdown('<div class="sb-h">Project</div>', unsafe_allow_html=True); project = st.text_input("Project", project)

if not gee_ok: st.error(f"Earth Engine init failed: {gee_err}"); st.stop()

# ============================================================ HEADER (Logout sits level with Project)
logo_html = f'<img src="{logo}">' if logo else "🌾"
right_html = ((f'<span class="chip"><i class="dot on"></i>Active</span><span class="chip">Project: {project}</span>'
               '<a class="chip ok" href="?logout=1" target="_self">⏻ Logout</a>') if authed else
              '<span class="chip"><i class="dot off"></i>Inactive</span><span class="chip ok">● GEE Connected</span>')
st.markdown('<div class="hero"><div class="strip"><span>DREAM · Digital Rice Emission Avoidance Methodology</span><span>Sentinel-1 GRD · CHIRPS V3 · Google Earth Engine</span></div>'
            f'<div class="body">{logo_html}<div><h2>AWD dMRV Platform</h2><small>Farmer records vs Sentinel-1 · buffer gate · dB_wet · flowering lock</small></div>'
            f'<div class="sp"></div>{right_html}</div></div>', unsafe_allow_html=True)

if not authed:                                           # landing is shown ONLY before sign-in
    landing(); st.stop()

# ============================================================ FILE (optional) + SIDEBAR 2 (parameters)
gdf_all, _sig, cols = None, None, []
if upl is not None or paste.strip():
    try: gdf_all = load_cached(upl.name if upl is not None else "", upl.getvalue() if upl is not None else b"", paste)
    except Exception as e: st.error(f"Could not read the plots: {e}")
if gdf_all is not None:
    _sig = (upl.name, upl.size) if upl is not None else ("paste", len(paste))
    cols = [c for c in gdf_all.columns if c != "geometry"]
    if st.session_state.get("dr_sig") != _sig:           # file-based defaults, recomputed only for a new file
        st.session_state["dr_default"] = auto_range(gdf_all, cols); st.session_state["dr_sig"] = _sig
        st.session_state["n_rows_w"] = min(10, len(gdf_all))
    st.session_state["n_rows_w"] = int(min(max(st.session_state.get("n_rows_w", 1), 1), max(len(gdf_all), 1)))

cur = STORE["jobs"].get(st.session_state.get("job"))
job_running = bool(cur and cur["status"] == "running")
run, Qlive = False, None
with st.sidebar:
    if gdf_all is None:
        st.markdown('<div class="sb-h">Parameters</div>', unsafe_allow_html=True)
        st.caption("Upload plots above to set the analysis parameters.")
    else:
        with st.form("params"):
            with st.expander("🗂 Column mapping", expanded=True):
                def pick(label, guess, optional=False):
                    opts = ([None] if optional else []) + cols
                    return st.selectbox(label, opts, index=next((i for i, c in enumerate(opts) if c and guess in c.lower() and "iso" not in c.lower()), 0))
                c_id = pick("Plot ID", "name"); c_fm = pick("Farmer", "farmer_name", True)
                c_sw = pick("Sowing date", "sowing", True); c_tr = pick("Transplanting date", "transplant")
                c_pp = pick("Pipe installation", "pipe", True)
                c_a1 = pick("Aeration 1", "aeration_1", True); c_a2 = pick("Aeration 2", "aeration_2", True)
            with st.expander("📅 Date range", expanded=True):
                dr = st.date_input("From – To (used everywhere: S1, rain, graphs, ranges)", st.session_state["dr_default"], key=f"dr_{abs(hash(_sig))}",
                                   help="Pre-filled from the dates in your file. Change it freely, then press Run analysis.")
            with st.expander("🛰 Sentinel-1 settings"):
                orbit = st.selectbox("Orbit", ["DESCENDING", "ASCENDING", "BOTH"])
                buffer_m = st.number_input("Edge buffer (m)", 0, 30, 10)
                det = st.radio("Detection band (drying)", ["VV", "VH", "VV+VH"], horizontal=True)
                w0, w1 = st.slider("dB_wet search window (days after anchor date)", 0, 90, (0, 30))
                delta = st.number_input("Delta above dB_wet (dB)", 0.5, 8.0, 3.0, 0.5)
                tol = st.number_input("Match tolerance ± days", 1, 15, 6)
            with st.expander("🌧 Rainfall classes"):
                rain_mod = st.number_input("Moderate rain from (mm/day)", 1, 50, 10)
                rain_thr = st.number_input("Heavy rain / gap-fill rain event (mm/day)", 5, 150, 20)
                rain_mod = min(rain_mod, rain_thr)
            with st.expander("🧩 Cluster (small plots)"):
                cl_mode = st.radio("Aggregate into Cluster Polygons", ["Auto", "By column", "Off"], horizontal=True)
                cl_col = st.selectbox("Cluster ID column (used by 'By column')", cols)
                cl_micro = st.toggle("Also pull in touching micro-plots < 0.5 ha (Auto)", True)
            with st.expander("🌱 Agronomy"):
                anchor_mode = st.radio("Anchor date (dB_wet window & flowering lock)", ["Sowing", "Transplanting"], horizontal=True)
                lock_on = st.toggle("Apply flowering-lock rule", False)
                heading = st.number_input("Days to heading", 30, 150, 75)
                cycle = st.number_input("Monitoring window (days after anchor)", 40, 200, 120)
            with st.expander("♻ CH4 factors (editable)"):
                ef = st.number_input("Baseline EF (kg CH4/ha/day)", 0.1, 10.0, 1.3)
                sf = st.number_input("AWD scaling factor", 0.1, 1.0, 0.52)
                gwp = st.number_input("GWP CH4", 20, 40, 28)
                unc = st.number_input("Uncertainty deduction (%)", 0, 50, 15, help="Track 1 default 15%; 20% if ground-truth accuracy < 80%")
            run = st.form_submit_button("▶ Run analysis", type="primary", use_container_width=True, disabled=job_running)

if gdf_all is not None:
    Qlive = dict(c_id=c_id, c_fm=c_fm, c_sw=c_sw, c_tr=c_tr, c_pp=c_pp, c_a1=c_a1, c_a2=c_a2, orbit=orbit, buffer_m=buffer_m, det=det, w0=w0, w1=w1,
                 delta=delta, tol=tol, rain_mod=rain_mod, rain_thr=rain_thr, cl_mode=cl_mode, cl_col=cl_col if cl_mode == "By column" else None,
                 cl_micro=bool(cl_micro) and cl_mode == "Auto", anchor_mode=anchor_mode, lock_on=lock_on, heading=heading, cycle=cycle,
                 ef=ef, sf=sf, gwp=gwp, unc=unc, n_rows=int(st.session_state["n_rows_w"]),
                 dr=(dr[0], dr[1]) if isinstance(dr, (tuple, list)) and len(dr) == 2 else None)
    if run and not job_running:                          # start in the background, return immediately
        gh = gdf_all.head(Qlive["n_rows"]).reset_index(drop=True)
        st.session_state["job"] = start_job(gh, Qlive, gdf_all)
        cur = STORE["jobs"][st.session_state["job"]]; job_running = True

# adopt a finished background job (once)
if cur and cur["status"] == "done" and st.session_state.get("adopted") != cur["id"]:
    st.session_state["pipe"] = cur["result"]; st.session_state["adopted"] = cur["id"]; st.session_state["pipe_name"] = ""
    st.session_state.pop("sel", None)

# progress (right-edge vertical bar) / errors
if cur:
    if cur["status"] == "running": job_bar(cur["id"])
    elif cur["status"] == "error" and st.session_state.get("job_seen") != cur["id"]:
        st.error(f"Analysis stopped: {cur['err']}"); st.session_state["job_seen"] = cur["id"]

PIPE = st.session_state.get("pipe")
HAS = bool(PIPE) and "rep" in PIPE

# ============================================================ ONE HORIZONTAL BAR:  [6 tabs] ... [◀ ▶] [💾 Save analysis]
st.session_state.setdefault("ws", "run"); st.session_state.setdefault("page", "Spatial Audit")
ws = st.session_state["ws"]
def _ws(w): st.session_state["ws"] = w
def _tab(k): st.session_state.update(page=k, ws="run")
with st.container(key="nav"):
    bc = st.columns([1.35] * len(NAV) + [0.8, 1.25], vertical_alignment="center")
    for c_, (lab, key_) in zip(bc, NAV):
        c_.button(lab, key=f"nav_{key_}", type="primary" if (ws == "run" and st.session_state["page"] == key_) else "secondary", use_container_width=True,
                  disabled=(key_ == "Paddy"), help="Locked for now" if key_ == "Paddy" else None, on_click=_tab, args=(key_,))
    with bc[len(NAV)]:
        with st.container(key="arrows"):
            a_l, a_r = st.columns(2)
            a_l.button("◀", key="ws_l", on_click=_ws, args=("run",), disabled=ws == "run", use_container_width=True, help="Back to the analysis")
            a_r.button("▶", key="ws_r", on_click=_ws, args=("saved",), disabled=ws == "saved", use_container_width=True, help="Open saved analyses")
    with bc[len(NAV) + 1]:
        with st.container(key="savebox"):
            if HAS:
                with st.popover("💾 Save analysis", use_container_width=True):
                    uid = PIPE.get("uid", "x")
                    nm = st.text_input("Name", value=st.session_state.get("pipe_name") or f"{project} · {PIPE['w_start']} → {PIPE['w_end']}", key=f"save_name_{uid}")
                    if not hf_token(): st.warning("Add HF_TOKEN (write access) to st.secrets / environment first.")
                    if st.button("Save to Hugging Face", type="primary", key=f"save_go_{uid}", use_container_width=True, disabled=not hf_token()):
                        try:
                            with st.spinner("Uploading to Hugging Face…"): aid = save_analysis(PIPE, nm.strip() or "Analysis", project)
                            st.success(f"Saved ✓  {HF_FOLDER}/{aid}")
                        except Exception as e: st.error(f"Save failed: {e}")
            else:
                st.button("💾 Save analysis", key="save_off", disabled=True, use_container_width=True, help="Run an analysis first")
st.write("")

if ws == "saved":
    saved_workspace(); st.stop()

page = st.session_state["page"]
if not HAS and gdf_all is None: st.stop()

# ---- Inspect plot  +  Plots to audit (number of polygons), side by side as before
_, sel_slot, nrow_slot = st.columns([2.4, 1.2, 1.4])
n_rows = Qlive["n_rows"] if Qlive else 0
if gdf_all is not None:
    n_rows = int(nrow_slot.number_input(f"Plots to audit (file has {len(gdf_all)})", 1, max(len(gdf_all), 1), key="n_rows_w",
                                        help="Applies the next time you press ▶ Run analysis."))

# ---- no finished results yet: you can still browse the plots and use Satellite Path
if not HAS:
    if PIPE and PIPE.get("err"): st.error(PIPE["err"])
    if PIPE and PIPE.get("s1_empty"):
        Q0 = PIPE["Q"]
        st.warning(f"No Sentinel-1 data came back. Plots passing the −{Q0['buffer_m']} m gate: {PIPE['n_gate']}. Window: {PIPE['w_start']} → {PIPE['w_end']}, orbit: {Q0['orbit']}. "
                   "Try Orbit = BOTH, a smaller edge buffer, or a wider date range.")
    pv = preview_gdf(gdf_all, Qlive["c_id"], n_rows)
    if page == "Satellite Path": satellite_page(pv, pv.lat_c.mean(), pv.lon_c.mean())
    elif page == "Spatial Audit": preview_map(pv)
    else: st.info("This tab fills in when the analysis finishes." if job_running else "Press ▶ Run analysis in the side panel first.")
    st.stop()

# ---- results available: every setting comes from the run that produced them
Q = PIPE["Q"]
orbit, buffer_m, det, w0, w1, delta, tol = Q["orbit"], Q["buffer_m"], Q["det"], Q["w0"], Q["w1"], Q["delta"], Q["tol"]
rain_thr, rain_mod, cl_mode, anchor_mode, lock_on = Q["rain_thr"], Q.get("rain_mod", min(10, Q["rain_thr"])), Q["cl_mode"], Q["anchor_mode"], Q["lock_on"]
heading, cycle, ef, sf, gwp, unc = Q["heading"], Q["cycle"], Q["ef"], Q["sf"], Q["gwp"], Q["unc"]
gdf, gdf_plots, cl_info, utm = PIPE["gdf"], PIPE["gdf_plots"], PIPE["cl_info"], PIPE["utm"]
rain_cells, cell_of = PIPE["rain_cells"], PIPE["cell_of"]
def rain_for(pid): return rain_cells.get(cell_of.get(pid), pd.DataFrame(columns=["date", "rain"]))

w_start, w_end = PIPE["w_start"], PIPE["w_end"]
XR = [str(w_start), str(w_end + dt.timedelta(days=1))]            # the chosen range is applied to every graph
st.caption(f"📅 Date range in use: **{w_start} → {w_end}**" + (f"  ·  📂 Showing saved analysis: **{st.session_state['pipe_name']}**" if st.session_state.get("pipe_name") else ""))
if PIPE.get("rain_err"): st.caption(f"CHIRPS unavailable: {PIPE['rain_err']}")
res, rep = PIPE["res"], PIPE["rep"]
st.session_state["ids"] = rep.Plot_ID.tolist()
with sel_slot: sel = st.selectbox("Inspect plot", rep.Plot_ID.tolist(), key="sel")
r = rep[rep.Plot_ID == sel].iloc[0]; g = gdf[gdf.pid == sel].iloc[0]; R = res[sel]; ts = R["ts"]

# ============================================================ KPIs / CHART HELPERS
n_valid = int((rep.Status == "VALID").sum()); tot_base = rep.Baseline_kgCH4.sum()
pct = 100 * rep.Avoided_kgCH4.sum() / tot_base if tot_base else 0
evs = [e for x in res.values() for e in x["ev"]]
conf = 100 * sum(e["verdict"] == "CONFIRMED" for e in evs) / len(evs) if evs else 0
cards = [("🌿", "#E3F7EC", f"{pct:.1f}%", "CH4 avoided vs baseline"), ("☁️", "#E3F0FF", f"{rep.Net_tCO2e.sum():,.2f}", f"tCO2e avoided (net of {unc}%)"),
         ("✅", "#E8F7E3", f"{n_valid} / {len(rep)}", "Plots valid"), ("🛰", "#FFF1D6", f"{conf:.0f}%", "Farmer aerations confirmed by S1"),
         ("📐", "#EDE7FF", f"{rep.Core_ha.sum():,.2f} ha", "Core area audited")]

COL = {"VV": "#2F80ED", "VH": "#0FB5A6", "VV+VH": "#C026D3"}
BANDS = {"VV": ["VV"], "VH": ["VH"], "VV & VH lines": ["VV", "VH"], "VV+VH combined": ["VV+VH"]}
def marks_on(fig):
    for i, (lab, d, c) in enumerate([("🌾 Sowing", g.sow, "#8B5E34"), ("🌱 Transplant", g.trans, "#16A34A"), ("🔧 Pipe", g.pipe_dt, "#64748B"),
                                     ("💧 AE1 (farmer)", g.ae1, "#F97316"), ("💧 AE2 (farmer)", g.ae2, "#F97316")]):
        if pd.isna(d): continue
        fig.add_shape(type="line", x0=d, x1=d, y0=0, y1=1, xref="x", yref="paper", line=dict(color=c, width=2, dash="dot"))
        fig.add_annotation(x=d, y=1.01 + 0.085 * (i % 2), yref="paper", yanchor="bottom", text=f"<b>{lab}</b><br>{d:%d %b}", showarrow=False,
                           font=dict(size=10, color=c), bgcolor="#fff", bordercolor=c, borderwidth=1, borderpad=2)

def chart(show, rain_on=True, audited_on=True, lock_vis=True):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.2, 0.8])
    rf = rain_for(sel) if rain_on else pd.DataFrame(); rain_idx, heavy_shapes = {"lo": [], "hi": []}, []
    if not rf.empty:
        for nm, part, colr, grp in [(f"Rain < {rain_mod} mm", rf[rf.rain < rain_mod], "#9CC9F5", "lo"),
                                    (f"Rain {rain_mod}–{rain_thr} mm", rf[(rf.rain >= rain_mod) & (rf.rain < rain_thr)], "#3B8CF5", "lo"),
                                    (f"Rain ≥ {rain_thr} mm (heavy)", rf[rf.rain >= rain_thr], "#E5484D", "hi")]:
            if not part.empty:
                fig.add_bar(x=part.date, y=part.rain, name=nm, marker_color=colr, row=1, col=1, hovertemplate="%{y:.1f} mm"); rain_idx[grp].append(len(fig.data) - 1)
        fig.add_hline(y=rain_mod, row=1, col=1, line=dict(color="#3B8CF5", dash="dot", width=1), annotation_text=f"{rain_mod} mm", annotation_position="top left", annotation_font_size=9)
        fig.add_hline(y=rain_thr, row=1, col=1, line=dict(color="#E5484D", dash="dot", width=1), annotation_text=f"{rain_thr} mm", annotation_position="top left", annotation_font_size=9)
        heavy = rf[rf.rain >= rain_thr]
        if not heavy.empty:
            fig.add_scatter(x=heavy.date, y=heavy.rain, mode="markers+text", showlegend=False, row=1, col=1, text=[f"⛈ {v:.0f} mm" for v in heavy.rain],
                            textposition="bottom center", textfont=dict(size=10, color="#C53030"), marker=dict(symbol="triangle-down", size=9, color="#E5484D"),
                            hovertemplate="%{y:.1f} mm heavy rain")
            rain_idx["hi"].append(len(fig.data) - 1)
            for d in heavy.date:
                heavy_shapes.append(len(fig.layout.shapes))
                fig.add_shape(type="rect", x0=d - pd.Timedelta(hours=12), x1=d + pd.Timedelta(hours=12), y0=0, y1=1, xref="x", yref="paper", fillcolor="#E5484D", opacity=.10, line_width=0, layer="below")
    for b in BANDS[show]:
        if ts.empty: break
        fig.add_scatter(x=ts.date, y=ts[b], name=f"{b} (dB)", mode="lines+markers", line=dict(color=COL[b], width=2.4), marker=dict(size=6), row=2, col=1, hovertemplate="%{y:.2f} dB")
        fig.add_hline(y=R["wet"][b], row=2, col=1, line=dict(color=COL[b], dash="dot", width=1.2), annotation_text=f"dB_wet {b}", annotation_position="bottom right", annotation_font_size=10)
        fig.add_hline(y=R["aer"][b], row=2, col=1, line=dict(color="#E5484D", dash="dash", width=1.2), annotation_text=f"Drying trigger {b}", annotation_position="top right", annotation_font_size=10)
        fig.add_scatter(x=[R["wet_date"][b]], y=[R["wet"][b]], mode="markers", name=f"Min {b} (dB_wet)", row=2, col=1, marker=dict(symbol="star", size=15, color="#F5A623", line=dict(color="#fff", width=1.5)))
    if not ts.empty:
        a = ts[ts.aerated]
        fig.add_scatter(x=a.date, y=a[det], mode="markers", name=f"S1 drying detected ({det})", row=2, col=1, marker=dict(symbol="diamond", size=11, color="#E5484D", line=dict(color="#fff", width=1.2)))
        if audited_on: fig.add_scatter(x=ts.date, y=ts.audited, name=f"Audited {det} (lock applied)", mode="lines", row=2, col=1, line=dict(color="#1F2937", width=1.6, dash="dash"))
        for e in R["ev"]:
            if e["obs"] is not None and e["verdict"] == "CONFIRMED" and e.get("hit"):
                fig.add_scatter(x=[e["obs"]], y=[e["val"]], mode="markers", showlegend=False, row=2, col=1, marker=dict(symbol="circle-open", size=26, color="#16A34A", line=dict(width=3)), hovertemplate=f"{e['name']} confirmed by S1")
    if not ts.empty and pd.notna(g.anchor):
        b0, b1 = g.anchor + pd.Timedelta(days=w0), g.anchor + pd.Timedelta(days=w1)
        fig.add_shape(type="rect", x0=b0, x1=b1, y0=0, y1=1, xref="x", yref="paper", fillcolor="#2F80ED", opacity=.13, line=dict(color="#2F80ED", width=1, dash="dot"), layer="below")
        fig.add_annotation(x=b0 + (b1 - b0) / 2, y=0.76, yref="paper", yanchor="top", showarrow=False, text=f"💧 Flooded-baseline window (day {w0}–{w1}) → dB_wet", font=dict(size=10, color="#1D5FD1"))
    for a_, b_, _n in R["runs"]:
        fig.add_shape(type="rect", x0=a_ - pd.Timedelta(hours=12), x1=b_ + pd.Timedelta(hours=12), y0=0, y1=0.77, xref="x", yref="paper", fillcolor="#16A34A", opacity=.13, line_width=0, layer="below")
    if R["runs"]: fig.add_scatter(x=[None], y=[None], mode="markers", name="Credited dry days (≥ 3 d)", row=2, col=1, marker=dict(symbol="square", size=11, color="rgba(22,163,74,.35)"))
    if lock_vis and pd.notna(R["ls"]):
        fig.add_shape(type="rect", x0=R["ls"], x1=R["le"], y0=0, y1=1, xref="x", yref="paper", fillcolor="#FFC857", opacity=.22, line_width=0, layer="below")
        fig.add_annotation(x=R["ls"] + (R["le"] - R["ls"]) / 2, y=0.0, yref="paper", text="🌸 Flowering lock", showarrow=False, font=dict(size=10, color="#B7791F"), yanchor="bottom")
    marks_on(fig)
    ms = dict(type="buttons", direction="right", y=1.22, yanchor="bottom", bgcolor="#fff", bordercolor="#C9D8EE", font=dict(size=11, color="#0A2342"), pad=dict(r=3, t=2, b=2))
    anchor = g.sow if pd.notna(g.sow) else (g.trans if pd.notna(g.trans) else (ts.date.min() if not ts.empty else pd.NaT))
    zb = []
    if pd.notna(anchor):
        for m in (1, 2, 3):
            lo, hi = str(anchor.date()), str((anchor + pd.Timedelta(days=30 * m)).date())
            zb.append(dict(label=f"{m}m", method="relayout", args=[{"xaxis.range": [lo, hi], "xaxis2.range": [lo, hi]}]))
    zb.append(dict(label="All", method="relayout", args=[{"xaxis.range": XR, "xaxis2.range": XR}]))
    menus = [dict(ms, x=0, xanchor="left", buttons=zb, active=len(zb) - 1)]
    if rain_idx["lo"] or rain_idx["hi"]:
        n = len(fig.data)
        def vis(lo_on, hi_on):
            v = [True] * n
            for i in rain_idx["lo"]: v[i] = lo_on
            for i in rain_idx["hi"]: v[i] = hi_on
            return [{"visible": v}, {f"shapes[{k}].visible": hi_on for k in heavy_shapes}]
        menus.append(dict(ms, x=1, xanchor="right", active=0, buttons=[dict(label="All rain", method="update", args=vis(True, True)),
                          dict(label=f"< {rain_thr} mm", method="update", args=vis(True, False)), dict(label=f"≥ {rain_thr} mm", method="update", args=vis(False, True))]))
    fig.add_annotation(x=0, y=1.29, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False, text="📅 Zoom from sowing date", font=dict(size=10, color="#6B84A3"))
    fig.add_annotation(x=1, y=1.29, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False, text="🌧 Rainfall filter", font=dict(size=10, color="#6B84A3"))
    fig.update_layout(template="plotly_white", height=720, hovermode="x unified", paper_bgcolor="#fff", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=150, b=150),
                      legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, font_size=11),
                      xaxis2=dict(rangeslider=dict(visible=True, thickness=0.05, range=XR)), updatemenus=menus)
    fig.update_xaxes(range=XR)
    rmax = float(rf.rain.max()) if not rf.empty else 0.0
    fig.update_yaxes(title_text="Rain (mm)", range=[max(rmax, rain_thr) * 1.5, 0], row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="Backscatter (dB)", row=2, col=1, gridcolor="#EEF3FA")
    return fig

def badge(v):
    m = {"VALID": "b-ok", "CONFIRMED": "b-ok", "PARTIAL": "b-warn", "NOT_CONFIRMED": "b-warn", "DRY_NOT_CREDITED": "b-warn", "NO_PASS": "b-gray",
         "LOCK_VIOLATION": "b-bad", "NO_AERATION": "b-warn", "NO_DATA": "b-gray", "REJECTED_SMALL": "b-bad"}
    return f'<span class="badge {m.get(v, "b-gray")}">{v.replace("_", " ")}</span>'

def event_cards():
    if not R["ev"] and not R["extra"]: st.caption("No farmer aeration dates mapped for this plot."); return
    cs = st.columns(max(len(R["ev"]) + (1 if R["extra"] else 0), 1))
    for c, e in zip(cs, R["ev"]):
        colr = {"CONFIRMED": "#16A34A", "NOT_CONFIRMED": "#F5A623", "DRY_NOT_CREDITED": "#F5A623", "NO_PASS": "#94A3B8"}[e["verdict"]]
        if e["verdict"] == "CONFIRMED" and not e.get("hit"): msg = "Dry days credited around this date from the gap-filling rule (A 2.3.6), at least 3 consecutive days"
        elif e["verdict"] == "DRY_NOT_CREDITED": msg = f"S1 saw a dry pass on {e['obs']:%d %b}, but it did not reach 3 consecutive creditable days (single pass, rain gap, or lock)"
        elif e["verdict"] == "CONFIRMED": msg = f"S1 saw drying on <b>{e['obs']:%d %b}</b> ({abs((e['obs'] - e['date']).days)} d from report) · {det} {e['val']:.1f} dB ≥ trigger"
        elif e["verdict"] == "NOT_CONFIRMED": msg = f"Nearest S1 pass {e['obs']:%d %b}: {det} {e['val']:.1f} dB stayed below the {R['aer'][det]:.1f} dB trigger → field still wet"
        else: msg = f"No Sentinel-1 pass within ±{tol} days"
        if pd.notna(R["ls"]) and R["ls"] <= e["date"] <= R["le"]: msg += "<br>⚠️ <b>Reported inside the flowering lock window</b> (field must stay flooded)"
        c.markdown(f'<div class="ev" style="--c:{colr}"><b>{e["name"]}: farmer reported {e["date"]:%d %b %Y}</b> {badge(e["verdict"])}<div class="m">{msg}</div></div>', unsafe_allow_html=True)
    if R["extra"]:
        cs[-1].markdown('<div class="ev" style="--c:#E5484D"><b>Satellite-only drying</b><div class="m">S1 detected drying on ' + ", ".join(f"{d:%d %b}" for d in R["extra"]) + " with no matching farmer record.</div></div>", unsafe_allow_html=True)

def controls(key):
    a, b, c, d = st.columns([2.6, 1, 1, 1])
    return (a.radio("Show band", list(BANDS), horizontal=True, key=f"b{key}"), b.toggle("CHIRPS rain", True, key=f"r{key}"),
            c.toggle("Audited signal", False, key=f"a{key}"), d.toggle("Lock window", False, key=f"l{key}"))

def method_panel():
    n_pass = int((rep.Gate == "PASSED").sum())
    items = [("A 2.3.2", "GRD pre-processing", f"Sentinel-1 GRD · IW · VV+VH · orbit {orbit} · multi-temporal ratio + 3×3 boxcar speckle filter"),
             ("A 2.3.3", "Core-zone buffer gate", f"−{buffer_m} m inner buffer · {n_pass}/{len(rep)} plots pass"),
             ("A 2.3.4", "Flooded baseline (dB_wet)", f"minimum {det} in days {w0}–{w1} after the {anchor_mode.lower()} date"),
             ("A 2.3.5", "Aeration trigger", f"dB_wet + {delta} dB · matched to farmer dates ±{tol} d"),
             ("A 2.4", "Flowering lock", (f"{anchor_mode.lower()} + {heading} d ± 10 d · always flooded" if lock_on else "rule switched off")),
             ("A 2.3.6–7", "Gap-filling & 3-day rule", f"wet→dry from midpoint · dry→dry unless ≥{rain_thr} mm rain · credit only ≥3 consecutive dry days"),
             ("A 2.3.3b", "Cluster polygons", f"mode {cl_mode} · {len(cl_info)} cluster(s) formed"),
             ("Rain", "CHIRPS V3 daily", f"plot-centroid sample · classes <{rain_mod} / {rain_mod}–{rain_thr} / ≥{rain_thr} mm/day")]
    st.markdown('<div class="card"><h4>📘 Methodology &amp; compliance</h4>' + "".join(
        f'<div class="row"><span>{a} · {b}</span><span style="text-align:right;max-width:70%">{c}</span></div>' for a, b, c in items) + '</div>', unsafe_allow_html=True)
    params = dict(buffer_m=buffer_m, orbit=orbit, detection_band=det, dB_wet_window_days=[w0, w1], delta_dB=delta, tolerance_days=tol, days_to_heading=heading,
                  anchor=anchor_mode, cluster_mode=cl_mode, uncertainty_pct=unc, flowering_lock=lock_on, monitoring_days=cycle, moderate_rain_mm=rain_mod,
                  heavy_rain_mm=rain_thr, ef=ef, scaling_factor=sf, gwp=gwp, window=[str(w_start), str(w_end)], plots_audited=len(rep))
    st.download_button("⬇ Methodology parameters (JSON)", json.dumps(params, indent=2), "DREAM_parameters.json")

# ============================================================ PAGES
if page == "Spatial Audit":
    if not cl_info.empty:                                   # cluster / gate notices live in this tab only
        with st.expander(f"🧩 {len(cl_info)} Cluster Polygon(s) formed from small plots", expanded=False):
            st.dataframe(cl_info, use_container_width=True)
            if (cl_info.Max_date_spread_days > tol).any():
                st.warning("Some clusters have members whose dates differ by more than the match tolerance; the cluster uses the median date.")
    _rej = gdf[gdf.gate == "REJECTED"]
    if len(_rej): st.warning(f"{len(_rej)} unit(s) have no core pixel after the −{buffer_m} m buffer and no neighbour to cluster with: " + ", ".join(_rej.pid.head(8)))
    left, right = st.columns([1.7, 1])
    with left:
        c = g.geometry.centroid
        m = folium.Map([c.y, c.x], zoom_start=17, tiles=None); esri(m); MousePosition().add_to(m)
        sc = {"VALID": "#22C55E", "LOCK_VIOLATION": "#EF4444", "PARTIAL": "#F59E0B", "NOT_CONFIRMED": "#F59E0B"}
        near = gdf
        if len(gdf) > 300:
            near = gdf.iloc[np.argsort(((gdf["lon_c"] - g["lon_c"]) ** 2 + (gdf["lat_c"] - g["lat_c"]) ** 2).to_numpy())[:300]]
            st.caption(f"Map shows the 300 plots nearest to the selected one (of {len(gdf):,}).")
        gj = near[["pid", "geometry"]].merge(rep[["Plot_ID", "Status"]], left_on="pid", right_on="Plot_ID")
        folium.GeoJson(gj, name="Plots (status colour)", style_function=lambda f: {"color": sc.get(f["properties"]["Status"], "#94A3B8"), "weight": 2, "fillOpacity": .08},
                       tooltip=folium.GeoJsonTooltip(["pid", "Status"])).add_to(m)
        folium.GeoJson(g.geometry.__geo_interface__, name="Selected raw boundary", style_function=lambda f: {"color": "red", "fillOpacity": 0, "weight": 3}).add_to(m)
        if g.n_members > 1:
            folium.GeoJson(gdf_plots[gdf_plots.pid.isin(str(g.members).split(", "))][["pid", "geometry"]], name="Cluster member plots",
                           style_function=lambda f: {"color": "#F59E0B", "weight": 2, "fillOpacity": .15}, tooltip=folium.GeoJsonTooltip(["pid"])).add_to(m)
        cb = gpd.GeoSeries(g.geometry, crs=4326).to_crs(utm).buffer(-buffer_m).to_crs(4326)
        if not cb.iloc[0].is_empty:
            folium.GeoJson(cb.__geo_interface__, name=f"Buffered core (-{buffer_m} m)", style_function=lambda f: {"color": "cyan", "weight": 3, "fillOpacity": .15}).add_to(m)
        folium.LayerControl().add_to(m)
        out = st_folium(m, height=420, use_container_width=True, returned_objects=["last_active_drawing"])
        pid_ = (((out or {}).get("last_active_drawing") or {}).get("properties") or {}).get("pid")
        if pid_ and pid_ != sel and pid_ != st.session_state.get("last_click"):
            st.session_state["last_click"] = pid_; st.session_state["pending"] = pid_; st.rerun()
    with right:
        f = lambda d: d.strftime("%d %b %Y") if pd.notna(d) else "—"
        nm = lambda x: f"{x:.2f} dB" if pd.notna(x) else "—"
        st.markdown('<div class="card" style="min-height:455px"><h4>🔍 Plot Audit Inspector</h4>' +
            "".join(f'<div class="row"><span>{a}</span><span>{b}</span></div>' for a, b in [
            ("Plot ID", sel), ("Farmer", r.Farmer), ("Gate", f"{r.Gate} (−{buffer_m} m)"), ("Core area", f"{r.Core_ha} ha"),
            ("Sowing", f(g.sow)), ("Transplant", f(g.trans)), ("Pipe installed", f(g.pipe_dt)),
            (f"dB_wet ({det})", nm(R['wet'].get(det))), ("Min reached on", f(R['wet_date'].get(det))), (f"Drying trigger (+{delta} dB)", nm(R['aer'].get(det))),
            ("Flowering lock", f"{R['ls']:%d %b} – {R['le']:%d %b}" if pd.notna(R['ls']) else "—"),
            ("S1 observations", R["n_obs"]), ("Lock violations", R["lock_hits"]), ("Credited dry days", f"{R['dry_days']} in {R['dry_events']} event(s)"),
            ("Anchor date", f(g.anchor)), ("Unit", "Cluster of " + str(int(g.n_members)) + " plots" if g.n_members > 1 else "Single plot")]) +
            f'<div style="margin-top:10px">Status: {badge(R["status"])}</div></div>', unsafe_allow_html=True)
    st.markdown("#### 📈 Farmer actions vs Sentinel-1 · rainfall on top, backscatter below")
    show, rain_on, aud_on, lock_vis = controls("main")
    st.plotly_chart(chart(show, rain_on, aud_on, lock_vis), use_container_width=True)
    st.markdown("#### 🧾 Verification: did the satellite see what the farmer reported?")
    event_cards(); st.write(""); method_panel()

elif page == "Export":
    kpi_row(cards); st.write("")
    a, b = st.columns([2, 1])
    fig = go.Figure()
    fig.add_bar(x=rep.Plot_ID, y=rep.Avoided_kgCH4, name="CH4 avoided (kg)", marker_color="#16A34A")
    fig.add_bar(x=rep.Plot_ID, y=rep.Baseline_kgCH4 - rep.Avoided_kgCH4, name="Remaining emission (kg)", marker_color="#9CC9F5")
    fig.update_layout(template="plotly_white", barmode="stack", height=380, title="Baseline CH4 per plot: avoided vs remaining (kg)", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    a.plotly_chart(fig, use_container_width=True)
    cnt = rep.Status.value_counts()
    pie = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=.55)); pie.update_layout(template="plotly_white", height=380, title="Plot status", margin=dict(l=10, r=10, t=40, b=10))
    b.plotly_chart(pie, use_container_width=True)
    net = go.Figure(go.Bar(x=rep.Plot_ID, y=rep.Net_tCO2e, marker_color="#2F80ED"))
    net.update_layout(template="plotly_white", height=300, title=f"Net tCO2e per plot (after {unc}% deduction)", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(net, use_container_width=True)
    st.caption("Estimate = EF × window days × core ha × (1 − scaling factor), VALID plots only. Replace factors with your approved methodology values.")
    st.dataframe(rep, use_container_width=True)
    X = PIPE["exp"]
    st.markdown("##### ⬇ Export audit results")
    for col, (lab, k, fn, mime) in zip(st.columns(4), [("CSV", "csv", "DREAM_audit.csv", "text/csv"), ("KML", "kml", "DREAM_audit.kml", "application/vnd.google-earth.kml+xml"),
                                                        ("GeoJSON", "geojson", "DREAM_audit.geojson", "application/geo+json"), ("Shapefile (zip)", "shp", "DREAM_audit_shp.zip", "application/zip")]):
        col.download_button(f"⬇ {lab}", X[k] or b"", fn, mime, use_container_width=True, disabled=X[k] is None, key=f"dl_{k}")
    st.markdown("##### 📏 Extract Ranges")
    RG = PIPE["ranges"]
    st.caption(f"All {len(RG)} uploaded plots: original attributes + Range_from / Range_to / Range_days (clipped to the side-panel date range) + "
               "`range` = boundary as a JSON-style list of {\"lat\": …, \"lng\": …} dictionaries. CSV only.")
    if PIPE.get("ranges_err"): st.warning(f"Ranges could not be computed ({PIPE['ranges_err']}); attributes only.")
    st.download_button("⬇ Extract Ranges CSV", RG.to_csv(index=False), "DREAM_extract_ranges.csv", "text/csv")

elif page == "Multispectral Analysis":
    INFO = {"NDVI": ("Greenness / crop vigour. The peak is roughly the heading stage.", "#16A34A"), "NDMI": ("Canopy moisture (NIR vs SWIR). Falls when the crop or soil dries.", "#0FB5A6"),
            "NDWI": ("Open water (green vs NIR). Above 0 usually means standing water.", "#2F80ED"), "MNDWI": ("Water and wet soil (green vs SWIR). Good in the early, open-canopy stage.", "#7C3AED"),
            "EVI": ("Greenness that does not saturate in dense canopy.", "#F59E0B"), "SAVI": ("Greenness with soil-background correction for sparse early canopy.", "#8B5E34")}
    st.markdown("#### 🛰️ Sentinel-2 multispectral analysis")
    with st.form("s2form"):
        c1, c2, c3, c4 = st.columns([2.4, 1, 1, 1])
        idxs = c1.multiselect("Indices", list(INFO), default=["NDVI", "NDMI", "NDWI"])
        cloud = c2.slider("Max scene cloud %", 10, 100, 60, 5)
        minclear = c3.slider("Min clear share in plot", 0.1, 1.0, 0.5, 0.1)
        c4.write(""); s2_run = c4.form_submit_button("▶ Run Sentinel-2", type="primary", use_container_width=True)
    if s2_run: st.session_state["s2"] = dict(idxs=idxs, cloud=cloud, minclear=minclear)
    req = st.session_state.get("s2")
    if not req: st.info("Pick the indices and press ▶ Run Sentinel-2. It analyses the plot chosen under 'Inspect plot'.")
    else:
        cg = gpd.GeoSeries([g.geometry], crs=4326).to_crs(utm).buffer(-buffer_m).to_crs(4326).iloc[0]
        with st.spinner("Querying Sentinel-2 on Earth Engine…"):
            d2 = fetch_s2(json.dumps(mapping(g.geometry if cg.is_empty else cg)), str(w_start), str(w_end + dt.timedelta(days=1)), req["cloud"])
        d2 = d2[d2.clear >= req["minclear"]] if not d2.empty else d2
        if d2.empty: st.warning("No usable Sentinel-2 scenes for this plot and window. Raise the cloud limit or lower the clear-share limit (monsoon clouds are common).")
        else:
            use = [k for k in req["idxs"] if k in d2.columns]; pk = d2.loc[d2.NDVI.idxmax()]
            kpi_row([("🌱", "#E3F7EC", f"{pk.NDVI:.2f}", f"NDVI peak · {pk.date:%d %b}"),
                     ("📅", "#E3F0FF", (f"{(pk.date - g.anchor).days} d" if pd.notna(g.anchor) else "—"), "Peak NDVI, days after anchor"),
                     ("💧", "#E8F7E3", f"{int((d2.MNDWI > 0).sum())} of {len(d2)}", "Scenes with MNDWI > 0 (water)"), ("🛰", "#FFF1D6", f"{len(d2)}", "Clear Sentinel-2 scenes used")])
            st.write("")
            f2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.2, 0.8]); rf2 = rain_for(sel)
            if not rf2.empty:
                f2.add_bar(x=rf2.date, y=rf2.rain, name="CHIRPS rain (mm/d)", row=1, col=1, hovertemplate="%{y:.1f} mm", marker_color=["#E5484D" if v >= rain_thr else "#7DB8F5" for v in rf2.rain])
            for k in use: f2.add_scatter(x=d2.date, y=d2[k], name=k, mode="lines+markers", line=dict(color=INFO[k][1], width=2.2), marker=dict(size=7), row=2, col=1, hovertemplate="%{y:.3f}")
            if "NDVI" in use: f2.add_scatter(x=[pk.date], y=[pk.NDVI], mode="markers", name="NDVI peak", row=2, col=1, marker=dict(symbol="star", size=15, color="#F5A623", line=dict(color="#fff", width=1.5)))
            f2.add_hline(y=0, row=2, col=1, line=dict(color="#94A3B8", dash="dot", width=1)); marks_on(f2)
            rmax = float(rf2.rain.max()) if not rf2.empty else 0.0
            f2.update_layout(template="plotly_white", height=680, hovermode="x unified", paper_bgcolor="#fff", plot_bgcolor="#fff", margin=dict(l=10, r=10, t=110, b=140),
                             legend=dict(orientation="h", yanchor="top", y=-0.3, x=0, font_size=11), xaxis2=dict(rangeslider=dict(visible=True, thickness=0.05, range=XR)))
            f2.update_xaxes(range=XR)
            f2.update_yaxes(title_text="Rain (mm)", range=[max(rmax, rain_thr) * 1.5, 0], row=1, col=1, showgrid=False)
            f2.update_yaxes(title_text="Index value", row=2, col=1, gridcolor="#EEF3FA")
            st.plotly_chart(f2, use_container_width=True)
            with st.expander("What do these indices tell me?"):
                st.markdown("\n".join(f"- **{k}**: {v[0]}" for k, v in INFO.items()))
                st.caption("Sentinel-2 is optical, so cloudy monsoon dates are removed (SCL mask) and gaps are normal. Use it to support the radar result, not to replace it.")
            st.dataframe(d2[["date"] + use + ["clear", "cloud"]].round(3), use_container_width=True)
            st.download_button("⬇ Sentinel-2 indices CSV", d2.to_csv(index=False), f"DREAM_S2_{sel}.csv")

elif page == "Ground truth":
    st.markdown("#### 🧪 Ground-truth accuracy check (DREAM 13.4)")
    st.caption("Upload water-tube observations (Pani-pipe photos or sensor readings): one row per observation with plot ID, date and wet/dry.")
    gt = st.file_uploader("Observations (CSV / XLSX)", type=["csv", "xlsx"], key="gt")
    if gt is None: st.info("Columns needed: plot ID, date, and either text (Wet/Dry, Water visible/Soil visible) or water level in cm."); st.stop()
    gdf_in = pd.read_csv(gt, dtype=str) if gt.name.lower().endswith("csv") else pd.read_excel(gt, dtype=str)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
    gid, gdate, gval = c1.selectbox("Plot ID", gdf_in.columns), c2.selectbox("Date", gdf_in.columns), c3.selectbox("Observation", gdf_in.columns)
    gval2 = c4.selectbox("Water level column (optional)", [None] + list(gdf_in.columns))
    kind = c5.radio("Observation type", ["Text (wet/dry)", "Water level (cm)"], horizontal=True)
    to_unit = {m.strip(): u for u, ms in zip(gdf.pid, gdf.members) for m in str(ms).split(",")}
    def num_dry(v):
        if pd.isna(v): return None
        try: return float(str(v).strip()) <= -15
        except ValueError: return None
    def gt_dry(v):
        if pd.isna(v): return None
        t = str(v).strip().lower()
        if kind.startswith("Water"):
            try: return float(t) <= -15
            except ValueError: return None
        if any(k in t for k in ("dry", "soil", "aerat", "drain")): return True
        if any(k in t for k in ("wet", "water", "flood")): return False
        return None
    def sat_dry(pid, d):
        R_ = res.get(to_unit.get(str(pid).strip(), str(pid).strip()))
        if R_ is None or R_["daily"].empty: return None
        row = R_["daily"][R_["daily"].date == d]
        return None if row.empty else bool(row.dry.iloc[0])
    t = pd.DataFrame({"Plot": gdf_in[gid], "Date": gdf_in[gdate].apply(parse_day).dt.normalize(),
                      "Ground": [a if a is not None else b for a, b in zip(gdf_in[gval].apply(gt_dry), gdf_in[gval2].apply(num_dry) if gval2 else [None] * len(gdf_in))]})
    t["Satellite"] = [sat_dry(p_, d_) for p_, d_ in zip(t.Plot, t.Date)]
    t["Match"] = [("✔" if a == b else "✘") if (a is not None and b is not None and not pd.isna(a) and not pd.isna(b)) else "—" for a, b in zip(t.Ground, t.Satellite)]
    ok = t.dropna(subset=["Ground", "Satellite"])
    if ok.empty: st.warning("No observation matched a plot/date with satellite status. Check plot IDs, dates and the audit window."); st.dataframe(t, use_container_width=True); st.stop()
    tp = int(((ok.Ground == True) & (ok.Satellite == True)).sum()); tn = int(((ok.Ground == False) & (ok.Satellite == False)).sum())
    fp = int(((ok.Ground == False) & (ok.Satellite == True)).sum()); fn = int(((ok.Ground == True) & (ok.Satellite == False)).sum())
    acc = 100 * (tp + tn) / len(ok)
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{acc:.1f}%", "pass ≥ 80%" if acc >= 80 else "below 80%", delta_color="normal" if acc >= 80 else "inverse")
    m2.metric("Matched samples", f"{len(ok)} of {len(t)}"); m3.metric("Uncertainty deduction", "15%" if acc >= 80 else "20%")
    if acc < 80: st.error("Accuracy below 80%: the draft (13.4.2.2) raises the deduction to 20% for this period and requires recalibrating the backscatter thresholds before the next season.")
    cm = go.Figure(go.Heatmap(z=[[tp, fn], [fp, tn]], x=["Satellite dry", "Satellite wet"], y=["Ground dry", "Ground wet"], text=[[tp, fn], [fp, tn]], texttemplate="%{text}", colorscale="Blues", showscale=False))
    cm.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=30, b=10), title="Confusion matrix")
    st.plotly_chart(cm, use_container_width=True)
    _lab = lambda v: "—" if v is None or pd.isna(v) else ("Dry" if v else "Wet")
    shown = t.copy(); shown["Date"] = shown.Date.dt.strftime("%d %b %Y"); shown["Ground"], shown["Satellite"] = shown.Ground.map(_lab), shown.Satellite.map(_lab)
    _clr = lambda v: ("background-color:#FFE4CC;color:#C2570C;font-weight:600" if v == "Dry" else "background-color:#DCEBFF;color:#1D5FD1;font-weight:600" if v == "Wet" else "")
    sty = shown.style; sty = (sty.map if hasattr(sty, "map") else sty.applymap)(_clr, subset=["Ground", "Satellite"])
    st.caption("🟧 Dry (orange) · 🟦 Wet (blue)"); st.dataframe(sty, use_container_width=True)

elif page == "Satellite Path":
    satellite_page(gdf, g.lat_c, g.lon_c)
