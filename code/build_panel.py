"""Build a tidy section-visit panel from LTPP SDR 39 Access databases.

Target: MRI (mean roughness index, m/km) per section-visit.
Key:    STATE_CODE + SHRP_ID + CONSTRUCTION_NO + VISIT_DATE

CONSTRUCTION_NO is carried through the whole pipeline so that post-rehabilitation
observations are never blended into a single deterioration curve.
"""

import os
import warnings

import numpy as np
import pandas as pd
import pyodbc

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "code":
    ROOT = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "panel")
os.makedirs(OUT, exist_ok=True)

DRIVER = "DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};DBQ=%s;"


def q(module, dbfile, sql):
    path = os.path.join(os.environ.get("LTPP_RAW_DIR", os.path.join(ROOT, "extracted")),
                        f"SDR39 - {module}", f"{dbfile}.accdb")
    with pyodbc.connect(DRIVER % path) as cn:
        return pd.read_sql(sql, cn)


def key(df):
    df["STATE_CODE"] = df["STATE_CODE"].astype(int)
    df["SHRP_ID"] = df["SHRP_ID"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------- target
iri = q("MONITORING", "Monitoring", "SELECT * FROM [ANALYSIS_IRI]")
iri = key(iri)
iri["VISIT_DATE"] = pd.to_datetime(iri["VISIT_DATE"])
iri = iri[iri["MRI"].notna() & (iri["MRI"] > 0)]

# one record per section-CN-visit: average repeat profiler runs
iri = (
    iri.groupby(["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "VISIT_DATE"], as_index=False)
    .agg(MRI=("MRI", "mean"), n_runs=("MRI", "size"))
)
print(f"IRI visits (run-averaged): {len(iri):,}  sections: {iri.groupby(['STATE_CODE','SHRP_ID']).ngroups:,}")

# ------------------------------------------------------- age / structure
# Age origin. INV_AGE has the precise traffic-open date but covers GPS sections
# at CONSTRUCTION_NO=1 only (787 of 2,575 sections), which would silently discard
# every post-rehabilitation observation. EXPERIMENT_SECTION.CN_ASSIGN_DATE dates
# each construction event for all sections, so it is the primary origin and
# INV_AGE refines it where available.
es = key(q("INVENTORY", "Inventory",
           "SELECT STATE_CODE, SHRP_ID, CONSTRUCTION_NO, CN_ASSIGN_DATE, GPS_SPS, "
           "EXPERIMENT_NO FROM [EXPERIMENT_SECTION]"))
es["CN_ASSIGN_DATE"] = pd.to_datetime(es["CN_ASSIGN_DATE"], errors="coerce")
es = (es.sort_values("CN_ASSIGN_DATE")
        .drop_duplicates(subset=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"], keep="first"))

inv_age = key(q("INVENTORY", "Inventory",
                "SELECT STATE_CODE, SHRP_ID, CONSTRUCTION_NO, CONSTRUCTION_DATE, "
                "TRAFFIC_OPEN_DATE FROM [INV_AGE]"))
for c in ["CONSTRUCTION_DATE", "TRAFFIC_OPEN_DATE"]:
    inv_age[c] = pd.to_datetime(inv_age[c], errors="coerce")
inv_age["INV_OPEN"] = inv_age["TRAFFIC_OPEN_DATE"].fillna(inv_age["CONSTRUCTION_DATE"])

age = es.merge(inv_age[["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "INV_OPEN"]],
               on=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"], how="left")
age["OPEN_DATE"] = age["INV_OPEN"].fillna(age["CN_ASSIGN_DATE"])

# SECTION_LAYER_STRUCTURE covers all 2,575 sections and is keyed on
# CONSTRUCTION_NO, so the structure reflects the pavement as it stood at each
# construction event. INV_LAYER covers GPS/CN=1 only and is not used.
lay = key(q("MATERIAL_TEST", "Material_Test",
            "SELECT STATE_CODE, SHRP_ID, CONSTRUCTION_NO, LAYER_TYPE, REPR_THICKNESS "
            "FROM [SECTION_LAYER_STRUCTURE]"))
lay = lay[lay["REPR_THICKNESS"].notna()]
lay["LAYER_TYPE"] = lay["LAYER_TYPE"].astype(str).str.strip()
thk = (
    lay.pivot_table(index=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"],
                    columns="LAYER_TYPE", values="REPR_THICKNESS", aggfunc="sum")
    .add_prefix("thk_")
    .reset_index()
)
nlay = (lay.groupby(["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"], as_index=False)
        .size().rename(columns={"size": "n_layers"}))
thk = thk.merge(nlay, on=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"], how="left")
thk["thk_total"] = thk[[c for c in thk.columns if c.startswith("thk_")]].sum(axis=1)

gen = key(q("INVENTORY", "Inventory",
            "SELECT STATE_CODE, SHRP_ID, CONSTRUCTION_NO, NO_OF_LANES, PAVEMENT_TYPE, "
            "LANE_WIDTH, SUB_DRAINAGE_TYPE, DEPTH_TO_RIGID FROM [INV_GENERAL]"))

# ------------------------------------------------------------- climate
link = key(q("CLIMATE_SUMMARY_DATA", "Climate_Summary_Data",
             "SELECT STATE_CODE, SHRP_ID, VWS_ID FROM [CLM_SITE_VWS_LINK]"))
link["VWS_ID"] = link["VWS_ID"].astype(str).str.strip()

# Direct VWS-link coverage is reported by summarise_panel.py in raw_summary.csv.
# Remaining sections are assigned the
# nearest VWS by great-circle distance; clim_dist_km is kept as a feature so the
# assignment quality is visible to the model and to the reader.
coords = key(q("ADMINISTRATION", "Administration",
               "SELECT STATE_CODE, SHRP_ID, LATITUDE, LONGITUDE FROM [SECTION_COORDINATES]"))
coords = (coords.dropna(subset=["LATITUDE", "LONGITUDE"])
          .groupby(["STATE_CODE", "SHRP_ID"], as_index=False)[["LATITUDE", "LONGITUDE"]].mean())

vws_xy = (link.merge(coords, on=["STATE_CODE", "SHRP_ID"], how="inner")
          .groupby("VWS_ID", as_index=False)[["LATITUDE", "LONGITUDE"]].mean()
          .rename(columns={"LATITUDE": "vws_lat", "LONGITUDE": "vws_lon"}))
# keep only stations that actually have climate records
vws_have = set(q("CLIMATE_SUMMARY_DATA", "Climate_Summary_Data",
                 "SELECT DISTINCT VWS_ID FROM [CLM_VWS_TEMP_ANNUAL]")
               ["VWS_ID"].astype(str).str.strip())
vws_xy = vws_xy[vws_xy["VWS_ID"].isin(vws_have)].reset_index(drop=True)


def haversine(lat1, lon1, lat2, lon2):
    r = np.radians
    dlat, dlon = r(lat2 - lat1), r(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(r(lat1)) * np.cos(r(lat2)) * np.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


sec_xy = coords.copy()
d = haversine(sec_xy["LATITUDE"].values[:, None], sec_xy["LONGITUDE"].values[:, None],
              vws_xy["vws_lat"].values[None, :], vws_xy["vws_lon"].values[None, :])
nearest = d.argmin(axis=1)
sec_xy["VWS_NEAREST"] = vws_xy["VWS_ID"].values[nearest]
sec_xy["clim_dist_km"] = d.min(axis=1)

link = sec_xy.merge(link, on=["STATE_CODE", "SHRP_ID"], how="left")
link["VWS_ID"] = link["VWS_ID"].fillna(link["VWS_NEAREST"])
link["clim_onsite"] = (link["clim_dist_km"] < 1.0).astype(int)
link = link[["STATE_CODE", "SHRP_ID", "VWS_ID", "clim_dist_km", "clim_onsite",
             "LATITUDE", "LONGITUDE"]]
print(f"climate assigned: {len(link):,} sections, median distance "
      f"{link.clim_dist_km.median():.1f} km, on-site {link.clim_onsite.mean()*100:.0f}%")

# --- climate: trailing 12 months ending the month BEFORE each survey ---------
# The previous version joined calendar-year annual summaries on the survey year,
# which admitted post-measurement information for any survey before December.
# Monthly tables are aggregated over the 12 months ending one month before the
# survey. Availability at an earlier forecast issue date is not established.
mt = q("CLIMATE_SUMMARY_DATA", "Climate_Summary_Data",
       "SELECT VWS_ID, YEAR, MONTH, MEAN_MON_TEMP_AVG, MAX_MON_TEMP, MIN_MON_TEMP "
       "FROM [CLM_VWS_TEMP_MONTH]")
mp = q("CLIMATE_SUMMARY_DATA", "Climate_Summary_Data",
       "SELECT VWS_ID, YEAR, MONTH, TOTAL_MON_PRECIP, WET_DAYS_MONTH, "
       "TOTAL_SNOWFALL_MONTH FROM [CLM_VWS_PRECIP_MONTH]")
mh = q("CLIMATE_SUMMARY_DATA", "Climate_Summary_Data",
       "SELECT VWS_ID, YEAR, MONTH, MAX_MON_HUM_AVG, MIN_MON_HUM_AVG "
       "FROM [CLM_VWS_HUMIDITY_MONTH]")

mon = mt.merge(mp, on=["VWS_ID", "YEAR", "MONTH"], how="outer")         .merge(mh, on=["VWS_ID", "YEAR", "MONTH"], how="outer")
mon["VWS_ID"] = mon["VWS_ID"].astype(str).str.strip()
mon = mon.dropna(subset=["YEAR", "MONTH"])
mon["mi"] = mon["YEAR"].astype(int) * 12 + (mon["MONTH"].astype(int) - 1)
# freeze-thaw proxy: months that both drop below and rise above 0 C
mon["ft_month"] = ((mon["MIN_MON_TEMP"] < 0) & (mon["MAX_MON_TEMP"] > 0)).astype(float)
mon = mon.sort_values(["VWS_ID", "mi"])

def _roll(g):
    g = g.set_index("mi")
    full = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))
    r = full.rolling(12, min_periods=6)
    out = pd.DataFrame({
        "t12_mean_temp":  r["MEAN_MON_TEMP_AVG"].mean(),
        "t12_max_temp":   r["MAX_MON_TEMP"].max(),
        "t12_min_temp":   r["MIN_MON_TEMP"].min(),
        "t12_freeze_thaw": r["ft_month"].sum(),
        "t12_precip":     r["TOTAL_MON_PRECIP"].sum(),
        "t12_wet_days":   r["WET_DAYS_MONTH"].sum(),
        "t12_snowfall":   r["TOTAL_SNOWFALL_MONTH"].sum(),
        "t12_max_hum":    r["MAX_MON_HUM_AVG"].mean(),
        "t12_min_hum":    r["MIN_MON_HUM_AVG"].mean(),
    })
    out.index.name = "mi"
    return out.reset_index()

clm = (mon.groupby("VWS_ID", group_keys=True)[
          ["mi", "MEAN_MON_TEMP_AVG", "MAX_MON_TEMP", "MIN_MON_TEMP", "ft_month",
           "TOTAL_MON_PRECIP", "WET_DAYS_MONTH", "TOTAL_SNOWFALL_MONTH",
           "MAX_MON_HUM_AVG", "MIN_MON_HUM_AVG"]]
       .apply(_roll).reset_index(level=0))
print(f"climate windows built: {len(clm):,} VWS-months")
link["VWS_ID"] = link["VWS_ID"].astype(str).str.strip()

# ------------------------------------------------------------- traffic
# TRF_TREND covers all sections, is keyed on CONSTRUCTION_NO + YEAR, and already
# carries cumulative loading since the construction event.
trf = key(q("TRAFFIC", "Traffic",
            "SELECT STATE_CODE, SHRP_ID, CONSTRUCTION_NO, YEAR, AADTT_ALL_TRUCKS_TREND, "
            "ANNUAL_TRUCK_VOLUME_TREND, CMLTV_VOL_HEAVY_TRUCKS_TREND, "
            "ANNUAL_ESAL_TREND, ANNUAL_GESAL_TREND, ANNUAL_TOTAL_GVW_TREND "
            "FROM [TRF_TREND]"))
trf = trf.groupby(["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "YEAR"],
                  as_index=False).mean(numeric_only=True)
# shift forward one year: a survey in year Y is matched to traffic for Y-1, the
# last complete calendar year before the survey; earlier issuance is unverified
trf["YEAR"] = trf["YEAR"] + 1

esal = key(q("TRAFFIC", "Traffic",
             "SELECT STATE_CODE, SHRP_ID, YEAR, KESAL_YEAR FROM [TRF_ESAL_COMPUTED]"))
esal = esal.groupby(["STATE_CODE", "SHRP_ID", "YEAR"], as_index=False).KESAL_YEAR.mean()
esal["YEAR"] = esal["YEAR"] + 1          # same one-year lag as TRF_TREND

fun = key(q("TRAFFIC", "Traffic",
            "SELECT STATE_CODE, SHRP_ID, FUNC_CLASS, TOT_LANES FROM [TRF_BASIC_INFO]"))
fun = fun.drop_duplicates(subset=["STATE_CODE", "SHRP_ID"])

# ------------------------------------------------------------- assemble
df = iri.merge(age[["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "OPEN_DATE",
                    "GPS_SPS", "EXPERIMENT_NO"]],
               on=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"], how="left")
df["age_years"] = (df["VISIT_DATE"] - df["OPEN_DATE"]).dt.days / 365.25
df = df[df["age_years"].notna() & (df["age_years"] >= 0)]
df["YEAR"] = df["VISIT_DATE"].dt.year
# window ends the month before the survey
df["mi"] = df["YEAR"] * 12 + (df["VISIT_DATE"].dt.month - 1) - 1

df = df.merge(thk, on=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"], how="left")
df = df.merge(gen, on=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"], how="left")
df = df.merge(link, on=["STATE_CODE", "SHRP_ID"], how="left")
df = df.merge(clm, on=["VWS_ID", "mi"], how="left")
df = df.merge(trf, on=["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "YEAR"], how="left")
df = df.merge(esal, on=["STATE_CODE", "SHRP_ID", "YEAR"], how="left")
df = df.merge(fun, on=["STATE_CODE", "SHRP_ID"], how="left")

# Completed-calendar-year event-specific traffic exposure, independent of visits.
df = df.sort_values(["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO", "VISIT_DATE"])
grp = df.groupby(["STATE_CODE", "SHRP_ID", "CONSTRUCTION_NO"])
from annual_loading import cumulative_loading
annual_traffic = trf.copy()
annual_traffic["YEAR"] -= 1  # undo join-key shift; these are the measured years
df["cum_kesal"], loading_ledger = cumulative_loading(df, annual_traffic)
loading_ledger.to_csv(os.path.join(OUT, "annual_loading_provenance.csv"), index=False)

# First recorded event MRI; baseline rows are excluded by the modelling loader.
df["initial_MRI"] = grp["MRI"].transform("first")
df["visit_seq"] = grp.cumcount()

df["section_id"] = df["STATE_CODE"].astype(str) + "_" + df["SHRP_ID"]
df["unit_id"] = df["section_id"] + "_CN" + df["CONSTRUCTION_NO"].astype(str)

df.to_parquet(os.path.join(OUT, "ltpp_panel.parquet"), index=False)

feat = [c for c in df.columns if c not in
        ("MRI", "VISIT_DATE", "OPEN_DATE", "section_id", "unit_id", "SHRP_ID", "VWS_ID", "n_runs", "mi")]
core = ["age_years", "thk_total", "t12_mean_temp", "t12_precip",
        "AADTT_ALL_TRUCKS_TREND"]
complete = df.dropna(subset=core)

print(f"\npanel rows                : {len(df):,}")
print(f"unique sections           : {df.section_id.nunique():,}")
print(f"unique section-CN units   : {df.unit_id.nunique():,}")
print(f"date range                : {df.VISIT_DATE.min().date()} -> {df.VISIT_DATE.max().date()}")
print(f"rows w/ core features     : {len(complete):,}  ({complete.section_id.nunique():,} sections)")
print(f"\nmissingness by column (top 15):")
print((df[feat].isna().mean().sort_values(ascending=False).head(15) * 100).round(1).to_string())
