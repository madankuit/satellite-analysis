# satellite-analysis

Reproducible workflows for processing and evaluating satellite trace-gas products,
with a primary focus on **TROPOMI** and **TEMPO** L2/L3 observations (NO2, HCHO, CO).

---

## Repository layout

```
satellite-analysis/
├── Regriding/
│   ├── tropomi_regrid_l2_to_l3.py    # TROPOMI L2 → 0.05° CONUS L3 via OPeNDAP
│   ├── tropomi_tempo_recalc_match.py # match + recalc TROPOMI VCD on a shared a priori
│   ├── recalc_vcd_with_apriori.py    # recalculation core (general reference, any a priori)
│   └── tempo_monthly_means.py        # TEMPO L3 V04 per-UTC-hour monthly means
├── Visualization/
│   └── extract_tempo_at_monitor.py   # TEMPO VCD time-series at a ground monitor
├── utils/
│   └── satellite_utils.py            # Shared helper functions
├── environment.yml
└── README.md
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate satellite-analysis
```

---

## Scripts

### 1. `Regriding/tropomi_regrid_l2_to_l3.py`

Regrid TROPOMI L2 swath data (HCHO / NO2 / CO) to a regular **0.05° × 0.05°** CONUS
grid using nearest-neighbor interpolation with a 0.1° distance mask.
Data are read in real-time via NASA GES DISC **OPeNDAP** — no local L2 download required.

**Inputs (CLI)**

| Argument | Description |
|---|---|
| `--var` | Species: `HCHO`, `NO2`, or `CO` |
| `--years` | One or more years, e.g. `2022 2023 2024` |
| `--start-date` | MM-DD within each year, e.g. `04-01` |
| `--end-date` | MM-DD within each year, e.g. `09-30` |
| `--output-dir` | Directory for output NetCDF files |
| `--test-file` | Local reference TROPOMI L2 NetCDF (for layer count / attributes) |
| `--qa-threshold` | Minimum QA flag to retain (default `0.75`) |
| `--earthdata-username` | NASA Earthdata username (or `EARTHDATA_USERNAME` env var) |
| `--earthdata-password` | NASA Earthdata password (or `EARTHDATA_PASSWORD` env var) |

**Credential setup (recommended)**

```bash
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password
```

**Example**

```bash
python Regriding/tropomi_regrid_l2_to_l3.py \
    --var HCHO \
    --years 2022 2023 2024 \
    --start-date 04-01 \
    --end-date 09-30 \
    --output-dir /data/TROPOMI_L3/HCHO_CONUS_005/ \
    --test-file /data/TROPOMI_L2/S5P_RPRO_L2__HCHO___20180601T002121_....nc
```

**Output** — one NetCDF per day with variables:
- `TROPOMI_HCHO_TropVCD`, `TROPOMI_HCHO_TropVCDPrecision`
- `tropospheric_air_mass_factor`, `tropospheric_air_mass_factor_precision`
- `clear_air_mass_factor`, `cloudy_air_mass_factor`

---

### 2. `Regriding/tempo_monthly_means.py`

Calculate per-**UTC-half-hour** monthly mean VCDs from TEMPO L3 V04 files, with
quality-flag (DQ = 0) and effective cloud-fraction (< 0.2) filtering.

**Inputs (CLI)**

| Argument | Description |
|---|---|
| `--var` | `NO2_tropVCD`, `NO2_totalVCD`, `NO2_stratVCD`, or `HCHO_totalVCD` |
| `--years` | One or more years |
| `--months` | Zero-padded month(s), e.g. `04 05 06 07` |
| `--tempo-dir` | Root of TEMPO V04 data (contains `TEMPO_HCHO_L3_V04/` etc.) |

**Example**

```bash
python Regriding/tempo_monthly_means.py \
    --var HCHO_totalVCD \
    --years 2024 \
    --months 04 05 06 07 08 09 \
    --tempo-dir /data/TEMPO/V04/
```

**Output** — NetCDF files in `<tempo-dir>/<subdir>/MonthlyMeans/`
named e.g. `TEMPO_HCHO_L3_V04_202404_UTC0930_28Files.nc`.

---

### 3. `Visualization/extract_tempo_at_monitor.py`

Extract TEMPO L3 VCDs at a single ground-based monitor location and save a
time-series **CSV** (GMT and local time).

**Inputs (CLI)**

| Argument | Description |
|---|---|
| `--var` | `NO2`, `HCHO`, or `O3` |
| `--tempo-version` | `V03` or `V04` |
| `--tempo-dir` | Root of TEMPO data (contains `TEMPO_NO2_L3_V03/` etc.) |
| `--output-dir` | Directory for output CSV files |
| `--start-date` | `YYYY-MM-DD` |
| `--end-date` | `YYYY-MM-DD` (exclusive) |
| `--site-abbr` | Short site label used in output filename, e.g. `CCNY` |
| `--site-lat` | Site latitude (decimal degrees) |
| `--site-lon` | Site longitude (decimal degrees, negative = West) |
| `--aqs-id` | EPA AQS monitor ID used in output filename (optional) |

**Example**

```bash
python Visualization/extract_tempo_at_monitor.py \
    --var NO2 \
    --tempo-version V03 \
    --tempo-dir /data/TEMPO/BETA/ \
    --output-dir /data/TEMPO/CSVs/ \
    --start-date 2024-04-01 \
    --end-date   2024-09-01 \
    --site-abbr  CCNY \
    --site-lat   40.821 \
    --site-lon  -73.948 \
    --aqs-id     36-061-0135
```

---

### 4. `Regriding/tropomi_tempo_recalc_match.py`

**TEMPO–TROPOMI matching.** Match TROPOMI and TEMPO L2 pixels and **recalculate TROPOMI
VCD using the GEOS-CF a priori** from TEMPO, so both instruments share the same a priori
for a fair inter-comparison. Uses mass-conservative vertical interpolation (72 → 34
layers). Method reference: ESS Open Archive
[doi:10.22541/essoar.15007514/v1](https://essopenarchive.org/doi/abs/10.22541/essoar.15007514/v1).

**Inputs (CLI)**

| Argument | Description |
|---|---|
| `--var` | `HCHO` or `NO2` |
| `--year` | Year to process |
| `--start-date` / `--end-date` | MM-DD date range |
| `--output-dir` | Directory for daily point NetCDF files |
| `--tempo-dir` | Root of local TEMPO L2 V03 data |
| `--tropomi-test-file` | Local TROPOMI L2 NetCDF for unit conversion factor |
| `--earthdata-username/password` | NASA Earthdata credentials (or env vars) |

**Example**

```bash
python Regriding/tropomi_tempo_recalc_match.py \
    --var HCHO \
    --year 2024 \
    --start-date 04-01 \
    --end-date 08-01 \
    --output-dir /data/Matched/Recalc_HCHO/Daily_Points/ \
    --tempo-dir /data/TEMPO/BETA/ \
    --tropomi-test-file /data/TROPOMI_L2/S5P_RPRO_L2__HCHO___....nc
```

---

### TEMPO–TROPOMI matching & a-priori VCD recalculation: `Regriding/recalc_vcd_with_apriori.py`

**Method reference:** see the accompanying working paper for the full derivation and
validation — ESS Open Archive,
[doi:10.22541/essoar.15007514/v1](https://essopenarchive.org/doi/abs/10.22541/essoar.15007514/v1).

The **core recalculation**, isolated as a small general reference — no file I/O, no
pixel matching, no cluster paths. Recompute a tropospheric VCD under a *different* a
priori profile (MUSICA / GCHP / GEOS-CF) via the TROPOMI ATBD Eq. 4:

```
VCD_trop*  =  VCD_trop · Σ_trop(x_new) / Σ_trop(AK_trop · x_new)
```

`x_new` is the new a priori partial-column profile, mass-conservatively interpolated in
sigma = P/Ps onto the retrieval's own (TM5) layers; `AK_trop` is the tropospheric
averaging kernel (convert from a stored total-column AK with `AK_trop = AK_total ·
AMF_total / AMF_trop`, stratosphere zeroed); the troposphere is defined by the a priori
model's own tropopause. The a priori enters only as a ratio, so its normalization
cancels — pass partial columns directly, and swap a priori just by changing
`apriori_profile`. Two functions — `mass_conservative_sigma_interp()` and
`recalc_tropospheric_vcd()`; run the module directly for a synthetic example. This is
the shareable insight; the operational per-day matcher lives outside this repo.

---

### 5. `Visualization/extract_tropomi_at_monitor.py`

Extract TROPOMI L2 HiR VCDs at a single ground monitor over multiple years and
save to CSV. Finds the daily swath closest to 1:30 PM EDT overpass.

**Inputs (CLI)**

| Argument | Description |
|---|---|
| `--var` | `NO2`, `HCHO`, or `CO` |
| `--years` | Year(s) to process |
| `--start-date` / `--end-date` | MM-DD date range within each year |
| `--output-dir` | Directory for output CSV |
| `--tropomi-test-file` | Local TROPOMI L2 NetCDF for unit conversion factor |
| `--site-abbr` | Short site label, e.g. `CCNY` |
| `--site-lat` / `--site-lon` | Site coordinates |
| `--aqs-id` | EPA AQS ID (optional, used in filename) |
| `--qa-threshold` | QA filter (default `0.75`) |
| `--earthdata-username/password` | NASA Earthdata credentials (or env vars) |

**Example**

```bash
python Visualization/extract_tropomi_at_monitor.py \
    --var NO2 \
    --years 2018 2019 2020 2021 2022 2023 2024 \
    --start-date 04-01 \
    --end-date 09-01 \
    --output-dir /data/TROPOMI_L2/CSVs/ \
    --tropomi-test-file /data/TROPOMI_L2/S5P_RPRO_L2__NO2____....nc \
    --site-abbr CCNY \
    --site-lat 40.821 \
    --site-lon -73.948 \
    --aqs-id 36-061-0135
```

---

## Sections

### Regriding
- Regrid Level-2 products to Level-3
- Extract relevant variables (NO₂ VCD, HCHO VCD, QA values, AMFs)
- Match TROPOMI with TEMPO and recalculate using shared a priori

### Visualization
- Extract time series at monitoring sites (TROPOMI and TEMPO)
- Spatial maps
- Diurnal cycle plots

---
