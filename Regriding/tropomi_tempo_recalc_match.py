"""
Match TROPOMI and TEMPO L2 pixels and recalculate TROPOMI VCD using the
GEOS-CF a priori gas profile from TEMPO, so both instruments share the same
a priori for fair comparison.

Algorithm (per day)
-------------------
1. Find the TROPOMI L2 swath file closest to 1:30 PM EDT overpass over the
   target region (via OPeNDAP).
2. Find all TEMPO L2 files that observe the target region at ~17 UTC.
3. For each TEMPO pixel, locate the nearest TROPOMI pixel (KDTree, ≤ 0.02°).
4. Vertically interpolate TEMPO GEOS-CF gas profile from 72 → 34 TM5 layers
   using mass-conservative interpolation.
5. Compute the recalculation factor:
       factor = (1-w) * Σ(χ_GC) / Σ(AK * χ_GC)
              +    w  * Σ_cloud-to-TOA(χ_GC) / Σ_cloud-to-TOA(AK * χ_GC)
6. Save one NetCDF per day with matched TROPOMI/TEMPO points and recalculated
   TROPOMI VCDs.

Input
-----
- TROPOMI L2 HiR v2.4 (HCHO or NO2) accessed via OPeNDAP — requires NASA
  Earthdata credentials.
- TEMPO L2 V03 files stored locally in --tempo-dir.
- One local reference TROPOMI L2 NetCDF (--tropomi-test-file) to read the
  unit conversion factor.

Output
------
- One NetCDF per day in --output-dir containing:
    TROPOMI_latitudes, TROPOMI_longitudes
    TROPOMI_filtered_VCDvals   (mol cm⁻²)
    Matched_TEMPO_VCDvals      (mol cm⁻²)
    GCScaledFactor_vals
    Recalc_TROPOMI_VCDvals     (mol cm⁻²)

Usage example
-------------
    python tropomi_tempo_recalc_match.py \\
        --var HCHO \\
        --year 2024 \\
        --start-date 04-01 \\
        --end-date 08-01 \\
        --output-dir /data/Matched_TEMPO_TROPOMI/Recalc_HCHO/Daily_Points/ \\
        --tempo-dir /data/TEMPO/BETA/ \\
        --tropomi-test-file /data/TROPOMI_L2/S5P_RPRO_L2__HCHO___....nc

    export EARTHDATA_USERNAME=your_username
    export EARTHDATA_PASSWORD=your_password

MODIFICATION HISTORY
--------------------
    Original author: 2025-02-04, VERSION 1.0
    2025-02-17, VERSION 2.0 - Use TM5 lower-edge pressure from NO2 file
    2025-02-18, VERSION 3.0 - Mass-conservative vertical interpolation on sigma pressure
    Refactored: personal paths/credentials replaced with CLI arguments
"""

import argparse
import os
import re
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import requests
from bs4 import BeautifulSoup
from scipy.spatial import KDTree
from pydap.client import open_url
from pydap.cas.urs import setup_session

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.satellite_utils import get_day_of_year, is_on_date

# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
PRODUCT_INFO = {
    'HCHO': 'S5P_L2__HCHO___HiR.2',
    'NO2':  'S5P_L2__NO2____HiR.2',
}
FILE_HEADER = {
    'HCHO': 'S5P_OFFL_L2__HCHO___',
    'NO2':  'S5P_OFFL_L2__NO2____',
}

# Northeast U.S. bounding box (OTC region)
LON_RIGHT, LAT_BOT = -81.0, 36.5
LON_LEFT,  LAT_TOP = -66.5, 49.0

# UTC time of TROPOMI overpass over NE US (~1:30 PM EDT = 17:30 UTC)
TROPOMI_TARGET_UTC = '173000'

FILL_VAL = 9.9692100e+36
DQ_FLAG  = 0     # TEMPO good quality
EFF_CLOUD = 0.2  # TEMPO effective cloud fraction


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_opendap_directory(varname, url):
    """Return file links listed at an OPeNDAP catalogue URL."""
    fh = FILE_HEADER[varname]
    resp = requests.get(url)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.content, 'html.parser')
        return [a.get('href') for a in soup.find_all('a')
                if a.get('href', '').startswith((fh, '0'))]
    print(f"Failed to access {url}")
    return []


def get_closest_utc_file(filenames, varname, target_utc_str):
    """Return the filename whose scan-start time is closest to *target_utc_str* (HHMMSS)."""
    fh = FILE_HEADER[varname]
    closest, min_diff = None, timedelta.max
    for f in filenames:
        try:
            start_str = f.split(fh)[1][:15]
            start_dt  = datetime.strptime(start_str, '%Y%m%dT%H%M%S')
            date_str  = start_str[:8]
            target_dt = datetime.strptime(date_str + 'T' + target_utc_str, '%Y%m%dT%H%M%S')
            diff = abs(start_dt - target_dt)
            if diff < min_diff:
                min_diff, closest = diff, f
        except (IndexError, ValueError):
            continue
    return closest


def get_tempo_files_near_utc17(directory, given_day):
    """Return sorted TEMPO L2 filenames for *given_day* that contain 'T17'."""
    return sorted(f for f in os.listdir(directory) if 'T17' in f and given_day in f)


def extract_tempo_timestamp(filename):
    try:
        return datetime.strptime(filename.split('_')[4][:15], '%Y%m%dT%H%M%S')
    except ValueError:
        return None


def find_closest_within_threshold(query_loc, kdtree, threshold=0.02):
    """Return flat index of closest point in *kdtree*, or NaN if beyond threshold."""
    dist, idx = kdtree.query(query_loc)
    return idx if dist <= threshold else np.nan


def mass_conservative_interpolation(pressure_source, values_source, pressure_target):
    """
    Redistribute partial column values from a high-resolution pressure grid to a
    coarser target grid while conserving total column mass.

    Parameters
    ----------
    pressure_source : 1-D array  (e.g. 72 sigma levels, surface → TOA)
    values_source   : 1-D array  partial column amounts at each source layer
    pressure_target : 1-D array  (e.g. 35 sigma levels, surface → TOA)

    Returns
    -------
    1-D array of length len(pressure_target) - 1
    """
    # Sort descending (surface → TOA = decreasing pressure)
    if pressure_source[0] < pressure_source[-1]:
        pressure_source = pressure_source[::-1]
        values_source   = values_source[::-1]
    if pressure_target[0] < pressure_target[-1]:
        pressure_target = pressure_target[::-1]

    dp_src = np.abs(np.diff(pressure_source))
    n_tgt  = len(pressure_target) - 1
    values_target = np.zeros(n_tgt)

    for j in range(n_tgt):
        p_top, p_bot = pressure_target[j + 1], pressure_target[j]
        for i in range(len(dp_src)):
            s_top, s_bot = pressure_source[i + 1], pressure_source[i]
            if s_bot > p_top and s_top < p_bot:
                overlap = min(s_bot, p_bot) - max(s_top, p_top)
                if overlap > 0:
                    values_target[j] += values_source[i] * (overlap / dp_src[i])

    if len(values_target) == 33:
        values_target = np.append(values_target, np.nan)

    return values_target


# ---------------------------------------------------------------------------
# Per-day processing
# ---------------------------------------------------------------------------

def process_day(varname, given_day, session, tempo_dir,
                output_dir, multiplication_factor,
                tm5_a_lower, tm5_b_lower):
    """Match TROPOMI and TEMPO L2 pixels for one day and save to NetCDF."""
    out_path = os.path.join(output_dir, f'Matched_TEMPO_TROPOMI_{given_day}.nc')
    if os.path.exists(out_path):
        print(f"  {given_day} already processed — skipping.")
        return

    sel_year    = given_day[:4]
    day_of_year = get_day_of_year(given_day)

    # ------------------------------------------------------------------
    # Step 1: Get TM5 pressure-level constants from closest NO2 file
    # ------------------------------------------------------------------
    no2_base_url = (
        f'https://tropomi.gesdisc.eosdis.nasa.gov/opendap/'
        f'S5P_TROPOMI_Level2/{PRODUCT_INFO["NO2"]}/{sel_year}/{day_of_year}/'
    )
    no2_links = parse_opendap_directory('NO2', no2_base_url)
    no2_files = sorted(set(
        no2_base_url + f[:-5] for f in no2_links
        if f.endswith('.nc.html') and is_on_date(f, given_day)
    ))
    if len(no2_files) < 4:
        print(f"  Not enough NO2 files on {given_day}.")
        return

    closest_no2 = get_closest_utc_file(no2_files, 'NO2', TROPOMI_TARGET_UTC)
    no2_ds = open_url(closest_no2, session=session)
    tm5_a = np.array(no2_ds['PRODUCT_tm5_constant_a'][:])[:, 0]
    tm5_b = np.array(no2_ds['PRODUCT_tm5_constant_b'][:])[:, 0]

    # ------------------------------------------------------------------
    # Step 2: Read TROPOMI HCHO (or NO2) file closest to 1:30 PM EDT
    # ------------------------------------------------------------------
    hcho_base_url = (
        f'https://tropomi.gesdisc.eosdis.nasa.gov/opendap/'
        f'S5P_TROPOMI_Level2/{PRODUCT_INFO[varname]}/{sel_year}/{day_of_year}/'
    )
    hcho_links = parse_opendap_directory(varname, hcho_base_url)
    hcho_files = sorted(set(
        hcho_base_url + f[:-5] for f in hcho_links
        if f.endswith('.nc.html') and is_on_date(f, given_day)
    ))
    if len(hcho_files) < 4:
        print(f"  Not enough {varname} files on {given_day}.")
        return

    closest_hcho = get_closest_utc_file(hcho_files, varname, TROPOMI_TARGET_UTC)
    TROPOMI_scan_time = '_'.join(closest_hcho.split('_')[15:17])

    trop_ds = open_url(closest_hcho, session=session)

    lat  = np.array(trop_ds['PRODUCT_latitude'][:])[0]
    lon  = np.array(trop_ds['PRODUCT_longitude'][:])[0]
    qa   = np.array(trop_ds['PRODUCT_qa_value'][:])[0]

    if varname == 'HCHO':
        vcd = np.array(trop_ds['PRODUCT_formaldehyde_tropospheric_vertical_column'][:])[0]
    else:
        vcd = np.array(trop_ds['PRODUCT_nitrogendioxide_tropospheric_column'][:])[0]

    ak          = np.array(trop_ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_averaging_kernel'][:])[0]
    cloud_frac  = np.array(trop_ds['PRODUCT_SUPPORT_DATA_INPUT_DATA_cloud_fraction_crb'][:])[0]
    cloud_pres  = np.array(trop_ds['PRODUCT_SUPPORT_DATA_INPUT_DATA_cloud_pressure_crb'][:])[0]
    cloud_wfrac = np.array(trop_ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_cloud_fraction_intensity_weighted'][:])[0]
    surf_pres   = np.array(trop_ds['PRODUCT_SUPPORT_DATA_INPUT_DATA_surface_pressure'][:])[0]

    vcd[vcd == FILL_VAL] = np.nan

    # Auto-detect QA scale (0–1 vs 0–100)
    qa_thresh = 75 if np.nanmax(qa) > 1.5 else 0.75
    filt_vcd = np.where(qa > qa_thresh, vcd, np.nan)

    # Flatten and apply regional + QA mask
    flat_lat  = lat.ravel();  flat_lon = lon.ravel()
    flat_vcd  = filt_vcd.ravel()
    in_region = (
        ~np.isnan(flat_vcd) &
        (flat_lon >= LON_RIGHT) & (flat_lon <= LON_LEFT) &
        (flat_lat >= LAT_BOT)  & (flat_lat <= LAT_TOP)
    )
    t_lat  = flat_lat[in_region];  t_lon = flat_lon[in_region]
    t_vcd  = flat_vcd[in_region] * multiplication_factor
    t_cfrac   = cloud_frac.ravel()[in_region]
    t_cwfrac  = cloud_wfrac.ravel()[in_region]
    t_cpres   = cloud_pres.ravel()[in_region]
    t_ps      = surf_pres.ravel()[in_region]

    _, _, layer_dim = ak.shape
    t_ak = ak.reshape(-1, layer_dim)[in_region, :]

    # Pressure levels for each pixel: shape (N, layer_dim)
    t_plower = (tm5_a[np.newaxis, :layer_dim] +
                tm5_b[np.newaxis, :layer_dim] * t_ps[:, np.newaxis])

    n_matched = len(t_vcd)
    matched_tempo_vcd   = np.full(n_matched, np.nan)
    gc_scale_factor     = np.full(n_matched, np.nan)
    recalc_tropomi_vcd  = np.full(n_matched, np.nan)

    trop_kdtree = KDTree(np.column_stack((t_lat, t_lon)))

    # ------------------------------------------------------------------
    # Step 3: Read TEMPO L2 files and match pixels
    # ------------------------------------------------------------------
    tempo_var_dir = os.path.join(
        tempo_dir,
        'TEMPO_HCHO_L2_V03' if varname == 'HCHO' else 'TEMPO_NO2_L2_V03'
    ) + os.sep

    tempo_files = get_tempo_files_near_utc17(tempo_var_dir, given_day)
    if not tempo_files:
        print(f"  No TEMPO files on {given_day}.")
        return

    target_dt = datetime.strptime(f'{given_day}T1730', '%Y%m%dT%H%M')
    sorted_tempo = sorted(
        [(f, extract_tempo_timestamp(f)) for f in tempo_files if extract_tempo_timestamp(f)],
        key=lambda x: abs(x[1] - target_dt)
    )
    sorted_tempo_files = list(dict.fromkeys(f for f, _ in sorted_tempo))
    TEMPO_scan_time = (
        f'{sorted_tempo_files[0].split("_")[4]}_{sorted_tempo_files[-1].split("_")[4]}'
        if sorted_tempo_files else ''
    )

    for tempo_fname in sorted_tempo_files:
        try:
            fi_ds   = xr.open_dataset(tempo_var_dir + tempo_fname, group='product')
            geo_ds  = xr.open_dataset(tempo_var_dir + tempo_fname, group='geolocation')
            supp_ds = xr.open_dataset(tempo_var_dir + tempo_fname, group='support_data')

            o_lat = geo_ds['latitude'].values
            o_lon = geo_ds['longitude'].values
            var_col = 'vertical_column'
            fi_ds[var_col] = fi_ds[var_col].where(fi_ds['main_data_quality_flag'] == DQ_FLAG, np.nan)
            fi_ds[var_col] = fi_ds[var_col].where(supp_ds['eff_cloud_fraction'] < EFF_CLOUD, np.nan)

            te_vcd = fi_ds[var_col].values.ravel()
            te_lat = o_lat.ravel();  te_lon = o_lon.ravel()
            te_ps  = supp_ds['surface_pressure'].values.ravel() * 100
            eta_a  = supp_ds['surface_pressure'].Eta_A.values
            eta_b  = supp_ds['surface_pressure'].Eta_B.values

            _, _, swt_dim = supp_ds['gas_profile'].values.shape
            te_gp = supp_ds['gas_profile'].values.reshape(-1, swt_dim)

            valid = (
                ~np.isnan(te_vcd) &
                (te_lon >= LON_RIGHT) & (te_lon <= LON_LEFT) &
                (te_lat >= LAT_BOT)  & (te_lat <= LAT_TOP)
            )
            te_vcd  = te_vcd[valid];  te_lat = te_lat[valid]; te_lon = te_lon[valid]
            te_ps   = te_ps[valid];   te_gp  = te_gp[valid, :]

            if np.nansum(~np.isnan(te_vcd)) < 100:
                continue

            te_plower = (eta_a[np.newaxis, :swt_dim] +
                         eta_b[np.newaxis, :swt_dim] * te_ps[:, np.newaxis])

            print(f"    Matching with {tempo_fname} ({len(te_vcd)} pixels) …")

            for pi, (te_la, te_lo) in enumerate(zip(te_lat, te_lon)):
                ci = find_closest_within_threshold((te_la, te_lo), trop_kdtree)
                if np.isnan(ci):
                    continue
                ci = int(ci)
                if not np.isnan(matched_tempo_vcd[ci]):
                    continue  # already filled

                # Sigma pressure
                trop_sigma = t_plower[ci] / t_ps[ci]
                tempo_sigma = te_plower[pi] / te_ps[pi]

                # Mass-conservative vertical interpolation
                interp_gp = mass_conservative_interpolation(tempo_sigma, te_gp[pi], trop_sigma)

                total_col = np.nansum(interp_gp)
                if total_col == 0 or np.isnan(total_col):
                    norm_gp = np.full_like(interp_gp, np.nan)
                else:
                    norm_gp = interp_gp / total_col

                # Recalculation factor (clear + cloudy components)
                ak_i = t_ak[ci]
                w    = t_cwfrac[ci]
                clear_factor = (1 - w) * (np.nansum(norm_gp) / np.nansum(norm_gp * ak_i))

                cloud_mask = t_plower[ci] <= t_cpres[ci]
                norm_gp_cloud = norm_gp[cloud_mask]
                ak_cloud      = ak_i[cloud_mask]
                cloudy_factor = w * (np.nansum(norm_gp_cloud) / np.nansum(norm_gp_cloud * ak_cloud))

                factor = clear_factor + cloudy_factor

                matched_tempo_vcd[ci]  = te_vcd[pi]
                gc_scale_factor[ci]    = factor
                recalc_tropomi_vcd[ci] = t_vcd[ci] * factor

            fi_ds.close();  geo_ds.close();  supp_ds.close()

        except Exception as exc:
            print(f"    Skipped {tempo_fname}: {exc}")

    # ------------------------------------------------------------------
    # Step 4: Save matched points
    # ------------------------------------------------------------------
    mask = ~(np.isnan(t_lat) | np.isnan(t_lon) | np.isnan(t_vcd) | np.isnan(matched_tempo_vcd))
    print(f"  {given_day}: {mask.sum()} matched points.")

    ds_out = xr.Dataset({
        'TROPOMI_latitudes':           ('points', t_lat[mask]),
        'TROPOMI_longitudes':          ('points', t_lon[mask]),
        'TROPOMI_filtered_VCDvals':    ('points', t_vcd[mask]),
        'Matched_TEMPO_VCDvals':       ('points', matched_tempo_vcd[mask]),
        'GCScaledFactor_vals':         ('points', gc_scale_factor[mask]),
        'Recalc_TROPOMI_VCDvals':      ('points', recalc_tropomi_vcd[mask]),
    })
    ds_out.attrs['TROPOMI_TEMPO_Scantime'] = f'{TROPOMI_scan_time} | {TEMPO_scan_time}'

    os.makedirs(output_dir, exist_ok=True)
    ds_out.to_netcdf(out_path, mode='w', format='NETCDF4')
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Match TROPOMI and TEMPO L2 and recalculate TROPOMI VCD '
                    'using GEOS-CF a priori from TEMPO.'
    )
    parser.add_argument('--var', choices=['HCHO', 'NO2'], default='HCHO',
                        help='Species (default: HCHO)')
    parser.add_argument('--year', default='2024',
                        help='Year to process (default: 2024)')
    parser.add_argument('--start-date', default='04-01',
                        help='Start date MM-DD (default: 04-01)')
    parser.add_argument('--end-date', default='08-01',
                        help='End date MM-DD inclusive (default: 08-01)')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for output daily point NetCDF files')
    parser.add_argument('--tempo-dir', required=True,
                        help='Root of local TEMPO L2 V03 data, containing '
                             'TEMPO_HCHO_L2_V03/ and TEMPO_NO2_L2_V03/')
    parser.add_argument('--tropomi-test-file', required=True,
                        help='Local TROPOMI L2 NetCDF used to read the '
                             'molecules/cm² unit conversion factor')
    parser.add_argument('--earthdata-username',
                        default=os.environ.get('EARTHDATA_USERNAME'),
                        help='NASA Earthdata username (or EARTHDATA_USERNAME env var)')
    parser.add_argument('--earthdata-password',
                        default=os.environ.get('EARTHDATA_PASSWORD'),
                        help='NASA Earthdata password (or EARTHDATA_PASSWORD env var)')
    args = parser.parse_args()

    if not args.earthdata_username or not args.earthdata_password:
        parser.error(
            'NASA Earthdata credentials required. Pass --earthdata-username / '
            '--earthdata-password or set EARTHDATA_USERNAME / EARTHDATA_PASSWORD.'
        )

    # Read multiplication factor from reference file
    ref_ds = xr.open_dataset(args.tropomi_test_file, group='/PRODUCT/', engine='netcdf4')
    if args.var == 'HCHO':
        mult = float(ref_ds['formaldehyde_tropospheric_vertical_column']
                     .multiplication_factor_to_convert_to_molecules_percm2)
    else:
        mult = float(ref_ds['nitrogendioxide_tropospheric_column']
                     .multiplication_factor_to_convert_to_molecules_percm2)
    ref_ds.close()
    print(f'Unit conversion factor: {mult:.3e}')

    check_url = (
        f'https://tropomi.gesdisc.eosdis.nasa.gov/opendap/'
        f'S5P_TROPOMI_Level2/{PRODUCT_INFO[args.var]}/'
    )
    session = setup_session(args.earthdata_username, args.earthdata_password,
                            check_url=check_url)

    start = datetime.strptime(f'{args.year}-{args.start_date}', '%Y-%m-%d')
    end   = datetime.strptime(f'{args.year}-{args.end_date}',   '%Y-%m-%d')
    days  = [(start + timedelta(days=x)).strftime('%Y%m%d')
             for x in range((end - start).days + 1)]
    print(f'{len(days)} day(s) to process.')

    for day in days:
        print(f'\n--- {day} ---')
        try:
            process_day(
                varname=args.var,
                given_day=day,
                session=session,
                tempo_dir=args.tempo_dir,
                output_dir=args.output_dir,
                multiplication_factor=mult,
                tm5_a_lower=None,  # fetched inside process_day
                tm5_b_lower=None,
            )
        except Exception as exc:
            print(f'  Error: {exc}')


if __name__ == '__main__':
    main()
