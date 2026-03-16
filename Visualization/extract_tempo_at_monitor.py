"""
Extract TEMPO L3 V03/V04 trace-gas VCDs at a single ground-based monitor site.

For each day in the requested date range the script locates all TEMPO scan
files, finds the nearest grid cell to the site, applies quality and cloud
filters, and saves a time-series CSV (UTC times and, optionally, local times).

Supported species (--var)
-------------------------
    NO2  - extracts tropospheric, stratospheric, and total NO2
    HCHO - extracts total HCHO

Usage example
-------------
    python extract_tempo_at_monitor.py \\
        --var NO2 \\
        --tempo-version V03 \\
        --tempo-dir /path/to/TEMPO/BETA/ \\
        --output-dir /path/to/output/CSVs/ \\
        --start-date 2024-04-01 \\
        --end-date   2024-09-01 \\
        --site-abbr CCNY \\
        --site-lat  40.821 \\
        --site-lon -73.948 \\
        --aqs-id    36-061-0135

MODIFICATION HISTORY
--------------------
    Original author: 2024-11-25, VERSION 1.0
    Refactored: personal paths / site info replaced with CLI arguments
"""

import argparse
import os
import re
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import pytz

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Add repo root to path so satellite_utils is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.satellite_utils import find_nearest, filter_filenames, convert_to_local_time


# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
VAR_SUBDIR = {
    ('NO2',  'V03'): 'TEMPO_NO2_L3_V03',
    ('NO2',  'V04'): 'TEMPO_NO2_L3_V04',
    ('HCHO', 'V03'): 'TEMPO_HCHO_L3_V03',
    ('HCHO', 'V04'): 'TEMPO_HCHO_L3_V04',
    ('O3',   'V03'): 'TEMPO_O3TOT_L3_V03',
    ('O3',   'V04'): 'TEMPO_O3TOT_L3_V04',
}

DQ_FLAG   = 0    # Good quality only
EFF_CLOUD = 0.2  # Effective cloud fraction threshold


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_site(varname, tempo_version, tempo_dir, output_dir,
                 start_date, end_date,
                 site_abbr, site_lat, site_lon, aqs_id):
    """
    Extract TEMPO VCDs at a single site and save to CSV.
    """
    key = (varname, tempo_version)
    if key not in VAR_SUBDIR:
        raise ValueError(f'Unsupported (var, version) combination: {key}')

    subdir     = VAR_SUBDIR[key]
    vari_dir   = os.path.join(tempo_dir, subdir) + os.sep
    os.makedirs(output_dir, exist_ok=True)

    # Date list
    read_start = datetime.strptime(start_date, '%Y-%m-%d')
    read_end   = datetime.strptime(end_date,   '%Y-%m-%d')
    date_generated = [read_start + timedelta(days=x)
                      for x in range((read_end - read_start).days + 1)]
    fmt1_days = [d.strftime('%Y%m%d') for d in date_generated]
    fmt2_days = [d.strftime('%Y-%m-%d') for d in date_generated]

    filter_str = f'DQ{DQ_FLAG}Cloud{int(EFF_CLOUD*100):02d}'
    file_tag   = f'{fmt1_days[0]}T{fmt1_days[-1]}'

    out_gmt = os.path.join(
        output_dir,
        f'TEMPO_{varname}VCD_Timeseries_{site_abbr}_{aqs_id}_{file_tag}_{filter_str}_GMT.csv'
    )
    out_lt  = os.path.join(
        output_dir,
        f'TEMPO_{varname}VCD_Timeseries_{site_abbr}_{aqs_id}_{file_tag}_{filter_str}.csv'
    )

    if os.path.exists(out_lt):
        print(f'Output already exists: {out_lt}')
        return

    # Column names depend on species
    if varname == 'NO2':
        cols = ['TEMPO_trop_NO2', 'TEMPO_strat_NO2', 'TEMPO_total_NO2']
    else:  # HCHO
        cols = ['TEMPO_total_HCHO']

    all_dfs = []

    file_prefix = f'TEMPO_{varname}_L3_{tempo_version}_'

    for day_idx, gmt_date_fmt1 in enumerate(fmt1_days):
        matching = filter_filenames(vari_dir, file_prefix, gmt_date_fmt1)
        n_files  = len(matching)
        print(f'{fmt2_days[day_idx]}: {n_files} file(s)')

        timeseries = np.full((n_files, len(cols)), np.nan)
        gmt_times  = []

        for fi, fname in enumerate(matching):
            try:
                scan_ds = xr.open_dataset(vari_dir + fname)
                lats    = scan_ds.latitude.values
                lons    = scan_ds.longitude.values
                gmt_times.append(scan_ds.time.values[0])

                prod_ds   = xr.open_dataset(vari_dir + fname, group='product')
                supp_ds   = xr.open_dataset(vari_dir + fname, group='support_data')

                quality_mask = (
                    (prod_ds['main_data_quality_flag'] == DQ_FLAG) &
                    (supp_ds['eff_cloud_fraction'] < EFF_CLOUD)
                )

                lat_idx = find_nearest(lats, site_lat)
                lon_idx = find_nearest(lons, site_lon)

                if varname == 'NO2':
                    for col_ds in [prod_ds, supp_ds]:
                        for var in list(col_ds.data_vars):
                            col_ds[var] = col_ds[var].where(quality_mask, np.nan)

                    timeseries[fi, 0] = float(np.nanmean(
                        prod_ds['vertical_column_troposphere']
                        .isel(latitude=lat_idx, longitude=lon_idx).values
                    ))
                    timeseries[fi, 1] = float(np.nanmean(
                        prod_ds['vertical_column_stratosphere']
                        .isel(latitude=lat_idx, longitude=lon_idx).values
                    ))
                    timeseries[fi, 2] = float(np.nanmean(
                        supp_ds['vertical_column_total']
                        .isel(latitude=lat_idx, longitude=lon_idx).values
                    ))
                else:  # HCHO
                    prod_ds['vertical_column'] = prod_ds['vertical_column'].where(
                        quality_mask, np.nan
                    )
                    timeseries[fi, 0] = float(np.nanmean(
                        prod_ds['vertical_column']
                        .isel(latitude=lat_idx, longitude=lon_idx).values
                    ))

                scan_ds.close()
                prod_ds.close()
                supp_ds.close()

            except Exception as exc:
                gmt_times.append(np.nan)
                print(f'  Skipped {fname}: {exc}')

        day_df = pd.DataFrame(timeseries, columns=cols)
        day_df.insert(0, 'GMTdt', gmt_times)
        day_df['Unit'] = 'molecPcm2'
        all_dfs.append(day_df)

    if not all_dfs:
        print('No data extracted.')
        return

    result_df = pd.concat(all_dfs, ignore_index=True)

    # Format timestamps
    result_df['GMTdt'] = pd.to_datetime(result_df['GMTdt'], errors='coerce')
    result_df['GMTdt'] = result_df['GMTdt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    result_df.to_csv(out_gmt, index=False)
    print(f'Saved GMT CSV → {out_gmt}')

    # Add local time (drops rows with NaT)
    result_df = result_df.dropna(subset=['GMTdt']).reset_index(drop=True)
    result_df[['TimeZone', 'OffsetHours', 'LTDatetime', 'LTDT64Formatted']] = (
        result_df['GMTdt'].apply(
            lambda x: pd.Series(convert_to_local_time(x, site_lon, is_summer=True))
        )
    )
    result_df.to_csv(out_lt, index=False)
    print(f'Saved local-time CSV → {out_lt}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract TEMPO L3 VCDs at a ground-based monitor site.'
    )
    parser.add_argument('--var', choices=['NO2', 'HCHO', 'O3'], default='NO2',
                        help='Species to extract (default: NO2)')
    parser.add_argument('--tempo-version', choices=['V03', 'V04'], default='V03',
                        help='TEMPO product version (default: V03)')
    parser.add_argument('--tempo-dir', required=True,
                        help='Root directory containing TEMPO L3 sub-directories, '
                             'e.g. /data/TEMPO/BETA/ which holds TEMPO_NO2_L3_V03/ etc.')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write output CSV files')
    parser.add_argument('--start-date', default='2024-04-01',
                        help='Start date YYYY-MM-DD (default: 2024-04-01)')
    parser.add_argument('--end-date', default='2024-09-01',
                        help='End date YYYY-MM-DD (exclusive; default: 2024-09-01)')
    parser.add_argument('--site-abbr', required=True,
                        help='Short site identifier used in output filenames, e.g. CCNY')
    parser.add_argument('--site-lat', type=float, required=True,
                        help='Site latitude in decimal degrees')
    parser.add_argument('--site-lon', type=float, required=True,
                        help='Site longitude in decimal degrees (negative = West)')
    parser.add_argument('--aqs-id', default='',
                        help='EPA AQS monitor ID used in output filenames, e.g. 36-061-0135')
    args = parser.parse_args()

    extract_site(
        varname        = args.var,
        tempo_version  = args.tempo_version,
        tempo_dir      = args.tempo_dir,
        output_dir     = args.output_dir,
        start_date     = args.start_date,
        end_date       = args.end_date,
        site_abbr      = args.site_abbr,
        site_lat       = args.site_lat,
        site_lon       = args.site_lon,
        aqs_id         = args.aqs_id,
    )


if __name__ == '__main__':
    main()
