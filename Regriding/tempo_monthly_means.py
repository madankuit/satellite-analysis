"""
Calculate per-UTC-half-hour monthly mean vertical column densities from TEMPO L3 V04 files.

For each (year, month, UTC-half-hour-bin) combination the script reads all
available daily scan files, applies quality and cloud-fraction filters, and
saves a monthly-mean NetCDF into a MonthlyMeans/ sub-directory next to the
input data.

Supported species (--var)
-------------------------
    NO2_tropVCD   - tropospheric NO2 vertical column
    NO2_totalVCD  - total NO2 vertical column
    NO2_stratVCD  - stratospheric NO2 vertical column
    HCHO_totalVCD - total HCHO vertical column

Usage example
-------------
    python tempo_monthly_means.py \\
        --var HCHO_totalVCD \\
        --years 2024 \\
        --months 04 05 06 07 08 09 \\
        --tempo-dir /path/to/TEMPO/V04/

MODIFICATION HISTORY
--------------------
    Original author: 2024-11-16, VERSION 1.0
    2024-11-21, VERSION 2.2 - Cloud filter (< 0.2); positive-values filter
    Refactored: personal paths replaced with CLI arguments
"""

import argparse
import glob
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Add repo root to path so satellite_utils is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.satellite_utils import group_by_half_hour_ranges_todic

# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
VAR_SUBDIR = {
    'NO2_tropVCD':   'TEMPO_NO2_L3_V04',
    'NO2_totalVCD':  'TEMPO_NO2_L3_V04',
    'NO2_stratVCD':  'TEMPO_NO2_L3_V04',
    'HCHO_totalVCD': 'TEMPO_HCHO_L3_V04',
}

VAR_COLNAME = {
    'NO2_tropVCD':   'vertical_column_troposphere',
    'NO2_totalVCD':  'vertical_column_total',
    'NO2_stratVCD':  'vertical_column_stratosphere',
    'HCHO_totalVCD': 'vertical_column',
}

DQ_FLAG    = 0    # Good quality only
EFF_CLOUD  = 0.2  # Effective cloud fraction threshold


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_month(varname, sel_year, monthi, tempo_dir):
    """
    Compute and save per-UTC-half-hour monthly mean VCD for *varname*.

    Parameters
    ----------
    varname   : str, one of VAR_SUBDIR keys
    sel_year  : str, e.g. '2024'
    monthi    : str, zero-padded month, e.g. '04'
    tempo_dir : str, path to the TEMPO V04 root directory
    """
    var_subdir   = VAR_SUBDIR[varname]
    col_name     = VAR_COLNAME[varname]
    vari_dir     = os.path.join(tempo_dir, var_subdir)
    monthly_dir  = os.path.join(vari_dir, 'MonthlyMeans')
    os.makedirs(monthly_dir, exist_ok=True)

    # Collect all files for this month
    start = datetime.strptime(f'{sel_year}-{monthi}-01', '%Y-%m-%d')
    # Handle December → next year January
    next_month = int(monthi) % 12 + 1
    next_year  = int(sel_year) + (1 if int(monthi) == 12 else 0)
    end   = datetime.strptime(f'{next_year}-{next_month:02d}-01', '%Y-%m-%d')
    days  = [(start + timedelta(days=x)).strftime('%Y%m%d')
             for x in range((end - start).days)]

    file_paths = []
    for day in days:
        file_paths.extend(sorted(glob.glob(os.path.join(vari_dir, f'*{day}*'))))

    if not file_paths:
        print(f'  No files found for {sel_year}-{monthi}.')
        return

    # Read sample file for coordinates and attributes
    sample_ds     = xr.open_dataset(file_paths[0], group='product')
    geo_sample_ds = xr.open_dataset(file_paths[0])

    # Group by UTC half-hour bin
    hour_dic = group_by_half_hour_ranges_todic(file_paths)

    for utc_hour, hour_files in hour_dic.items():
        out_path = os.path.join(
            monthly_dir,
            f'{var_subdir}_{sel_year}{monthi}_{utc_hour}_{len(hour_files)}Files.nc'
        )

        if os.path.exists(out_path):
            print(f'  {sel_year}-{monthi} | {utc_hour} already done — skipping.')
            continue

        print(f'  {sel_year}-{monthi} | {utc_hour}: {len(hour_files)} file(s)')

        lat_vals = geo_sample_ds.latitude.values
        lon_vals = geo_sample_ds.longitude.values
        monthly_mean = np.full((1, len(lat_vals), len(lon_vals)), np.nan)

        for file_path in hour_files:
            try:
                fi_ds      = xr.open_dataset(file_path, group='product')
                fi_supp_ds = xr.open_dataset(file_path, group='support_data')

                # Quality filter
                fi_ds[col_name] = fi_ds[col_name].where(
                    fi_ds['main_data_quality_flag'] == DQ_FLAG, np.nan
                )
                # Cloud filter
                fi_ds[col_name] = fi_ds[col_name].where(
                    fi_supp_ds['eff_cloud_fraction'] < EFF_CLOUD, np.nan
                )

                filei_vals = fi_ds[col_name].values
                combined   = np.stack([monthly_mean, filei_vals], axis=0)
                monthly_mean = np.nanmean(combined, axis=0)

                fi_ds.close()
                fi_supp_ds.close()

            except Exception as exc:
                print(f'    Skipped {os.path.basename(file_path)}: {exc}')

        # Build output dataset
        vcd_da = xr.DataArray(
            monthly_mean,
            dims=('time', 'latitude', 'longitude'),
            coords={
                'time':      [f'{sel_year}-{monthi}-{utc_hour}'],
                'latitude':  lat_vals,
                'longitude': lon_vals,
            },
        )

        ds_out = xr.Dataset({col_name: vcd_da})

        # Copy variable attributes from sample
        for attr, val in sample_ds[col_name].attrs.items():
            ds_out[col_name].attrs[attr] = val
        for attr, val in geo_sample_ds['latitude'].attrs.items():
            ds_out['latitude'].attrs[attr] = val
        for attr, val in geo_sample_ds['longitude'].attrs.items():
            ds_out['longitude'].attrs[attr] = val

        ds_out.to_netcdf(out_path)
        print(f'    Saved → {out_path}')

    sample_ds.close()
    geo_sample_ds.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compute per-UTC-half-hour monthly means from TEMPO L3 V04 files.'
    )
    parser.add_argument('--var', choices=list(VAR_SUBDIR.keys()), default='HCHO_totalVCD',
                        help='Variable to process (default: HCHO_totalVCD)')
    parser.add_argument('--years', nargs='+', default=['2024'],
                        help='Year(s) to process, e.g. 2024')
    parser.add_argument('--months', nargs='+',
                        default=['04', '05', '06', '07', '08', '09'],
                        help='Month(s) to process as zero-padded strings '
                             '(default: 04 05 06 07 08 09)')
    parser.add_argument('--tempo-dir', required=True,
                        help='Root directory of TEMPO V04 data, '
                             'expected sub-structure: <tempo-dir>/TEMPO_HCHO_L3_V04/...')
    args = parser.parse_args()

    for year in args.years:
        for month in args.months:
            print(f'\nProcessing {year}-{month} | {args.var}')
            try:
                process_month(args.var, year, month, args.tempo_dir)
            except Exception as exc:
                print(f'  Error for {year}-{month}: {exc}')


if __name__ == '__main__':
    main()
