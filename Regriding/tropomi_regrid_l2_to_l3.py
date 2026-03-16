"""
Regrid TROPOMI L2 swath products (HCHO / NO2 / CO) to a regular 0.05° × 0.05° CONUS grid.

Data are accessed remotely via NASA GES DISC OPeNDAP; NASA Earthdata credentials are
required.  Pass them as CLI arguments or via environment variables
EARTHDATA_USERNAME and EARTHDATA_PASSWORD.

Input
-----
- TROPOMI L2 HiR products accessed via OPeNDAP (no local download needed):
    HCHO : S5P_L2__HCHO___HiR.2
    NO2  : S5P_L2__NO2____HiR.2
    CO   : S5P_L2__CO_____HiR.2
- One local test/reference NetCDF file for the chosen species (used to read
  layer count and copy variable attributes).

Output
------
- One NetCDF file per day in *output_dir*, containing:
    - Tropospheric (or total for CO) vertical column density and precision
    - Tropospheric, clear-sky, and cloudy-sky air mass factors (HCHO / NO2)

Usage example
-------------
    python tropomi_regrid_l2_to_l3.py \\
        --var HCHO \\
        --years 2022 2023 \\
        --start-date 04-01 \\
        --end-date 09-30 \\
        --output-dir /path/to/output/ \\
        --test-file /path/to/S5P_RPRO_L2__HCHO___....nc \\
        --qa-threshold 0.75

    # Credentials via environment variables (recommended):
    export EARTHDATA_USERNAME=your_username
    export EARTHDATA_PASSWORD=your_password

MODIFICATION HISTORY
--------------------
    Original author: 2023-03-29, VERSION 1.0
    2023-04-02, VERSION 2.0 - RPRO/OFFL auto-select; sub-day subdirectory handling
    2024-06-25, VERSION 3.0 - distance-masked nearest-neighbor regrid (0.1° cutoff)
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
from pydap.client import open_url
from pydap.cas.urs import setup_session

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Add repo root to path so satellite_utils is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.satellite_utils import (
    scipy_distantMasked_nearest_regrid_2dtoCartesian,
    get_day_of_year,
    is_on_date,
)

# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
PRODUCT_INFO = {
    'HCHO': 'S5P_L2__HCHO___HiR.2',
    'NO2':  'S5P_L2__NO2____HiR.2',
    'CO':   'S5P_L2__CO_____HiR.2',
}

FILE_HEADER_RPRO = {
    'HCHO': 'S5P_RPRO_L2__HCHO___',
    'NO2':  'S5P_RPRO_L2__NO2____',
    'CO':   'S5P_RPRO_L2__CO_____',
}

FILE_HEADER_OFFL = {
    'HCHO': 'S5P_OFFL_L2__HCHO___',
    'NO2':  'S5P_OFFL_L2__NO2____',
    'CO':   'S5P_OFFL_L2__CO_____',
}

# RPRO reprocessing availability cutoff (year < this → use RPRO)
RPRO_CUTOFF_YEAR = 2022

# CONUS bounding box for the output grid
CONUS_LON_LEFT, CONUS_LAT_BOT = -126.0, 23.0
CONUS_LON_RIGHT, CONUS_LAT_TOP = -66.0, 50.0
GRID_RES = 0.05  # degrees


# ---------------------------------------------------------------------------
# OPeNDAP helpers
# ---------------------------------------------------------------------------

def parse_directory(varname, url, fileheader_dic):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return [
            link.get('href') for link in soup.find_all('a')
            if link.get('href').startswith((fileheader_dic[varname], '0'))
        ]
    print(f"Failed to access {url}")
    return []


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_day(varname, given_day, session, fileheader_dic,
                final_lats, final_lons, output_dir, qa_threshold):
    """Download, quality-filter, and regrid all TROPOMI L2 swaths for *given_day*."""

    sel_year = given_day[:4]
    day_of_year = get_day_of_year(given_day)
    base_url = (
        f'https://tropomi.gesdisc.eosdis.nasa.gov/opendap/'
        f'S5P_TROPOMI_Level2/{PRODUCT_INFO[varname]}/{sel_year}/{day_of_year}/'
    )

    out_path = os.path.join(
        output_dir,
        fileheader_dic[varname] + f'CONUS_Regrid005deg_{given_day}.nc'
    )
    if os.path.exists(out_path):
        print(f"{given_day} already processed — skipping.")
        return

    print(f"Processing {given_day} …")
    subdirs = parse_directory(varname, base_url, fileheader_dic)

    try:
        _ = subdirs[3]  # sanity check: at least 4 files listed
    except (IndexError, TypeError):
        print(f"  No files found for {given_day}.")
        return

    files = sorted(set(
        base_url + f[:-5]  # strip '.html'
        for f in subdirs
        if f.endswith('.nc.html') and is_on_date(f, given_day)
    ))

    if not files:
        print(f"  No matching files on {given_day}.")
        return

    print(f"  {len(files)} file(s) found.")

    n = len(files)
    sh = (n, len(final_lats), len(final_lons))
    vcd_all   = np.full(sh, np.nan)
    prec_all  = np.full(sh, np.nan)
    tamf_all  = np.full(sh, np.nan)
    tamfp_all = np.full(sh, np.nan)
    camf_all  = np.full(sh, np.nan)
    clamf_all = np.full(sh, np.nan)

    FILL = 9.9692100e+36

    for i, filei in enumerate(files):
        try:
            ds = open_url(filei, session=session)

            lat = np.array(ds['PRODUCT_latitude'][:])[0, :, :]
            lon = np.array(ds['PRODUCT_longitude'][:])[0, :, :]

            if varname == 'HCHO':
                vcd  = np.array(ds['PRODUCT_formaldehyde_tropospheric_vertical_column'][:])[0]
                prec = np.array(ds['PRODUCT_formaldehyde_tropospheric_vertical_column_precision'][:])[0]
                qa   = np.array(ds['PRODUCT_qa_value'][:])[0]
                tamf = np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_formaldehyde_tropospheric_air_mass_factor'][:])[0]
                tamfp= np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_formaldehyde_tropospheric_air_mass_factor_precision'][:])[0]
                camf = np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_formaldehyde_clear_air_mass_factor'][:])[0]
                clamf= np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_formaldehyde_cloudy_air_mass_factor'][:])[0]
            elif varname == 'NO2':
                vcd  = np.array(ds['PRODUCT_nitrogendioxide_tropospheric_column'][:])[0]
                prec = np.array(ds['PRODUCT_nitrogendioxide_tropospheric_column_precision'][:])[0]
                qa   = np.array(ds['PRODUCT_qa_value'][:])[0]
                tamf = np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_air_mass_factor_troposphere'][:])[0]
                tamfp= np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_air_mass_factor_troposphere'][:])[0]  # reuse
                camf = np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_air_mass_factor_clear'][:])[0]
                clamf= np.array(ds['PRODUCT_SUPPORT_DATA_DETAILED_RESULTS_air_mass_factor_cloudy'][:])[0]
            else:  # CO
                vcd  = np.array(ds['PRODUCT_carbonmonoxide_total_column'][:])[0]
                prec = np.array(ds['PRODUCT_carbonmonoxide_total_column_precision'][:])[0]
                qa   = np.array(ds['PRODUCT_qa_value'][:])[0]
                # CO has no AMF in the same structure; fill with NaN
                tamf = tamfp = camf = clamf = np.full_like(vcd, np.nan)

            # Replace fill values
            for arr in [vcd, prec, tamf, tamfp, camf, clamf]:
                arr[arr == FILL] = np.nan

            # QA filter
            vcd  = np.where(qa < qa_threshold, np.nan, vcd)
            prec = np.where(qa < qa_threshold, np.nan, prec)

            # Valid pixel mask
            rav_lat = lat.ravel()
            rav_lon = lon.ravel()
            valid = ~np.isnan(rav_lat) & ~np.isnan(rav_lon)
            v_lat = rav_lat[valid]
            v_lon = rav_lon[valid]

            def _regrid(data_2d):
                return scipy_distantMasked_nearest_regrid_2dtoCartesian(
                    data_2d.ravel()[valid], v_lat, v_lon, final_lats, final_lons
                )

            vcd_all[i]   = _regrid(vcd)
            prec_all[i]  = _regrid(prec)
            tamf_all[i]  = _regrid(tamf)
            tamfp_all[i] = _regrid(tamfp)
            camf_all[i]  = _regrid(camf)
            clamf_all[i] = _regrid(clamf)

            print(f"    file {i+1}/{n} done.")

        except Exception as exc:
            print(f"    file {i+1}/{n} skipped: {exc}")

    # Daily mean across swaths
    date_dt = datetime.strptime(given_day, '%Y%m%d')
    coords = {'time': [date_dt], 'lat': final_lats, 'lon': final_lons}
    dims   = ('time', 'lat', 'lon')

    def _da(arr):
        return xr.DataArray(np.nanmean(arr, axis=0)[np.newaxis], dims=dims, coords=coords)

    if varname in ('HCHO', 'NO2'):
        vcd_name  = f'TROPOMI_{varname}_TropVCD'
        prec_name = f'TROPOMI_{varname}_TropVCDPrecision'
    else:
        vcd_name  = 'TROPOMI_CO_TotalVCD'
        prec_name = 'TROPOMI_CO_TotalVCDPrecision'

    ds_out = xr.Dataset({
        vcd_name:  _da(vcd_all),
        prec_name: _da(prec_all),
        'tropospheric_air_mass_factor':           _da(tamf_all),
        'tropospheric_air_mass_factor_precision': _da(tamfp_all),
        'clear_air_mass_factor':                  _da(camf_all),
        'cloudy_air_mass_factor':                 _da(clamf_all),
    })

    os.makedirs(output_dir, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Regrid TROPOMI L2 to 0.05° L3 CONUS grid via OPeNDAP.'
    )
    parser.add_argument('--var', choices=['HCHO', 'NO2', 'CO'], default='HCHO',
                        help='Species to process (default: HCHO)')
    parser.add_argument('--years', nargs='+', default=['2024'],
                        help='Year(s) to process, e.g. 2022 2023 2024')
    parser.add_argument('--start-date', default='04-01',
                        help='Start date within each year as MM-DD (default: 04-01)')
    parser.add_argument('--end-date', default='09-30',
                        help='End date within each year as MM-DD (default: 09-30)')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to write output NetCDF files')
    parser.add_argument('--test-file', required=True,
                        help='Path to a local reference TROPOMI L2 NetCDF file '
                             '(used to read layer count and copy attributes)')
    parser.add_argument('--qa-threshold', type=float, default=0.75,
                        help='Minimum QA value to retain (default: 0.75)')
    parser.add_argument('--earthdata-username',
                        default=os.environ.get('EARTHDATA_USERNAME'),
                        help='NASA Earthdata username '
                             '(or set EARTHDATA_USERNAME env var)')
    parser.add_argument('--earthdata-password',
                        default=os.environ.get('EARTHDATA_PASSWORD'),
                        help='NASA Earthdata password '
                             '(or set EARTHDATA_PASSWORD env var)')
    args = parser.parse_args()

    if not args.earthdata_username or not args.earthdata_password:
        parser.error(
            'NASA Earthdata credentials required. '
            'Pass --earthdata-username / --earthdata-password or '
            'set EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars.'
        )

    varname = args.var

    # Load reference file attributes
    test_ds = xr.open_dataset(args.test_file, group='/PRODUCT/', engine='netcdf4')
    test_ds.close()

    # Build target grid
    final_lats = np.arange(CONUS_LAT_BOT, CONUS_LAT_TOP, GRID_RES)
    final_lons = np.arange(CONUS_LON_LEFT, CONUS_LON_RIGHT, GRID_RES)

    # Set up OPeNDAP session (session is reused across all files)
    # The check_url just needs any valid OPeNDAP endpoint for authentication
    check_url = (
        f'https://tropomi.gesdisc.eosdis.nasa.gov/opendap/'
        f'S5P_TROPOMI_Level2/{PRODUCT_INFO[varname]}/'
    )
    session = setup_session(
        args.earthdata_username, args.earthdata_password, check_url=check_url
    )

    for sel_year in args.years:
        fileheader_dic = (
            FILE_HEADER_RPRO if int(sel_year) < RPRO_CUTOFF_YEAR else FILE_HEADER_OFFL
        )

        start = datetime.strptime(f'{sel_year}-{args.start_date}', '%Y-%m-%d')
        end   = datetime.strptime(f'{sel_year}-{args.end_date}',   '%Y-%m-%d')
        days  = [
            (start + timedelta(days=x)).strftime('%Y%m%d')
            for x in range((end - start).days + 1)
        ]
        print(f'\nYear {sel_year}: {len(days)} day(s) to process.')

        for day in days:
            try:
                process_day(
                    varname, day, session, fileheader_dic,
                    final_lats, final_lons, args.output_dir, args.qa_threshold
                )
            except Exception as exc:
                print(f"  Error on {day}: {exc}")


if __name__ == '__main__':
    main()
