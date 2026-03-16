"""
Extract TROPOMI L2 HiR trace-gas VCDs at a single ground-based monitor site.

For each day in the requested date range the script finds the TROPOMI L2 swath
whose scan-start time is closest to 1:30 PM EDT over the eastern US (≈ 17:30 UTC),
locates the nearest L2 pixel to the site, applies QA filtering, and appends the
result to a multi-year CSV file.

Supported species (--var): HCHO, NO2, CO
Data access: NASA GES DISC OPeNDAP (no local L2 download required)

Usage example
-------------
    python extract_tropomi_at_monitor.py \\
        --var NO2 \\
        --years 2018 2019 2020 2021 2022 2023 2024 \\
        --start-date 04-01 \\
        --end-date 09-01 \\
        --output-dir /data/TROPOMI_L2/CSVs/ \\
        --tropomi-test-file /data/TROPOMI_L2/S5P_RPRO_L2__NO2____....nc \\
        --site-abbr CCNY \\
        --site-lat 40.821 \\
        --site-lon -73.948 \\
        --aqs-id 36-061-0135

    export EARTHDATA_USERNAME=your_username
    export EARTHDATA_PASSWORD=your_password

MODIFICATION HISTORY
--------------------
    Original author: 2024-06-25, VERSION 1.0
    2024-11-25, VERSION 2.0 - Auto RPRO/OFFL selection
    Refactored: personal paths/credentials replaced with CLI arguments
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
import requests
from bs4 import BeautifulSoup
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
    'CO':   'S5P_L2__CO_____HiR.2',
}

FILE_HEADER_OPTIONS = [
    {'HCHO': 'S5P_RPRO_L2__HCHO___', 'NO2': 'S5P_RPRO_L2__NO2____', 'CO': 'S5P_RPRO_L2__CO_____'},
    {'HCHO': 'S5P_OFFL_L2__HCHO___', 'NO2': 'S5P_OFFL_L2__NO2____', 'CO': 'S5P_OFFL_L2__CO_____'},
]

FILL_VAL = 9.9692100e+36
UNIT_CONV = 6.02214e+19   # molecules cm⁻²  per  mol cm⁻²  (fallback if not in file)


# ---------------------------------------------------------------------------
# OPeNDAP helpers
# ---------------------------------------------------------------------------

def parse_opendap_directory(varname, url):
    """Return valid file links from an OPeNDAP catalogue page."""
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Failed to access {url}")
        return []
    soup = BeautifulSoup(resp.content, 'html.parser')
    for fh_dic in FILE_HEADER_OPTIONS:
        links = [a.get('href') for a in soup.find_all('a')
                 if a.get('href', '').startswith((fh_dic[varname], '0'))]
        if links:
            return links
    return []


def target_utc_from_day(given_day):
    """Return a datetime for 17:30 UTC on *given_day* (YYYYMMDD)."""
    return datetime.strptime(given_day + 'T173000', '%Y%m%dT%H%M%S')


def get_closest_file(files, given_day):
    """Return the OPeNDAP file URL whose scan-start time is closest to 17:30 UTC."""
    target = target_utc_from_day(given_day)

    def _scan_start(url):
        # filename contains YYYYMMDDTHHMMSS after the product header
        m = re.search(r'\d{8}T\d{6}', url)
        if m:
            try:
                return datetime.strptime(m.group(), '%Y%m%dT%H%M%S')
            except ValueError:
                pass
        return None

    best, best_diff = None, timedelta.max
    for f in files:
        dt = _scan_start(f)
        if dt and abs(dt - target) < best_diff:
            best_diff, best = abs(dt - target), f
    return best


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_site(varname, years, start_date, end_date,
                 output_dir, test_file,
                 site_abbr, site_lat, site_lon, aqs_id,
                 session, qa_threshold):

    os.makedirs(output_dir, exist_ok=True)

    # Read unit conversion factor from test file
    try:
        ref_ds = xr.open_dataset(test_file, group='/PRODUCT/', engine='netcdf4')
        if varname == 'HCHO':
            conv = float(ref_ds['formaldehyde_tropospheric_vertical_column']
                         .multiplication_factor_to_convert_to_molecules_percm2)
        elif varname == 'NO2':
            conv = float(ref_ds['nitrogendioxide_tropospheric_column']
                         .multiplication_factor_to_convert_to_molecules_percm2)
        else:
            conv = UNIT_CONV
        ref_ds.close()
    except Exception:
        conv = UNIT_CONV
    print(f'Unit conversion factor: {conv:.3e}')

    out_path = os.path.join(
        output_dir,
        f'TROPOMI_{varname}_Timeseries_{site_abbr}_{aqs_id}'
        f'_{years[0]}T{years[-1]}_{start_date.replace("-","")}T{end_date.replace("-","")}.csv'
    )
    if os.path.exists(out_path):
        print(f'Output already exists: {out_path}')
        return

    rows = []

    for year in years:
        start = datetime.strptime(f'{year}-{start_date}', '%Y-%m-%d')
        end   = datetime.strptime(f'{year}-{end_date}',   '%Y-%m-%d')
        days  = [(start + timedelta(days=x)).strftime('%Y%m%d')
                 for x in range((end - start).days + 1)]
        print(f'\n{year}: {len(days)} days')

        for day in days:
            doy      = get_day_of_year(day)
            base_url = (
                f'https://tropomi.gesdisc.eosdis.nasa.gov/opendap/'
                f'S5P_TROPOMI_Level2/{PRODUCT_INFO[varname]}/{year}/{doy}/'
            )
            links = parse_opendap_directory(varname, base_url)
            files = sorted(set(
                base_url + f[:-5] for f in links
                if f.endswith('.nc.html') and is_on_date(f, day)
            ))

            if len(files) < 4:
                continue

            closest = get_closest_file(files, day)
            if not closest:
                continue

            try:
                ds = open_url(closest, session=session)

                lat = np.array(ds['PRODUCT_latitude'][:])[0]
                lon = np.array(ds['PRODUCT_longitude'][:])[0]
                qa  = np.array(ds['PRODUCT_qa_value'][:])[0]

                if varname == 'HCHO':
                    vcd = np.array(ds['PRODUCT_formaldehyde_tropospheric_vertical_column'][:])[0]
                elif varname == 'NO2':
                    vcd = np.array(ds['PRODUCT_nitrogendioxide_tropospheric_column'][:])[0]
                else:
                    vcd = np.array(ds['PRODUCT_carbonmonoxide_total_column'][:])[0]

                vcd[vcd == FILL_VAL] = np.nan
                vcd = np.where(qa < qa_threshold, np.nan, vcd) * conv

                # Nearest pixel to site
                dist = np.sqrt((lat - site_lat) ** 2 + (lon - site_lon) ** 2)
                idx  = np.unravel_index(np.nanargmin(dist), dist.shape)

                rows.append({
                    'Date':      day,
                    'Latitude':  lat[idx],
                    'Longitude': lon[idx],
                    f'Trop{varname}Value': vcd[idx],
                })
                print(f'  {day}: {vcd[idx]:.3e} mol cm⁻²')

            except Exception as exc:
                print(f'  {day} skipped: {exc}')

    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f'\nSaved → {out_path}')
    else:
        print('No data extracted.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Extract TROPOMI L2 VCD time series at a ground monitor via OPeNDAP.'
    )
    parser.add_argument('--var', choices=['HCHO', 'NO2', 'CO'], default='NO2',
                        help='Species (default: NO2)')
    parser.add_argument('--years', nargs='+', default=['2024'],
                        help='Year(s) to process')
    parser.add_argument('--start-date', default='04-01',
                        help='Start date MM-DD within each year (default: 04-01)')
    parser.add_argument('--end-date', default='09-01',
                        help='End date MM-DD exclusive (default: 09-01)')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for output CSV file')
    parser.add_argument('--tropomi-test-file', required=True,
                        help='Local TROPOMI L2 NetCDF for reading the unit conversion factor')
    parser.add_argument('--site-abbr', required=True,
                        help='Short site label, e.g. CCNY')
    parser.add_argument('--site-lat', type=float, required=True,
                        help='Site latitude (decimal degrees)')
    parser.add_argument('--site-lon', type=float, required=True,
                        help='Site longitude (decimal degrees, negative = West)')
    parser.add_argument('--aqs-id', default='',
                        help='EPA AQS monitor ID (used in output filename)')
    parser.add_argument('--qa-threshold', type=float, default=0.75,
                        help='Minimum QA value to retain (default: 0.75)')
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

    check_url = (
        f'https://tropomi.gesdisc.eosdis.nasa.gov/opendap/'
        f'S5P_TROPOMI_Level2/{PRODUCT_INFO[args.var]}/'
    )
    session = setup_session(args.earthdata_username, args.earthdata_password,
                            check_url=check_url)

    extract_site(
        varname      = args.var,
        years        = args.years,
        start_date   = args.start_date,
        end_date     = args.end_date,
        output_dir   = args.output_dir,
        test_file    = args.tropomi_test_file,
        site_abbr    = args.site_abbr,
        site_lat     = args.site_lat,
        site_lon     = args.site_lon,
        aqs_id       = args.aqs_id,
        session      = session,
        qa_threshold = args.qa_threshold,
    )


if __name__ == '__main__':
    main()
