"""
Shared utility functions for satellite data processing (TROPOMI, TEMPO).

Functions:
    find_nearest             - nearest index in array
    get_day_of_year          - convert YYYYMMDD to 3-digit day-of-year string
    filter_filenames         - list files matching two substrings
    find_filename_with_given_day - find file containing a date string
    is_within_date_range     - check if filename date is in range
    is_on_date               - check if filename matches a target date
    mjd2k_to_datetime_array  - convert MJD2K array to datetime array
    mjd2k_to_edt_array       - convert MJD2K array to EDT datetime array
    convert_to_local_time    - convert UTC datetime string to local time
    group_by_half_hour_ranges_todic - group TEMPO filenames by UTC half-hour bins
    scipy_distantMasked_nearest_regrid_2dtoCartesian - nearest-neighbor regrid with distance masking
"""

import os
import re
import glob
import collections
import numpy as np
import scipy.interpolate
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
import pytz


# ---------------------------------------------------------------------------
# Array / coordinate helpers
# ---------------------------------------------------------------------------

def find_nearest(array, value):
    """Return the index of the nearest value found in an array."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def get_day_of_year(date_str):
    """Return a zero-padded 3-digit day-of-year string for *date_str* (YYYYMMDD)."""
    date_object = datetime.strptime(date_str, '%Y%m%d')
    day_of_year = date_object.timetuple().tm_yday
    return f"{day_of_year:03d}"


# ---------------------------------------------------------------------------
# File selection helpers
# ---------------------------------------------------------------------------

def filter_filenames(directory, filestr, datei):
    """
    Return sorted list of filenames in *directory* that contain both
    *filestr* and *datei*.
    """
    filenames = [
        f for f in os.listdir(directory)
        if filestr in f and datei in f
    ]
    return sorted(filenames)


def find_filename_with_given_day(directory, given_day):
    """
    Search *directory* for a file whose name contains *given_day* (YYYYMMDD).

    Returns the filename string if found, otherwise None.
    """
    for f in os.listdir(directory):
        if given_day in f and os.path.exists(os.path.join(directory, f)):
            return f
    return None


def is_within_date_range(filename, startdate, enddate):
    """Return True if the 8-digit date in *filename* falls in [startdate, enddate]."""
    match = re.search(r'\d{8}', filename)
    if match:
        return startdate <= match.group(0) <= enddate
    return False


def is_on_date(filename, target_date):
    """Return True if the 8-digit date in *filename* equals *target_date*."""
    match = re.search(r'\d{8}', filename)
    if match:
        return match.group(0) == target_date
    return False


# ---------------------------------------------------------------------------
# Time conversion
# ---------------------------------------------------------------------------

def mjd2k_to_datetime_array(mjd2k_array):
    """Convert an array of MJD2000 values to UTC datetime objects."""
    mjd2k_start = datetime(2000, 1, 1)
    return np.array([mjd2k_start + timedelta(days=v) for v in mjd2k_array])


def mjd2k_to_edt_array(mjd2k_array):
    """Convert an array of MJD2000 values to EDT datetime objects (UTC-4)."""
    mjd2k_start = datetime(2000, 1, 1)
    edt_offset = timedelta(hours=4)
    return np.array([mjd2k_start + timedelta(days=v) - edt_offset for v in mjd2k_array])


def convert_to_local_time(ut_time_str, lon, is_summer=False):
    """
    Convert a UTC datetime string to approximate local time based on longitude.

    Parameters
    ----------
    ut_time_str : str
        UTC datetime in format '%Y-%m-%d %H:%M:%S'.
    lon : float
        Longitude (degrees). Used to estimate UTC offset.
    is_summer : bool
        If True, add 1 hour for daylight saving time.

    Returns
    -------
    timezone_name : str
    offset_hours  : int
    local_time    : datetime
    dt64_formatted: numpy.datetime64
    """
    ut_time = datetime.strptime(ut_time_str, '%Y-%m-%d %H:%M:%S')
    ut_time = ut_time.replace(tzinfo=pytz.utc)
    offset_hours = round(lon / 15)
    if is_summer:
        offset_hours += 1
    local_time = ut_time + timedelta(hours=offset_hours)
    timezone_name = f'UTC{offset_hours:+d}'
    dt64_formatted = np.datetime64(local_time.strftime('%Y-%m-%d %H:%M'))
    return timezone_name, offset_hours, local_time, dt64_formatted


# ---------------------------------------------------------------------------
# TEMPO file grouping
# ---------------------------------------------------------------------------

def group_by_half_hour_ranges_todic(filenamePaths):
    """
    Group TEMPO file paths by the nearest UTC half-hour bin.

    Keys are strings like 'UTC0930', 'UTC1430', meaning files whose
    timestamp is closest to that half-hour mark.

    Parameters
    ----------
    filenamePaths : list of str

    Returns
    -------
    dict : {UTC_half_hour_key: [file_paths]}
    """
    grouped_files = collections.defaultdict(list)
    for file in filenamePaths:
        try:
            timestamp = file.split("_")[-2]
            date_part, time_part = timestamp.split('T')
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            range_start_hour = (hour + 1) % 24 if minute >= 30 else hour
            range_key = f"UTC{range_start_hour:02}30"
            grouped_files[range_key].append(file)
        except Exception as e:
            print(f"Skipping file with unexpected format: {file}. Error: {e}")
    return dict(grouped_files)


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def scipy_distantMasked_nearest_regrid_2dtoCartesian(
    Origi_data, Origi2d_lat, Origi2d_lon,
    target1d_lat, target1d_lon,
    max_distance=0.1,
):
    """
    Nearest-neighbor regrid from an irregular 2-D swath to a regular lat/lon grid,
    masking target cells whose nearest swath pixel is farther than *max_distance* degrees.

    Parameters
    ----------
    Origi_data   : 1-D or 2-D array  (will be ravelled)
    Origi2d_lat  : 1-D array of swath latitudes (same length as ravelled Origi_data)
    Origi2d_lon  : 1-D array of swath longitudes
    target1d_lat : 1-D array, target latitude vector
    target1d_lon : 1-D array, target longitude vector
    max_distance : float, degrees; cells farther than this are set to NaN

    Returns
    -------
    grid_z0 : 2-D array (len(target1d_lat), len(target1d_lon))
    """
    lon2d, lat2d = np.meshgrid(target1d_lon, target1d_lat)

    Origi_data = np.asarray(Origi_data).ravel()
    Origi2d_lat = np.asarray(Origi2d_lat).ravel()
    Origi2d_lon = np.asarray(Origi2d_lon).ravel()

    mask = np.isnan(Origi_data)
    points = np.column_stack((Origi2d_lon, Origi2d_lat))
    valid = ~mask
    points = points[valid]
    values = Origi_data[valid]

    grid_z0 = scipy.interpolate.griddata(
        points, values, (lon2d, lat2d), method='nearest'
    )

    tree = cKDTree(points)
    distances, _ = tree.query(
        np.column_stack((lon2d.ravel(), lat2d.ravel())), k=1
    )
    distances = distances.reshape(lon2d.shape)
    grid_z0[distances > max_distance] = np.nan

    return grid_z0
