import os
import osr
import glob
import gdal
import numpy as np
import matplotlib.pyplot as plt
import datetime
import h5py
import calendar
import gzip
from scipy.interpolate import interp1d
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data/nldas'
# Path of source MODIS data
path_modis = '/Volumes/SBac/data/MODIS/HDF_Data'
# Path of MODIS data for SM downscaling model input
path_modis_ip = '/Volumes/SBac/data/MODIS/Model_Input'
# Path of source output MODIS data
path_modis_op = '/Users/binfang/Downloads/Processing/Model_Output'
# Path of downscaled SM
path_smap_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'
# Path of 9 km SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'
# Path of VIIRS data
path_viirs = '/Volumes/MyPassport/SMAP_Project/Datasets/VIIRS'
# Path of VIIRS data regrid output
path_viirs_output = '/Users/binfang/Downloads/Processing/VIIRS/output'

# lst_folder = '/MYD11A1/'
# ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']
size_viirs_tile = 3750

# # Interdistance of EASE grid projection grids
# interdist_ease_1km = 1000.89502334956
# interdist_ease_9km = 9009.093602916

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min',
                'lat_conus_ease_400m', 'lon_conus_ease_400m', 'lat_conus_ease_9km', 'lon_conus_ease_9km',
                'lat_conus_ease_12_5km', 'lon_conus_ease_12_5km',
                'row_conus_ease_400m_ind', 'col_conus_ease_400m_ind', 'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind',
                'row_conus_ease_400m_from_geo_400m_ind', 'col_conus_ease_400m_from_geo_400m_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2015-01-01'
end_date = '2019-12-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

# Count how many days for a specific year
yearname = np.linspace(2015, 2020, 6, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    # if idt == 0:
    #     f_date = datetime.date(yearname[idt], monthnum[3], 1)
    #     l_date = datetime.date(yearname[idt], monthnum[-1], 31)
    #     delta_1y = l_date - f_date
    #     daysofyear.append(delta_1y.days + 1)
    # else:
        f_date = datetime.date(yearname[idt], monthnum[0], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)

daysofyear = np.asarray(daysofyear)

# Find the indices of each month in the list of days between 2015 - 2018
nlpyear = 1999 # non-leap year
lpyear = 2000 # leap year
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x+1].sum()) for x in range(0, len(monthnum))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x+1].sum()) for x in range(0, len(monthnum))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))]) # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthnum))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr # Add leap days to February

########################################################################################################################
# VIIRS data description:
# Dx=3750
# Dy=3750
# dlat/dlon = 0.04 degrees
# Lower left lat / lon
# T028: Lat = 45.002, -134.998
# T029: Lat = 45.002, -119.998
# T030: Lat = 45.002, -104.998
# T031: Lat = 45.002, -89.998
# T032: Lat = 45.002, -74.998
# % T052: Lat = 30.002, -134.998
# % T053: Lat = 30.002, -119.998
# % T054: Lat = 30.002, -104.998
# % T055: Lat = 30.002, -89.998
# % T056: Lat = 30.002, -74.998
# % T077: Lat = 15.002, -119.998
# % T078: Lat = 15.002, -104.998
# % T079: Lat = 15.002, -89.998
# % T080: Lat = 15.002, -74.998

# 1. Process viirs data of 2019
start_day = np.cumsum(daysofmonth_seq[0:3, 5])[-1] + 1
end_day = np.cumsum(daysofmonth_seq[0:9, 5])[-1]
datename_seq_2019 = np.arange(start_day, end_day+1)

subfolders = sorted(glob.glob(path_viirs + '/2019/*/'))

viirs_mat_initial = np.empty([size_viirs_tile, size_viirs_tile], dtype='float32')
viirs_mat_initial[:] = np.nan
for ifo in range(len(subfolders)-1): # Exclude the LAI data folder
    file_lstd = sorted(glob.glob(subfolders[ifo] + '/*DAY*'))
    file_lstn = sorted(glob.glob(subfolders[ifo] + '/*NIGHT*'))

    # Find the matched files for each day between March - September
    file_lstd_datename = np.array([int(file_lstd[x].split('/')[-1].split('.')[0].split('_')[-1][-3:])
                                   for x in range(len(file_lstd))])
    file_lstn_datename = np.array([int(file_lstn[x].split('/')[-1].split('.')[0].split('_')[-1][-3:])
                                   for x in range(len(file_lstn))])
    file_lstd_ind = [np.where(file_lstd_datename == datename_seq_2019[x])[0] for x in range(len(datename_seq_2019))]
    file_lstn_ind = [np.where(file_lstn_datename == datename_seq_2019[x])[0] for x in range(len(datename_seq_2019))]

    # Extract the day/night viirs data of one tile
    # Day
    viirs_mat_day_all = []
    for idt in range(len(datename_seq_2019)):
        if len(file_lstd_ind[idt]) != 0:
            with gzip.open(file_lstd[file_lstd_ind[idt][0]], 'rb') as file:
                viirs_mat_day = np.frombuffer(file.read(), dtype='f')
                viirs_mat_day = np.reshape(viirs_mat_day, (size_viirs_tile, size_viirs_tile)).copy()
                viirs_mat_day[viirs_mat_day <= 0] = np.nan
            file.close()
            viirs_mat_day_all.append(viirs_mat_day)
            print(file_lstd[file_lstd_ind[idt][0]])
            del(viirs_mat_day)
        else:
            viirs_mat_day_all.append(viirs_mat_initial)

    viirs_mat_day_all = np.array(viirs_mat_day_all)

    # Night
    viirs_mat_night_all = []
    for idt in range(len(datename_seq_2019)):
        if len(file_lstn_ind[idt]) != 0:
            with gzip.open(file_lstn[file_lstn_ind[idt][0]], 'rb') as file:
                viirs_mat_night = np.frombuffer(file.read(), dtype='f')
                viirs_mat_night = np.reshape(viirs_mat_night, (size_viirs_tile, size_viirs_tile)).copy()
                viirs_mat_night[viirs_mat_night <= 0] = np.nan
            file.close()
            viirs_mat_night_all.append(viirs_mat_night)
            print(file_lstn[file_lstn_ind[idt][0]])
            del(viirs_mat_night)
        else:
            viirs_mat_night_all.append(viirs_mat_initial)

    viirs_mat_night_all = np.array(viirs_mat_night_all)

    # # Calculate the daily LST difference
    # viirs_mat_lst_delta = np.absolute(np.subtract(viirs_mat_day_all, viirs_mat_night_all))
    # viirs_mat_lst_delta[viirs_mat_lst_delta == 0] = np.nan

    # write the variable to hdf file
    tile_name = subfolders[ifo].split('/')[-2]
    var_name = ['viirs_lstd_' + tile_name + '_2019', 'viirs_lstn_' + tile_name + '_2019']
    # data_name = ['viirs_mat_day_all', 'viirs_mat_night_all']

    for idt in range(len(datename_seq_2019)):
        # Day
        with h5py.File(path_viirs_output + '/tile_output/2019/' + var_name[0] + str(datename_seq_2019[idt]).zfill(3) + '.hdf5', 'w') as f:
            f.create_dataset(var_name[0], data=viirs_mat_day_all[idt, :, :])
        f.close()
        # Night
        with h5py.File(path_viirs_output + '/tile_output/2019/' + var_name[1] + str(datename_seq_2019[idt]).zfill(3) + '.hdf5', 'w') as f:
            f.create_dataset(var_name[1], data=viirs_mat_night_all[idt, :, :])
        f.close()

    del(file_lstd, file_lstn, file_lstd_datename, file_lstn_datename, file_lstd_ind, file_lstn_ind,
        viirs_mat_day_all, viirs_mat_night_all, tile_name, var_name)


