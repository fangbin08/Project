import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams["font.family"] = "serif"
import h5py
import calendar
import datetime
import glob
import pandas as pd
import rasterio
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#########################################################################################
# (Function 1) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

####################################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of source LTDR NDVI data
path_ltdr = '/Volumes/MyPassport/SMAP_Project/Datasets/LTDR/Ver5'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of source MODIS data
path_modis = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/HDF'
# Path of source output MODIS data
path_modis_op = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Output'
# Path of MODIS data for SM downscaling model input
path_modis_model_ip = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of SM model output
path_model_op = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP_ds/Model_Output'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Downscale'
# Path of 9 km SMAP SM
path_smap = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'
# Path of ISMN
path_ismn = '/Volumes/MyPassport/SMAP_Project/Datasets/ISMN/processed_data'
# Path of processed data
path_processed = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of Results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200810'

folder_400m = '/400m/'
folder_1km = '/1km/'
folder_9km = '/9km/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2019-12-31'
year = 2019 - 2015 + 1

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_doy = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(date_str.strftime('%Y%m%d'))
    date_seq_doy.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

# Count how many days for a specific year
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    if idt == 0:
        f_date = datetime.date(yearname[idt], monthnum[3], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)
    else:
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

daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=1)

ind_init = daysofmonth_seq_cumsum[2, :]
ind_end = daysofmonth_seq_cumsum[8, :] - 1
ind_gpm = np.stack((ind_init, ind_end), axis=1)
ind_gpm[0, :] = ind_gpm[0, :] - 90

# Extract the indices of the months between April - September
date_seq_month = np.array([int(date_seq[x][4:6]) for x in range(len(date_seq))])
monthnum_conus = monthnum[3:9]
date_seq_doy_conus_ind = np.where((date_seq_month >= 4) & (date_seq_month <= 9))[0]
date_seq_doy_conus = [date_seq_doy[date_seq_doy_conus_ind[x]] for x in range(len(date_seq_doy_conus_ind))]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min', 'cellsize_400m', 'cellsize_9km',
                'lat_conus_ease_1km', 'lon_conus_ease_1km', 'lat_conus_ease_9km', 'lon_conus_ease_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_conus_ease_400m', 'lon_conus_ease_400m',
                'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

########################################################################################################################
# 1. Read SMAP SM data in CONUS
# 1.1 Load the site lat/lon from excel files and Locate the SMAP 400m, 1/9 km SM positions by lat/lon of in-situ data

# Find the indices of the days between April - Sepetember
month_list = np.array([int(date_seq[x][4:6]) for x in range(len(date_seq))])
month_list_ind = np.where((month_list >= 4) & (month_list <= 9))[0]
month_list_ind = month_list_ind + 2 #First two columns are lat/lon

ismn_list = sorted(glob.glob(path_ismn + '/[A-Z]*.xlsx'))

coords_all = []
df_table_am_all = []
df_table_pm_all = []
for ife in range(14, len(ismn_list)):
    df_table_am = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='AM')
    df_table_pm = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='PM')

    netname = os.path.basename(ismn_list[ife]).split('_')[1]
    netname = [netname] * df_table_am.shape[0]
    coords = df_table_am[['lat', 'lon']]
    coords_all.append(coords)

    df_table_am_value = df_table_am.iloc[:, month_list_ind]
    df_table_am_value.insert(0, 'network', netname)
    df_table_pm_value = df_table_pm.iloc[:, month_list_ind]
    df_table_pm_value.insert(0, 'network', netname)
    df_table_am_all.append(df_table_am_value)
    df_table_pm_all.append(df_table_pm_value)
    del(df_table_am, df_table_pm, df_table_am_value, df_table_pm_value, coords, netname)
    print(ife)

df_coords = pd.concat(coords_all)
df_table_am_all = pd.concat(df_table_am_all)
df_table_pm_all = pd.concat(df_table_pm_all)


########################################################################################################################
# 1.2 Extract 400 m, 1 km / 9 km SMAP by lat/lon

# Locate the SM pixel positions
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

stn_row_400m_ind_all = []
stn_col_400m_ind_all = []
stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
stn_row_9km_ind_all = []
stn_col_9km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_400m_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_conus_ease_400m)).item()
    stn_col_400m_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_conus_ease_400m)).item()
    stn_row_400m_ind_all.append(stn_row_400m_ind)
    stn_col_400m_ind_all.append(stn_col_400m_ind)
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_conus_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_conus_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    stn_row_9km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_9km)).item()
    stn_col_9km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_9km)).item()
    stn_row_9km_ind_all.append(stn_row_9km_ind)
    stn_col_9km_ind_all.append(stn_col_9km_ind)
    del(stn_row_400m_ind, stn_col_400m_ind, stn_row_1km_ind, stn_col_1km_ind, stn_row_9km_ind, stn_col_9km_ind)


# 1.3 Extract 400 m SMAP SM (2019)
smap_400m_sta_all = []
tif_files_400m_name_ind_all = []
for iyr in [4]:  # range(yearname):

    os.chdir(path_smap + folder_400m + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Extract the file name
    tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
    tif_files_name_1year_ind = [date_seq_doy_conus.index(item) for item in tif_files_name if item in date_seq_doy_conus]
    date_seq_doy_conus_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy_conus]

    tif_files_400m_name_ind_all.append(tif_files_name_1year_ind)
    del(tif_files_name, tif_files_name_1year_ind)

    smap_400m_sta_1year = []
    for idt in range(len(date_seq_doy_conus_1year_ind)):
        src_tf = rasterio.open(tif_files[date_seq_doy_conus_1year_ind[idt]]).read()
        smap_400m_sta_1day = src_tf[:, stn_row_400m_ind_all, stn_col_400m_ind_all]
        smap_400m_sta_1year.append(smap_400m_sta_1day)
        del(src_tf, smap_400m_sta_1day)
        print(tif_files[date_seq_doy_conus_1year_ind[idt]])

    smap_400m_sta_all.append(smap_400m_sta_1year)
    del(smap_400m_sta_1year, date_seq_doy_conus_1year_ind)

tif_files_400m_name_ind_all = np.concatenate(tif_files_400m_name_ind_all)
smap_400m_sta_all = np.concatenate(smap_400m_sta_all)

# Fill the extracted SMAP SM into the proper position of days
smap_400m_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-1), dtype='float32')
smap_400m_sta_am[:] = np.nan
for idt in range(len(tif_files_400m_name_ind_all)):
    smap_400m_sta_am[:, tif_files_400m_name_ind_all[idt]] = smap_400m_sta_all[idt, 0, :]



# 1.4 Extract 1km SMAP SM (2019)
smap_1km_sta_all = []
tif_files_1km_name_ind_all = []
for iyr in [4]:  # range(yearname):

    os.chdir(path_smap + folder_1km + '/nldas/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Extract the file name
    tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
    tif_files_name_1year_ind  = [date_seq_doy_conus.index(item) for item in tif_files_name if item in date_seq_doy_conus]
    date_seq_doy_conus_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy_conus]

    tif_files_1km_name_ind_all.append(tif_files_name_1year_ind)
    del(tif_files_name, tif_files_name_1year_ind)

    smap_1km_sta_1year = []
    for idt in range(len(date_seq_doy_conus_1year_ind)):
        src_tf = rasterio.open(tif_files[date_seq_doy_conus_1year_ind[idt]]).read()
        smap_1km_sta_1day = src_tf[:, stn_row_1km_ind_all, stn_col_1km_ind_all]
        smap_1km_sta_1year.append(smap_1km_sta_1day)
        del(src_tf, smap_1km_sta_1day)
        print(tif_files[date_seq_doy_conus_1year_ind[idt]])

    smap_1km_sta_all.append(smap_1km_sta_1year)
    del(smap_1km_sta_1year, date_seq_doy_conus_1year_ind)


tif_files_1km_name_ind_all = np.concatenate(tif_files_1km_name_ind_all)
smap_1km_sta_all = np.concatenate(smap_1km_sta_all)

# Fill the extracted SMAP SM into the proper position of days
smap_1km_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-1), dtype='float32')
smap_1km_sta_am[:] = np.nan
for idt in range(len(tif_files_1km_name_ind_all)):
    smap_1km_sta_am[:, tif_files_1km_name_ind_all[idt]] = smap_1km_sta_all[idt, 0, :]


# 1.5 Extract 9km SMAP SM (2019)
smap_9km_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-1), dtype='float32')
smap_9km_sta_am[:] = np.nan

for iyr in [4]: #range(len(yearname)):

    smap_9km_sta_am_1year = []
    for imo in range(3, 9):#range(len(monthname)):

        smap_9km_sta_am_1month = []
        # Load in SMAP 9km SM data
        smap_file_path = path_smap + folder_9km + 'smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(smap_file_path) == True:

            f_smap_9km = h5py.File(smap_file_path, "r")
            varname_list_smap = list(f_smap_9km.keys())
            smap_9km_sta_am_1month = f_smap_9km[varname_list_smap[0]][()]
            smap_9km_sta_am_1month = smap_9km_sta_am_1month[stn_row_9km_ind_all, stn_col_9km_ind_all, :]

            print(smap_file_path)
            f_smap_9km.close()

        else:
            pass

        smap_9km_sta_am_1year.append(smap_9km_sta_am_1month)
        del(smap_9km_sta_am_1month)

    smap_9km_sta_am_1year = np.concatenate(smap_9km_sta_am_1year, axis=1)
    smap_9km_sta_am[:, iyr*183:(iyr+1)*183] = smap_9km_sta_am_1year
    del(smap_9km_sta_am_1year)


# Save variables
var_name_val = ['smap_400m_sta_am', 'smap_1km_sta_am', 'smap_9km_sta_am']
with h5py.File('/Users/binfang/Downloads/Processing/VIIRS/smap_validation_conus_viirs.hdf5', 'w') as f:
    for x in var_name_val:
        f.create_dataset(x, data=eval(x))
f.close()


########################################################################################################################
# 2.1 Subplots
# Site ID
# COSMOS: 0, 11, 25, 28, 34, 36, 42, 44
# SCAN: 250, 274, 279, 286, 296, 351, 362, 383
# SOILSCAPE: 860, 861, 870, 872, 896, 897, 904, 908
# USCRN: 918, 926, 961, 991, 1000，1002, 1012, 1016

# Load in the saved parameters
f_mat = h5py.File('/Users/binfang/Downloads/Processing/VIIRS/smap_validation_conus_viirs.hdf5', 'r')
varname_list = list(f_mat.keys())
for x in range(len(varname_list)):
    var_obj = f_mat[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f_mat.close()

# os.chdir(path_results + '/single')
ismn_sm_am = np.array(df_table_am_all.iloc[:, 1:])
ismn_sm_pm = np.array(df_table_pm_all.iloc[:, 1:])

# Make the plot
network_name = ['COSMOS', 'SCAN', 'SOILSCAPE', 'USCRN']
site_ind = [[0, 11, 25, 28, 34, 36, 42, 44], [250, 274, 279, 286, 296, 351, 362, 383],
            [860, 861, 870, 872, 896, 897, 904, 908], [918, 926, 961, 991, 1000, 1002, 1012, 1016]]

## len(df_table_am_all[df_table_am_all['network'] == 'USCRN'].index)
# site_num = np.array([0, 52, 9, 140, 9, 188, 404, 119, 113])
# site_num_sum = np.cumsum(site_num)
# site_ind_slc = [1, 5, 7, 8]
# site_ind = [np.arange(site_num_sum[site_ind_slc[x]-1], site_num_sum[site_ind_slc[x]]) for x in range(len(site_ind_slc))]



# Make the plots
stat_array_400m = []
stat_array_1km = []
stat_array_9km = []
ind_slc_all = []
for ist in range(len(site_ind)):

    x = ismn_sm_am[site_ind[ist], :].flatten()
    y1 = smap_400m_sta_am[site_ind[ist], :].flatten()
    y2 = smap_1km_sta_am[site_ind[ist], :].flatten()
    y3 = smap_9km_sta_am[site_ind[ist], :].flatten()
    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2) & ~np.isnan(y3))[0]

    if len(ind_nonnan) > 5:
        x = x[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]
        y3 = y3[ind_nonnan]

        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(x, y1)
        y1_estimated = intercept_1 + slope_1 * x
        number_1 = len(y1)
        r_sq_1 = r_value_1 ** 2
        ubrmse_1 = np.sqrt(np.mean((x - y1_estimated) ** 2))
        bias_1 = np.mean(x - y1)
        conf_int_1 = std_err_1 * 1.96  # From the Z-value
        stat_array_1 = [number_1, r_sq_1, ubrmse_1, bias_1, p_value_1, conf_int_1]

        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(x, y2)
        y2_estimated = intercept_2 + slope_2 * x
        number_2 = len(y2)
        r_sq_2 = r_value_2 ** 2
        ubrmse_2 = np.sqrt(np.mean((x - y2_estimated) ** 2))
        bias_2 = np.mean(x - y2)
        conf_int_2 = std_err_2 * 1.96  # From the Z-value
        stat_array_2 = [number_2, r_sq_2, ubrmse_2, bias_2, p_value_2, conf_int_2]

        slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = stats.linregress(x, y3)
        y3_estimated = intercept_3 + slope_3 * x
        number_3 = len(y3)
        r_sq_3 = r_value_3 ** 2
        ubrmse_3 = np.sqrt(np.mean((x - y3_estimated) ** 2))
        bias_3 = np.mean(x - y3)
        conf_int_3 = std_err_3 * 1.96  # From the Z-value
        stat_array_3 = [number_3, r_sq_3, ubrmse_3, bias_3, p_value_3, conf_int_3]

        if r_sq_1 >= 0.1:

            fig = plt.figure(figsize=(11, 6.5))
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            ax = fig.add_subplot(111)

            ax.scatter(x, y1, s=20, c='m', marker='s', label='400 m')
            ax.scatter(x, y2, s=20, c='b', marker='o', label='1 km')
            ax.scatter(x, y3, s=20, c='g', marker='^', label='9 km')
            ax.plot(x, intercept_1+slope_1*x, '-', color='m')
            ax.plot(x, intercept_2+slope_2*x, '-', color='b')
            ax.plot(x, intercept_3+slope_3*x, '-', color='g')

            plt.xlim(0, 0.4)
            ax.set_xticks(np.arange(0, 0.5, 0.1))
            plt.ylim(0, 0.4)
            ax.set_yticks(np.arange(0, 0.5, 0.1))
            ax.tick_params(axis='both', which='major', labelsize=13)
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            plt.grid(linestyle='--')
            plt.legend(loc='upper left', prop={'size': 13})
            plt.title(network_name[ist], fontsize=18, fontweight='bold')
            # plt.show()
            # plt.savefig(path_results + '/validation/' + df_table_am_all['network'][ist] + '_' + df_table_am_all.index[ist]
            #             + '_(' + str(ist) + ')' + '.png')
            plt.savefig(path_results + '/validation/' + network_name[ist] + '.png')
            plt.close(fig)
            stat_array_400m.append(stat_array_1)
            stat_array_1km.append(stat_array_2)
            stat_array_9km.append(stat_array_3)
            ind_slc_all.append(ist)
            print(ist)
            del(stat_array_1, stat_array_2, stat_array_3)

        else:
            pass

    else:
        pass


stat_array_400m = np.array(stat_array_400m)
stat_array_1km = np.array(stat_array_1km)
stat_array_9km = np.array(stat_array_9km)
# id = np.array(ind_slc_all)
# id = np.expand_dims(id, axis=1)

columns_validation = ['number', 'r_sq', 'ubrmse', 'bias', 'p_value', 'conf_int']
# index_validation = df_coords.index[ind_slc_all]
index_validation = ['COSMOS', 'SCAN', 'USCRN']

# stat_array_400m = np.concatenate((id, stat_array_400m), axis=1)
# stat_array_1km = np.concatenate((id, stat_array_1km), axis=1)
# stat_array_9km = np.concatenate((id, stat_array_9km), axis=1)
df_stat_400m = pd.DataFrame(stat_array_400m, columns=columns_validation, index=index_validation)
# df_stat_400m = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_400m], axis=1)
df_stat_1km = pd.DataFrame(stat_array_1km, columns=columns_validation, index=index_validation)
# df_stat_1km = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_1km], axis=1)
df_stat_9km = pd.DataFrame(stat_array_9km, columns=columns_validation, index=index_validation)
# df_stat_9km = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_9km], axis=1)
writer_400m = pd.ExcelWriter(path_results + '/stat_400m.xlsx')
writer_1km = pd.ExcelWriter(path_results + '/stat_1km.xlsx')
writer_9km = pd.ExcelWriter(path_results + '/stat_9km.xlsx')
df_stat_400m.to_excel(writer_400m)
df_stat_1km.to_excel(writer_1km)
df_stat_9km.to_excel(writer_9km)
writer_400m.save()
writer_1km.save()
writer_9km.save()

