import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import fiona
import rasterio
import rasterio.mask
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import h5py
import calendar
import datetime
import glob
import pandas as pd
import matplotlib.pyplot as plt
# import xlrd
import gdal
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.crs import CRS
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats

# Ignore runtime warning
import warnings
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
path_model = '/Users/binfang/Downloads/Processing/model_data'
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
# Path of processed data
path_processed = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
#Path of SMAP Mat-files
path_matfile = '/Volumes/MyPassport/SMAP_Project/Datasets/CONUS'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '1979-12-31'
end_date = '2018-12-31'
year = 2018 - 1981 + 1

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


# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


####################################################################################################################################
# 1.1 Check the completeness of downloaded GLDAS files
# os.chdir(path_ltdr)
os.chdir('/Users/binfang/Downloads/Processing/bash_codes/download')
files = sorted(glob.glob('*.nc4'))
# date_seq_late = date_seq[6939:]
files_group = []
for idt in range(13879):
    # files_group_1day = [files.index(i) for i in files if 'A' + date_seq_late[idt] in i]
    files_group_1day = [files.index(i) for i in files if 'A' + date_seq[idt] in i]
    files_group.append(files_group_1day)
    print(idt)

ldasfile_miss = []
for idt in range(len(files_group)):
    if len(files_group[idt]) != 8:
        ldasfile_miss.append(date_seq[idt])
        print(date_seq[idt])
        # file_miss.append(date_seq_late[idt])
        # print(date_seq_late[idt])
        print(len(files_group[idt]))
    else:
        pass

########################################################################################################################
# 1.2 Check the completeness of downloaded NLDAS files
# Generate 24-hour sequence names of each day
hours_seq = np.arange(24)*100
hours_seq = [str(hours_seq[x]).zfill(4) for x in range(len(hours_seq))]
date_seq_24h = ['A' + date_seq[x] + '.' + hours_seq[y] for x in range(len(date_seq)) for y in range(len(hours_seq))]

# List of files to download
os.chdir('/Users/binfang/Downloads/Processing/bash_codes')
with open("nldas_data.txt", "r") as ins:
    url_list = []
    for line in ins:
        url_list.append(line)
ins.close()

year_tocheck = '2018'
date_seq_24h_ind = [url_list.index(i) for i in url_list if 'A' + year_tocheck in i]
url_list_1year = [url_list[date_seq_24h_ind[i]] for i in range(len(date_seq_24h_ind))]

os.chdir('/Volumes/MyPassport/SMAP_Project/Datasets/NLDAS/' + year_tocheck)
download_files = sorted(glob.glob('*.nc4'))
download_files_list = [download_files[x].split('.')[1] + '.' + download_files[x].split('.')[2] for x in range(len(download_files))]

url_files_list = [url_list_1year[x].split('.')[7] + '.' + url_list_1year[x].split('.')[8] for x in range(len(url_list_1year))]

len(download_files_list) == len(url_files_list)

download_file_miss_ind = [url_files_list.index(i) for i in url_files_list if i not in download_files_list]
download_file_miss_url = [url_list_1year[download_file_miss_ind[x]] for x in range(len(download_file_miss_ind))]

os.chdir('/Users/binfang/Downloads/Processing/bash_codes/download')
file = open(year_tocheck + '_miss.txt', 'w')
with open(year_tocheck + '_miss.txt', 'w') as f:
    for item in download_file_miss_url:
        f.write("%s" % item)
f.close()

####################################################################################################################################
# 2. Check the completeness of downloaded MODIS files
# os.chdir(path_modis + lst_folder + '2019/05/')
os.chdir('/Volumes/MyPassport/SMAP_Project/NewData/SMAP/2019')
# Downloaded files
files = sorted(glob.glob('*.h5'))

# List of files to download
os.chdir('/Users/binfang/Downloads/bash_codes/')
with open("3365467267-download.txt", "r") as ins:
    url_list = []
    for line in ins:
        url_list.append(line)
ins.close()

len(files) == len(url_list)

files_list = [url_list[x][:-1].split('/')[-1] for x in range(len(url_list))]
modfile_miss_ind = [files_list.index(i) for i in files_list if i not in files]
modfile_miss_url = [url_list[modfile_miss_ind[x]] for x in range(len(modfile_miss_ind))]

os.chdir('/Users/binfang/Downloads/bash_codes/missed')
file = open('smap_2019_miss.txt', 'w')
with open('smap_2019_miss.txt', 'w') as f:
    for item in modfile_miss_url:
        f.write("%s" % item)
f.close()


########################################################################################################################
#3. Create a map of dams in the three states

dam_csv = pd.read_csv('/Users/binfang/Documents/20_Spring/documents/nsf_proposal/3st_dams_stats.csv')

sca = dam_csv['Storage ca'].values
sca_ind = [0, 0.13, 0.2, 0.3, 0.6, np.inf]
sca_group = [len(sca[np.where((sca > sca_ind[x]) & (sca < sca_ind[x+1]))]) for x in range(len(sca_ind)-1)]

ms = dam_csv['Main use'].values
ms_unique = list(np.unique(ms))
ms_group = [len(np.where(ms == ms_unique[x])[0]) for x in range(len(ms_unique))]

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels_1 = ['0.1-0.13', '0.13-0.2', '0.2-0.3', '0.3-0.6', '>0.6']
labels_2 = [str(ms_group[x]) + '\n' + ms_unique[x] for x in range(len(ms_group))]

fig = plt.figure(figsize=(14, 7), facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Storage Capacity $\mathregular{(km^3)}$', pad=20, fontsize=25, fontweight='bold')
ax.pie(sca_group, labels=labels_1, autopct='%1.1f%%', labeldistance=1,
        shadow=True, startangle=90, textprops={'fontsize': 19})
ax.axis('equal')

ax = fig.add_subplot(1, 2, 2)
ax.set_title('Main Use', pad=20, fontsize=25, fontweight='bold')
ax.pie(ms_group, labels=labels_2, autopct='%1.1f%%', labeldistance=1,
        shadow=True, startangle=90, textprops={'fontsize': 19})
ax.axis('equal')
plt.subplots_adjust(left=0.05, right=0.9, bottom=0.05, top=0.85, hspace=0.35, wspace=0.35)

plt.savefig('/Users/binfang/Downloads/piechart.tif')
plt.close()



########################################################################################################################
# 4. Subset the 1 km downscaled SMAP SM data

# Bounding box of Peru
# (-18.3479753557, -81.4109425524, -0.0572054988649, -68.6650797187)
lat_peru_max = -0.05
lat_peru_min = -18.35
lon_peru_max = -68.66
lon_peru_min = -81.42
output_crs = 'EPSG:4326'
[lat_peru_ease_1km, row_peru_ease_1km_ind, lon_peru_ease_1km, col_peru_ease_1km_ind] = coordtable_subset\
    (lat_world_ease_1km, lon_world_ease_1km, lat_peru_max, lat_peru_min, lon_peru_max, lon_peru_min)

for iyr in range(len(yearname)):
    smap_file_path = path_smap_sm_ds + '/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src_tf = rasterio.open(smap_file_list[idt])

        # Subset the SM data by bounding box
        sub_window_1km = Window(col_peru_ease_1km_ind[0], row_peru_ease_1km_ind[0],
                                len(col_peru_ease_1km_ind), len(row_peru_ease_1km_ind))
        kwargs_1km_sub = src_tf.meta.copy()
        kwargs_1km_sub.update({
            'height': sub_window_1km.height,
            'width': sub_window_1km.width,
            'transform': rasterio.windows.transform(sub_window_1km, src_tf.transform)})

        # Write to Geotiff file
        with rasterio.open('/Users/binfang/Downloads/Processing/smap_sm_peru/' + str(yearname[iyr]) +
                           '/' + os.path.basename(smap_file_list[idt]), 'w', **kwargs_1km_sub) as output_ds:
            output_ds.write(src_tf.read(window=sub_window_1km))

        print(os.path.basename(smap_file_list[idt]))



os.chdir('/Users/binfang/Downloads')

ds = open('_Clim_Pred_LRF_New_GridDataDownload_Rainfall_ind2019_rfp25.grd', 'r')
array = np.fromfile(ds, dtype=np.dtype('f8'))
ds1[ds1<-1] = np.nan

array = np.fromfile('_Clim_Pred_LRF_New_GridDataDownload_Rainfall_ind2019_rfp25.grd')




########################################################################################################################
# 1.3 Locate the land cover/soil type by lat/lon of in-situ data
ismn_land_list = sorted(glob.glob(path_processed + '/landcover/[A-Z]*.xlsx'))

df_land_all = []
for ife in range(14, len(ismn_land_list)):
    df_land = pd.read_excel(ismn_land_list[ife], index_col=0)
    df_land_all.append(df_land)
    del(df_land)
    print(ife)

df_land_desc = pd.concat(df_land_all)


########################################################################################################################
## 2. NN training
data_table = pd.read_excel('/Users/binfang/Downloads/Data File -5-23.xlsx', index_col=None, header=[1])
data_table = data_table.drop(data_table.index[[54, 78]])
# data_table = data_table.drop(data_table.index[54:])

x = data_table.drop(['K (m/s)'], axis=1)
y = data_table['K (m/s)']

# Imputate the input array
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(x)
x = imp_mean.transform(x)

# Normalize input array, and calculate base-10 logarithm of y
x_norm = preprocessing.normalize(x)
y_log = np.log10(y)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y_log, test_size=0.25, random_state=42)


regr = MLPRegressor(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(30, 30, 30), max_iter=1000)
regr.fit(x_train, y_train)
regr.score(x_train, y_train)
y_pred = regr.predict(x_test)


slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
plt.scatter(y_test, y_pred)
plt.plot(y_test, intercept+slope*y_test, '-', color='m')
plt.xlim(np.min(y_test)*1.05, np.max(y_test)*1.05)
plt.ylim(np.min(y_test)*1.05, np.max(y_test)*1.05)
plt.xlabel('Target Test')
plt.ylabel('Target Pred')
plt.text(-11.5, -8, 'R=' + str(round(r_value, 3)), ha='center', fontsize=16, fontweight='bold')


slope, intercept, r_value, p_value, std_err = stats.linregress(10**y_test, 10**y_pred)
plt.scatter(10**y_test, 10**y_pred)
plt.plot(10**y_test, intercept+slope*10**y_test, '-', color='m')
plt.xlim(np.min(10**y_test)*1.1, np.max(10**y_test)*1.1)
plt.ylim(np.min(10**y_test)*1.1, np.max(10**y_test)*1.1)
plt.xlabel('Target Test')
plt.ylabel('Target Pred')


# clf = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1000, alpha=0.0001,
#                      solver='lbfgs', verbose=10,  random_state=42,tol=0.000000001)
# y_train_arr = np.asarray(y_train, dtype='|S6')
# clf.fit(x_train, y_train_arr)
# y_pred = clf.predict(x_test)
