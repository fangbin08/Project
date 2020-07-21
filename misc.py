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
# Path of SMAP Mat-files
path_matfile = '/Volumes/MyPassport/SMAP_Project/Datasets/CONUS'
# Path of shapefile
path_shp = '/Users/binfang/Downloads/Processing/shapefiles'
# Path of SMAP output
path_output = '/Users/binfang/Downloads/Processing/smap_output'

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
yearname = np.linspace(2015, 2020, 6, dtype='int')
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
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'interdist_ease_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


# # Load in variables
# os.chdir(path_workspace)
# f = h5py.File('ds_parameters.hdf5', 'r')
# varname_list = list(f.keys())
#
# for x in range(len(varname_list)):
#     var_obj = f[varname_list[x]][()]
#     exec(varname_list[x] + '= var_obj')
# f.close()

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
os.chdir('/Users/binfang/Downloads/Processing/bash_codes/download/')
# Downloaded files
files = sorted(glob.glob('*.nc4'))

# List of files to download
os.chdir('/Users/binfang/Downloads/Processing/bash_codes/download/')
with open("4969365288-download.txt", "r") as ins:
    url_list = []
    for line in ins:
        url_list.append(line)
ins.close()

len(files) == len(url_list)

files_list = [url_list[x][:-1].split('/')[-1] for x in range(len(url_list))]
modfile_miss_ind = [files_list.index(i) for i in files_list if i not in files]
modfile_miss_url = [url_list[modfile_miss_ind[x]] for x in range(len(modfile_miss_ind))]

os.chdir('/Users/binfang/Downloads/Processing/bash_codes/download/missed')
file = open('modis_miss.txt', 'w')
with open('modis_miss.txt', 'w') as f:
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

# ########################################################################################################################
# # 1.3 Locate the land cover/soil type by lat/lon of in-situ data
# ismn_land_list = sorted(glob.glob(path_processed + '/landcover/[A-Z]*.xlsx'))
#
# df_land_all = []
# for ife in range(14, len(ismn_land_list)):
#     df_land = pd.read_excel(ismn_land_list[ife], index_col=0)
#     df_land_all.append(df_land)
#     del(df_land)
#     print(ife)
#
# df_land_desc = pd.concat(df_land_all)
#

########################################################################################################################
## 4. NN training
data_table = pd.read_excel('/Users/binfang/Downloads/Processing/NN_training/Data File -7-18.xlsx', index_col=None, header=[1])
# data_table = data_table.drop(data_table.index[[54, 78]])
# data_table = data_table.drop(data_table.index[54:])


x = data_table.drop(['K (m/s)'], axis=1)
y = data_table['K (m/s)']

# # Imputate the input array
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp_mean.fit(x)
# x = imp_mean.transform(x)

# Normalize input array, and calculate base-10 logarithm of y
x_norm = preprocessing.normalize(x)
y_log = np.log10(y)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y_log, test_size=0.25, random_state=42)


regr = MLPRegressor(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(30, 30, 30), max_iter=5000)
regr.fit(x_train, y_train)
regr.score(x_train, y_train)
y_pred = regr.predict(x_test)


slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
plt.scatter(y_test, y_pred)
plt.plot(y_test, intercept+slope*y_test, '-', color='m')
plt.xlim(np.min(y_test)*1.05, np.max(y_test)*1.05)
plt.ylim(np.min(y_test)*1.05, np.max(y_test)*1.05)
plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
plt.grid(linestyle='--')
plt.xlabel('Target Test')
plt.ylabel('Target Pred')
plt.text(-10, -6, 'R=' + str(round(r_value, 3)), ha='center', fontsize=16, fontweight='bold')


df_data = pd.DataFrame({'y_test': np.array(y_test),
                        'y_pred': y_pred})
writer = pd.ExcelWriter('/Users/binfang/Downloads/nn_data.xlsx')
df_data.to_excel(writer)
writer.save()




# clf = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1000, alpha=0.0001,
#                      solver='lbfgs', verbose=10,  random_state=42,tol=0.000000001)
# y_train_arr = np.asarray(y_train, dtype='|S6')
# clf.fit(x_train, y_train_arr)
# y_pred = clf.predict(x_test)

########################################################################################################################
# 5. Define a function to find the matched gantry id and assigned to the corresponding car GPS points
def gps_identifier(path_main, file_gantry, file_car, file_car_output):

    # Read the gantry and car coordinates files
    grid_size = 0.01  # 1km ~= 0.01 degree, half of the grid size is 0.005 degree (500 m)

    df_gantry = pd.read_csv(path_main + file_gantry, delim_whitespace=True)
    f_vc_lat = np.array(df_gantry['f_vc_lat'])
    f_vc_lon = np.array(df_gantry['f_vc_lon'])
    # Calculate max/min lat/lon for each gantry grid using half grid size
    f_vc_lat_min = f_vc_lat - grid_size/2
    f_vc_lat_max = f_vc_lat + grid_size/2
    f_vc_lon_min = f_vc_lon - grid_size/2
    f_vc_lon_max = f_vc_lon + grid_size/2

    df_car = pd.read_csv(path_main + file_car, index_col=0, sep=',')
    car_lon = np.array(df_car['F_LONGITUDE'])
    car_lat = np.array(df_car['F_LATITUDE'])

    # Find the car GPS points dropped in any single gantry point
    coord_drop = [np.where((car_lat <= f_vc_lat_max[x]) & (car_lat >= f_vc_lat_min[x])
                           & (car_lon <= f_vc_lon_max[x]) & (car_lon >= f_vc_lon_min[x]))[0]
                  for x in range(len(f_vc_lat_max))]

    # Find the gantry points containing car GPS points (by determining the list of gantry points are not empty)
    coord_drop_len = np.array([len(coord_drop[x]) for x in range(len(coord_drop))])
    coord_drop_len_nonzero_ind = np.where(coord_drop_len != 0)[0]

    # Create a new field and find the matched gantry id for the car GPS points dropped in it and add it to the new field
    # The field will be appended to the car GPS table
    nan_array = np.empty(len(car_lon))
    nan_array[:] = np.nan
    df_newfield = pd.DataFrame({'f_vc_gantry_id': nan_array})
    for x in range(len(coord_drop_len_nonzero_ind)):
        df_newfield.iloc[coord_drop[coord_drop_len_nonzero_ind[x]]] = \
            df_gantry['f_vc_gantry_id'].iloc[coord_drop_len_nonzero_ind[x]]

    # Join the new field to the car GPS dataframe
    df_car.reset_index(inplace=True, drop=True)
    df_newfield.reset_index(inplace=True, drop=True)

    df_car_edited = pd.concat([df_car, df_newfield], axis=1)
    df_car_edited_output = df_car_edited.to_csv(path_main + file_car_output)

    return df_car_edited_output


########################################################################################################################
# Test
import pandas as pd
import numpy as np

path_main_test = '/Users/binfang/Downloads/Processing/car_coords'
file_gantry_test = '/gantry_locations.csv'
file_car_test = '/suA8J27S.csv'
file_car_output_test = file_car_test.split('.')[0] + '_new.' + file_car_test.split('.')[1]

gps_identifier(path_main_test, file_gantry_test, file_car_test, file_car_output_test)


########################################################################################################################
#6. Subset SMAP 1 km/9 km data
shapefile = fiona.open(path_shp + '/Indochina_boundary/Indochina_boundary.shp', 'r')
crop_shape = [feature["geometry"] for feature in shapefile]
shp_extent = list(shapefile.bounds)

lat_sub_max = shp_extent[3]
lat_sub_min = shp_extent[1]
lon_sub_max = shp_extent[2]
lon_sub_min = shp_extent[0]

output_crs = 'EPSG:4326'
[lat_sub_ease_1km, row_sub_ease_1km_ind, lon_sub_ease_1km, col_sub_ease_1km_ind] = coordtable_subset\
    (lat_world_ease_1km, lon_world_ease_1km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)
[lat_sub_ease_9km, row_sub_ease_9km_ind, lon_sub_ease_9km, col_sub_ease_9km_ind] = coordtable_subset\
    (lat_world_ease_9km, lon_world_ease_9km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)

# 1 km
for iyr in range(4, len(yearname)):
    smap_file_path = path_smap + '/1km/gldas/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src_tf = rasterio.open(smap_file_list[idt])

        # Subset the SM data by bounding box
        sub_window_1km = Window(col_sub_ease_1km_ind[0], row_sub_ease_1km_ind[0],
                                len(col_sub_ease_1km_ind), len(row_sub_ease_1km_ind))
        kwargs_1km_sub = src_tf.meta.copy()
        kwargs_1km_sub.update({
            'height': sub_window_1km.height,
            'width': sub_window_1km.width,
            'transform': rasterio.windows.transform(sub_window_1km, src_tf.transform)})

        # Write to Geotiff file
        with rasterio.open(path_output + '/1km/' + \
                           os.path.basename(smap_file_list[idt]), 'w', **kwargs_1km_sub) as output_ds:
            output_ds.write(src_tf.read(window=sub_window_1km))

        print(os.path.basename(smap_file_list[idt]))


# 9 km
col_coor_sub = -17367530.44516138 + col_sub_ease_9km_ind[0].item() * interdist_ease_9km
row_coor_sub = 7314540.79258289 - row_sub_ease_9km_ind[0].item() * interdist_ease_9km

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_sub_ease_9km),
           'height': len(lat_sub_ease_9km), 'count': 2, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_9km, 0.0, col_coor_sub, 0.0, -interdist_ease_9km, row_coor_sub)}

for iyr in range(4, len(yearname)):
    for imo in range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'
        if os.path.exists(hdf_file_smap_9km) == True:
            f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
            varname_list_smap_9km = list(f_read_smap_9km.keys())
            smap_9km_load_am = f_read_smap_9km[varname_list_smap_9km[0]][row_sub_ease_9km_ind[0]:row_sub_ease_9km_ind[-1]+1,
                    col_sub_ease_9km_ind[0]:col_sub_ease_9km_ind[-1]+1, :]
            smap_9km_load_pm = f_read_smap_9km[varname_list_smap_9km[1]][row_sub_ease_9km_ind[0]:row_sub_ease_9km_ind[-1]+1,
                    col_sub_ease_9km_ind[0]:col_sub_ease_9km_ind[-1]+1, :]
            f_read_smap_9km.close()

            for idt in range(smap_9km_load_am.shape[2]):
                dst = np.stack([smap_9km_load_am[:, :, idt], smap_9km_load_pm[:, :, idt]], axis=0)
                day_str = str(yearname[iyr]) + '-' + monthname[imo] + '-' + str(idt+1).zfill(2)
                dst_writer = rasterio.open(path_output + '/9km' + '/smap_sm_9km_' + str(yearname[iyr]) + \
                                           str(datetime.datetime.strptime(day_str, '%Y-%m-%d').timetuple().tm_yday).zfill(3) + '.tif',
                                   'w', **profile)
                dst_writer.write(dst)
                dst_writer = None

                print(day_str)
                del(day_str, dst_writer, dst)

            del (smap_9km_load_am, smap_9km_load_pm, f_read_smap_9km)

        else:
            pass



########################################################################################################################
# 7.1 Generate the coordinates table for SMAP data
import fiona
import pandas as pd
import numpy as np
import h5py
import glob
import rasterio

#########################################################################################
# (Function 1) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

#########################################################################################
path_hdf = '/Users/binfang/Documents/SMAP_Project/smap_codes/coords_table.hdf5'
path_shp = '/Users/binfang/Downloads/Processing/shapefiles/Indochina_boundary/Indochina_boundary.shp'
path_smap_file = '/Users/binfang/Downloads/Processing/smap_output/1km_ic'
path_smap_coord = '/Users/binfang/Downloads/Processing/Indochina/grid_coords.csv'
path_smap_data = '/Users/binfang/Downloads/Processing/Indochina/smap_sm_data.csv'

f = h5py.File(path_hdf, 'r')
varname_list = list(f.keys())
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()

shapefile = fiona.open(path_shp, 'r')
crop_shape = [feature["geometry"] for feature in shapefile]
shp_extent = list(shapefile.bounds)

lat_sub_max = shp_extent[3]
lat_sub_min = shp_extent[1]
lon_sub_max = shp_extent[2]
lon_sub_min = shp_extent[0]

[lat_sub_ease_1km, row_sub_ease_1km_ind, lon_sub_ease_1km, col_sub_ease_1km_ind] = coordtable_subset\
    (lat_world_ease_1km, lon_world_ease_1km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)
[lon_sub_ease_1km_mesh, lat_sub_ease_1km_mesh] = np.meshgrid(lon_sub_ease_1km, lat_sub_ease_1km)
lon_sub_ease_1km_mesh = lon_sub_ease_1km_mesh.flatten()
lat_sub_ease_1km_mesh = lat_sub_ease_1km_mesh.flatten()
id_array = np.arange(len(lon_sub_ease_1km_mesh))

df_smap_grids = pd.DataFrame({'GridID': id_array,
                              'Latitude': lat_sub_ease_1km_mesh,
                              'Longitude':lon_sub_ease_1km_mesh})
df_smap_grids.to_csv(path_smap_coord)

# 7.2 Write the SMAP SM data to the table
smap_file_list = sorted(glob.glob(path_smap_file + '/*.tif'))

# Write the first day's data to the table
date_id = int(smap_file_list[0].split('.')[0][-7:])
src_tf = rasterio.open(smap_file_list[0]).read()
src_tf = np.nanmean(src_tf, axis=0).flatten()
src_tf = src_tf.reshape(1, -1)
id_array_str = [str(id_array[x]) for x in range(len(id_array))]
df_tf = pd.DataFrame(src_tf, index=[date_id], columns=id_array_str)
df_tf.to_csv(path_smap_data, mode='w')
del(id_array)

# Write the rest days to the table
for idt in range(1, len(smap_file_list)):
    date_id = int(smap_file_list[idt].split('.')[0][-7:])
    src_tf = rasterio.open(smap_file_list[idt]).read()
    src_tf = np.nanmean(src_tf, axis=0).flatten()
    src_tf = src_tf.reshape(1, -1)
    df_tf = pd.DataFrame(src_tf, index=[date_id], columns=id_array_str)
    df_tf.to_csv(path_smap_data, mode='a', header=False)
    print(smap_file_list[idt])
    del(date_id, src_tf, df_tf)


# var_name = ['lat_world_ease_1km', 'lon_world_ease_1km']
# with h5py.File('/Users/binfang/Documents/SMAP_Project/smap_codes/coords_table.hdf5', 'w') as f:
#     for x in var_name:
#         f.create_dataset(x, data=eval(x))
# f.close()