import os
import numpy as np
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
import itertools
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

########################################################################################################################
# Function 2. Subset and reproject the Geotiff data to WGS84 projection

def sub_n_reproj(input_mat, kwargs_input, sub_window, output_crs, band_n):
    # Get the georeference and bounding parameters of subset image
    kwargs_sub = kwargs_input.copy()
    kwargs_sub.update({
        'height': sub_window.height,
        'width': sub_window.width,
        'transform': rasterio.windows.transform(sub_window, kwargs_sub['transform']),
        'count': band_n
    })

    # Generate subset rasterio dataset
    input_ds_subset = MemoryFile().open(**kwargs_sub)
    if band_n == 1:
        input_mat = np.expand_dims(input_mat, axis=0)
    else:
        pass

    for i in range(band_n):
        input_ds_subset.write(
            input_mat[i, sub_window.row_off:sub_window.row_off+sub_window.height, sub_window.col_off:sub_window.col_off+sub_window.width], i+1)

    # Reproject a dataset
    transform_reproj, width_reproj, height_reproj = \
        calculate_default_transform(input_ds_subset.crs, output_crs,
                                    input_ds_subset.width, input_ds_subset.height, *input_ds_subset.bounds)
    kwargs_reproj = input_ds_subset.meta.copy()
    kwargs_reproj.update({
            'crs': output_crs,
            'transform': transform_reproj,
            'width': width_reproj,
            'height': height_reproj,
            'count': band_n
        })

    output_ds = MemoryFile().open(**kwargs_reproj)
    for i in range(band_n):
        reproject(source=rasterio.band(input_ds_subset, i+1), destination=rasterio.band(output_ds, i+1),
                  src_transform=input_ds_subset.transform, src_crs=input_ds_subset.crs,
                  dst_transform=transform_reproj, dst_crs=output_crs, resampling=Resampling.nearest)

    return output_ds

####################################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
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
# Path of ISMN data
path_ismn = '/Volumes/MyPassport/SMAP_Project/Datasets/ISMN'
# Path of VA watersheds
path_vaws = '/Users/binfang/Downloads/Processing/virginia_watersheds'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2020-12-31'
year = 2020 - 2015 + 1

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

varname_list_2 = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min',
                'lat_conus_ease_1km', 'lon_conus_ease_1km',
                'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind', 'row_conus_ease_1km_ind', 'col_conus_ease_1km_ind',
                'row_conus_ease_1km_from_9km_ind', 'col_conus_ease_1km_from_9km_ind',
                'row_conus_ease_9km_from_1km_ext33km_ind', 'col_conus_ease_9km_from_1km_ext33km_ind',
                'lat_conus_ease_9km', 'lon_conus_ease_9km', 'lat_conus_ease_12_5km', 'lon_conus_ease_12_5km',
                'row_conus_ease_1km_from_12_5km_ind', 'col_conus_ease_1km_from_12_5km_ind']

for x in range(len(varname_list_2)):
    var_obj = f[varname_list_2[x]][()]
    exec(varname_list_2[x] + '= var_obj')
    del(var_obj)
f.close()


########################################################################################################################
# 1. Extract SMAP SM data by lat/lon (CONUS)
df_gages_selected = pd.read_csv('/Users/binfang/Documents/SMAP_Project/data/gis_data/gagesII_9322_point_shapefile/gagesII_selected.csv',
                    index_col=0, sep=',')
df_smap_1km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_1km.xlsx', index_col=0, sheet_name='AM')

# 1.2 Locate the SMAP 1/9 km SM positions by lat/lon of in-situ data
stn_lat_all = np.array(df_gages_selected['LAT_GAGE'])
stn_lon_all = np.array(df_gages_selected['LNG_GAGE'])

stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
stn_row_9km_ind_all = []
stn_col_9km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_conus_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_conus_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    stn_row_9km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_conus_ease_9km)).item()
    stn_col_9km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_conus_ease_9km)).item()
    stn_row_9km_ind_all.append(stn_row_9km_ind)
    stn_col_9km_ind_all.append(stn_col_9km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind, stn_row_9km_ind, stn_col_9km_ind)

########################################################################################################################
# 2. Extract the SMAP 1/9 km SM by the indexing files

# 2.1 Load in watershed shapefile boundaries
path_shp = path_gis_data + '/va_watersheds/'
shp_file = "va_watersheds_reprj.shp"
shapefile = fiona.open(path_shp + '/' + shp_file, 'r')
# crop_shape = [feature['geometry'] for feature in shapefile]
shp_va_extent = list(shapefile.bounds)
output_crs = 'EPSG:4326'
[lat_1km_va, row_va_1km_ind, lon_1km_va, col_va_1km_ind] = \
    coordtable_subset(lat_conus_ease_1km, lon_conus_ease_1km, shp_va_extent[3], shp_va_extent[1], shp_va_extent[2], shp_va_extent[0])
sub_window_1km = Window(col_va_1km_ind[0], row_va_1km_ind[0], len(col_va_1km_ind), len(row_va_1km_ind))
kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_1km), 'height': len(lat_conus_ease_1km),
          'count': 2, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(1000.89502334956, 0.0, -12060785.031554235, 0.0, -1000.89502334956, 5854234.991137885)
          }
# Extract watershed boundaries
shape_name = []
shape_geometry = []
with fiona.open(path_shp + '/' + shp_file) as shapefile:
    for shape in shapefile:
        shape_name.append(shape['properties']['NAME'])
        shape_geometry.append(shape['geometry'])

ws_ind = [8, 21, 29, 31]
crop_shape = [[shape_geometry[x]] for x in ws_ind]
crop_shape_name = [[shape_name[x]] for x in ws_ind]


# 2.2 Extract 1km SMAP SM
smap_1km_sta_all = []
tif_files_1km_name_ind_all = []
for iyr in range(len(yearname)):

    os.chdir(path_smap +'/1km' + '/nldas/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Extract the file name
    tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
    tif_files_name_1year_ind = [date_seq_doy.index(item) for item in tif_files_name if item in date_seq_doy]
    date_seq_doy_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy]

    tif_files_1km_name_ind_all.append(tif_files_name_1year_ind)
    del(tif_files_name, tif_files_name_1year_ind)

    smap_1km_sta_1year = []
    for idt in range(len(date_seq_doy_1year_ind)):
        src_tf = rasterio.open(tif_files[date_seq_doy_1year_ind[idt]]).read()
        src_tf_out = sub_n_reproj(src_tf, kwargs_1km_sub, sub_window_1km, output_crs, 2)

        smap_1km_sta_1day = []
        for ilr in range(len(crop_shape)):
            masked_ds_1km, mask_transform_ds_1km = mask(dataset=src_tf_out, shapes=crop_shape[ilr], crop=True)
            masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
            masked_ds_1km = np.nanmean(masked_ds_1km, axis=0)
            smap_1km_sta_1day.append(masked_ds_1km)
            del(masked_ds_1km, mask_transform_ds_1km)
        # smap_1km_sta_1day = src_tf[:, stn_row_1km_ind_all, stn_col_1km_ind_all]
        # smap_1km_sta_1day = np.nanmean(smap_1km_sta_1day, axis=0)
        smap_1km_sta_1year.append(smap_1km_sta_1day)
        del(src_tf, smap_1km_sta_1day)
        print(tif_files[date_seq_doy_1year_ind[idt]])

    smap_1km_sta_all.append(smap_1km_sta_1year)
    del(smap_1km_sta_1year, date_seq_doy_1year_ind)

smap_1km_sta_all = list(itertools.chain(*smap_1km_sta_all))
smap_1km_ws_all = []
for idt in range(len(smap_1km_sta_all[0])):
    smap_1km_ws = [smap_1km_sta_all[x][idt] for x in range(len(smap_1km_sta_all))]
    smap_1km_ws = np.stack(smap_1km_ws)
    smap_1km_ws = np.reshape(smap_1km_ws, (smap_1km_ws.shape[0], smap_1km_ws.shape[1]*smap_1km_ws.shape[2]))
    smap_1km_ws = np.nanmean(smap_1km_ws, axis=1)
    smap_1km_ws_all.append(smap_1km_ws)
    del(smap_1km_ws)

smap_1km_ws_all = np.stack(smap_1km_ws_all, axis=0)
# smap_1km_sta_all = np.transpose(smap_1km_sta_all, (1, 0))

tif_files_1km_name_ind_all = np.concatenate(tif_files_1km_name_ind_all)
# smap_1km_sta_all = np.concatenate(smap_1km_sta_all)

# Fill the extracted SMAP SM into the proper position of days
smap_1km_ws_filled = np.empty((4, df_smap_1km_sta_am.shape[1]-1), dtype='float32')
smap_1km_ws_filled[:] = np.nan
for idt in range(len(tif_files_1km_name_ind_all)):
    smap_1km_ws_filled[:, tif_files_1km_name_ind_all[idt]] = smap_1km_ws_all[:, idt]

columns = df_smap_1km_sta_am.columns[1:]
index = list(itertools.chain(*crop_shape_name))
# index = df_gages_selected.index
# staname = df_gages_selected['STANAME']
df_smap_1km_ws = pd.DataFrame(smap_1km_ws_filled, columns=columns, index=index)
# df_smap_1km_ws = pd.concat([staname, df_smap_1km_ws], axis=1, sort=False)
writer_1km = pd.ExcelWriter(path_vaws + '/df_smap_1km_ws.xlsx')
df_smap_1km_ws.to_excel(writer_1km)
writer_1km.save()

df_smap_1km_ws = pd.read_excel(path_vaws + '/df_smap_1km_ws.xlsx', index_col=0)

# Calculate metrics of SMAP SM
smap_1km_array = np.array(df_smap_1km_sta.iloc[:, 1:])
smap_1km_array_max = np.nanmax(smap_1km_array, axis=1)
smap_1km_array_min = np.nanmin(smap_1km_array, axis=1)
smap_1km_array_mean = np.nanmean(smap_1km_array, axis=1)
smap_1km_array_std = np.nanstd(smap_1km_array, axis=1)
smap_1km_array_metrics = \
    np.stack((smap_1km_array_max, smap_1km_array_min, smap_1km_array_mean, smap_1km_array_std))
smap_1km_array_metrics = np.transpose(smap_1km_array_metrics, (1, 0))
df_smap_1km_array_metrics = pd.DataFrame(smap_1km_array_metrics, columns=['max', 'min', 'mean', 'stdev'], index=index)
df_smap_1km_array_metrics = pd.concat([staname, df_smap_1km_array_metrics], axis=1, sort=False)
writer_1km = pd.ExcelWriter(path_vaws + '/df_smap_1km_array_metrics.xlsx')
df_smap_1km_array_metrics.to_excel(writer_1km)
writer_1km.save()


########################################################################################################################
# 3. Process the daily streamflow data

streamflow_files = sorted(glob.glob(path_vaws + '/streamflow/*.txt'))

sf_data_all = []
sf_name_all = []
for ife in range(len(streamflow_files)):
    sf_table = pd.read_csv(streamflow_files[ife], delimiter='\t', header=29)
    sf_date = sf_table.iloc[:, 2]
    sf_data = np.array(sf_table.iloc[:, 3], dtype=float)
    sf_name = os.path.basename(streamflow_files[ife]).split('.')[-2][1:]

    sf_date_doy_all = []
    for idt in range(len(sf_date)):
        year_string = str(datetime.datetime.strptime(sf_date[idt], '%Y-%m-%d').date().timetuple().tm_year)
        day_string = str(datetime.datetime.strptime(sf_date[idt], '%Y-%m-%d').date().timetuple().tm_yday).zfill(3)
        sf_date_doy = year_string + day_string
        sf_date_doy_all.append(sf_date_doy)
        del(sf_date_doy, year_string, day_string)

    sf_date_doy_ind = [date_seq_doy.index(item) for item in sf_date_doy_all if item in date_seq_doy]
    sf_data_filled = np.empty((len(date_seq_doy),), dtype='float32')
    sf_data_filled[:] = np.nan
    for idt in range(len(sf_date_doy_ind)):
        sf_data_filled[sf_date_doy_ind[idt]] = sf_data[idt]

    sf_data_all.append(sf_data_filled)
    sf_name_all.append(sf_name)
    print(sf_name)
    del(sf_table, sf_date, sf_data, sf_name, sf_date_doy_all, sf_date_doy_ind, sf_data_filled)


df_gages_selected = pd.read_csv('/Users/binfang/Documents/SMAP_Project/data/gis_data/gagesII_9322_point_shapefile/gagesII_selected.csv',
                    index_col=0, sep=',')
df_smap_1km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_1km.xlsx', index_col=0, sheet_name='AM')
columns = df_smap_1km_sta_am.columns[1:]
index = df_gages_selected.index
staname = df_gages_selected['STANAME']
df_sf_data = pd.DataFrame(sf_data_all, columns=columns, index=index)
df_sf_data = pd.concat([staname, df_sf_data], axis=1, sort=False)
writer_1km = pd.ExcelWriter(path_vaws + '/df_sf_data.xlsx')
df_sf_data.to_excel(writer_1km)
writer_1km.save()

df_sf_data = pd.read_excel(path_vaws + '/df_sf_data.xlsx', index_col=0)


########################################################################################################################
# 4. Plot the SMAP SM data versus daily streamflow data
nan_array_1 = np.empty((df_smap_1km_ws.shape[0], 90), dtype='float32')
nan_array_1[:] = np.nan
nan_array_2 = np.empty((df_sf_data.shape[0], 90), dtype='float32')
nan_array_2[:] = np.nan
smap_1km_ws = np.concatenate((nan_array_1, df_smap_1km_ws.iloc[:, :]), axis=1)
sf_data = np.concatenate((nan_array_2, df_sf_data.iloc[:, 1:]), axis=1)
stn_name = list(df_sf_data['STANAME'])
ws_name = list(df_smap_1km_ws.index)

# Figure 1
fig = plt.figure(figsize=(8, 4), dpi=200)
plt.subplots_adjust(left=0.13, right=0.9, bottom=0.14, top=0.9, hspace=0.25, wspace=0.25)
y1 = sf_data[6, :]
y2 = smap_1km_ws[0, :]
y2 = pd.Series(y2).fillna(method='ffill', limit=7)
ax = fig.add_subplot(1, 1, 1)
lns1 = ax.bar(np.arange(len(y1)), y1, width=1, color='b', label='Q('+stn_name[6]+')', alpha=0.5)
# lns1 = ax.plot(y1, c='b', label='Stream Flow', marker='s', linestyle='None', markersize=0.3)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, len(y1) + 12, len(y1) // 6))
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019', '2020']
mticks = ax.get_xticks()
ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xticklabels(labels, minor=True)

ax.set_ylim(0, 5000)
ax.set_yticks(np.arange(0, 6000, 1000))
ax.tick_params(axis='y', labelsize=10)
ax.grid(linestyle='--')

ax2 = ax.twinx()
ax2.set_ylim(0, 0.5)
# ax2.invert_yaxis()
ax2.set_yticks(np.arange(0, 0.6, 0.1))
# lns2 = ax2.bar(np.arange(len(y2)), y2, width=1, color='k', label='Soil Moisture', alpha=0.5)
lns2 = ax2.plot(y2, c='k', label='Soil Moisture', marker='s', markersize=0.3, linewidth=0.5)
ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = [lns1] + lns2
labels = [l.get_label() for l in handles]
plt.legend(handles, labels, loc=(0, 0.94), borderaxespad=0, ncol=2, prop={"size": 7})

fig.text(0.5, 0.02, 'Years', ha='center', fontsize=10, fontweight='bold')
fig.text(0.02, 0.4, 'Stream Flow ($\mathregular{ft^3/s}$)', rotation='vertical', fontsize=10, fontweight='bold')
fig.text(0.96, 0.4, 'SM ($\mathregular{m^3/m^3}$)', rotation='vertical', fontsize=10, fontweight='bold')
plt.suptitle(ws_name[0], fontsize=12, y=0.98, fontweight='bold')
plt.savefig(path_vaws + '/plots/' + ws_name[0] + '.png')
plt.close(fig)

# Figure 2
fig = plt.figure(figsize=(8, 4), dpi=200)
plt.subplots_adjust(left=0.13, right=0.9, bottom=0.14, top=0.9, hspace=0.25, wspace=0.25)
y1 = sf_data[4, :]
y2 = sf_data[5, :]
y3 = smap_1km_ws[1, :]
y3 = pd.Series(y3).fillna(method='ffill', limit=7)
ax = fig.add_subplot(1, 1, 1)
lns1 = ax.bar(np.arange(len(y1)), y1, width=1, color='b', label='Q('+stn_name[4]+')', alpha=0.5, zorder=2)
lns2 = ax.bar(np.arange(len(y2)), y2, width=1, color='g', label='Q('+stn_name[5]+')', alpha=0.5, zorder=1)
# lns1 = ax.plot(y1, c='b', label='Stream Flow', marker='s', linestyle='None', markersize=0.3)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, len(y1) + 12, len(y1) // 6))
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019', '2020']
mticks = ax.get_xticks()
ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xticklabels(labels, minor=True)

ax.set_ylim(0, 5000)
ax.set_yticks(np.arange(0, 6000, 1000))
ax.tick_params(axis='y', labelsize=10)
ax.grid(linestyle='--')

ax2 = ax.twinx()
ax2.set_ylim(0, 0.5)
# ax2.invert_yaxis()
ax2.set_yticks(np.arange(0, 0.6, 0.1))
# lns2 = ax2.bar(np.arange(len(y2)), y2, width=1, color='k', label='Soil Moisture', alpha=0.5)
lns3 = ax2.plot(y3, c='k', label='Soil Moisture', marker='s', linewidth=0.5, markersize=0.3)
ax2.tick_params(axis='y', labelsize=10)


# add all legends together
handles = [lns1] + [lns2] + lns3
labels = [l.get_label() for l in handles]
plt.legend(handles, labels, loc=(0, 0.9), borderaxespad=0, ncol=2, prop={"size": 7})

fig.text(0.5, 0.02, 'Years', ha='center', fontsize=10, fontweight='bold')
fig.text(0.02, 0.4, 'Stream Flow ($\mathregular{ft^3/s}$)', rotation='vertical', fontsize=10, fontweight='bold')
fig.text(0.96, 0.4, 'SM ($\mathregular{m^3/m^3}$)', rotation='vertical', fontsize=10, fontweight='bold')
plt.suptitle(ws_name[1], fontsize=12, y=0.98, fontweight='bold')
plt.savefig(path_vaws + '/plots/' + ws_name[1] + '.png')
plt.close(fig)

# Figure 3
fig = plt.figure(figsize=(8, 4), dpi=200)
plt.subplots_adjust(left=0.13, right=0.9, bottom=0.14, top=0.9, hspace=0.25, wspace=0.25)
y1 = sf_data[0, :]
y2 = sf_data[1, :]
y3 = sf_data[2, :]
y4 = smap_1km_ws[2, :]
y4 = pd.Series(y4).fillna(method='ffill', limit=7)
ax = fig.add_subplot(1, 1, 1)
lns1 = ax.bar(np.arange(len(y1)), y1, width=1, color='b', label='Q('+stn_name[0]+')', alpha=0.5, zorder=1)
lns2 = ax.bar(np.arange(len(y2)), y2, width=1, color='g', label='Q('+stn_name[1]+')', alpha=0.5, zorder=2)
lns3 = ax.bar(np.arange(len(y3)), y3, width=1, color='y', label='Q('+stn_name[2]+')', alpha=0.5, zorder=3)
# lns1 = ax.plot(y1, c='b', label='Stream Flow', marker='s', linestyle='None', markersize=0.3)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, len(y1) + 12, len(y1) // 6))
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019', '2020']
mticks = ax.get_xticks()
ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xticklabels(labels, minor=True)

ax.set_ylim(0, 5000)
ax.set_yticks(np.arange(0, 6000, 1000))
ax.tick_params(axis='y', labelsize=10)
ax.grid(linestyle='--')

ax2 = ax.twinx()
ax2.set_ylim(0, 0.5)
# ax2.invert_yaxis()
ax2.set_yticks(np.arange(0, 0.6, 0.1))
# lns2 = ax2.bar(np.arange(len(y2)), y2, width=1, color='k', label='Soil Moisture', alpha=0.5)
lns4 = ax2.plot(y4, c='k', label='Soil Moisture', marker='s', linewidth=0.5, markersize=0.3)
ax2.tick_params(axis='y', labelsize=10)


# add all legends together
handles = [lns1] + [lns2] + [lns3] + lns4
labels = [l.get_label() for l in handles]
plt.legend(handles, labels, loc=(0, 0.9), borderaxespad=0, ncol=2, prop={"size": 7})

fig.text(0.5, 0.02, 'Years', ha='center', fontsize=10, fontweight='bold')
fig.text(0.02, 0.4, 'Stream Flow ($\mathregular{ft^3/s}$)', rotation='vertical', fontsize=10, fontweight='bold')
fig.text(0.96, 0.4, 'SM ($\mathregular{m^3/m^3}$)', rotation='vertical', fontsize=10, fontweight='bold')
plt.suptitle(ws_name[2], fontsize=12, y=0.98, fontweight='bold')
plt.savefig(path_vaws + '/plots/' + ws_name[2] + '.png')
plt.close(fig)


# Figure 4
fig = plt.figure(figsize=(8, 4), dpi=200)
plt.subplots_adjust(left=0.13, right=0.9, bottom=0.14, top=0.9, hspace=0.25, wspace=0.25)
y1 = sf_data[7, :]
y2 = smap_1km_ws[3, :]
y2 = pd.Series(y2).fillna(method='ffill', limit=7)
ax = fig.add_subplot(1, 1, 1)
lns1 = ax.bar(np.arange(len(y1)), y1, width=1, color='b', label='Q('+stn_name[7]+')', alpha=0.5)
# lns1 = ax.plot(y1, c='b', label='Stream Flow', marker='s', linestyle='None', markersize=0.3)

plt.xlim(0, len(y1))
ax.set_xticks(np.arange(0, len(y1) + 12, len(y1) // 6))
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019', '2020']
mticks = ax.get_xticks()
ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xticklabels(labels, minor=True)

ax.set_ylim(0, 5000)
ax.set_yticks(np.arange(0, 6000, 1000))
ax.tick_params(axis='y', labelsize=10)
ax.grid(linestyle='--')

ax2 = ax.twinx()
ax2.set_ylim(0, 0.5)
# ax2.invert_yaxis()
ax2.set_yticks(np.arange(0, 0.6, 0.1))
# lns2 = ax2.bar(np.arange(len(y2)), y2, width=1, color='k', label='Soil Moisture', alpha=0.5)
lns2 = ax2.plot(y2, c='k', label='Soil Moisture', marker='s', linewidth=0.5, markersize=0.3)
ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = [lns1] + lns2
labels = [l.get_label() for l in handles]
plt.legend(handles, labels, loc=(0, 0.94), borderaxespad=0, ncol=2, prop={"size": 7})

fig.text(0.5, 0.02, 'Years', ha='center', fontsize=10, fontweight='bold')
fig.text(0.02, 0.4, 'Stream Flow ($\mathregular{ft^3/s}$)', rotation='vertical', fontsize=10, fontweight='bold')
fig.text(0.96, 0.4, 'SM ($\mathregular{m^3/m^3}$)', rotation='vertical', fontsize=10, fontweight='bold')
plt.suptitle(ws_name[3], fontsize=12, y=0.98, fontweight='bold')
plt.savefig(path_vaws + '/plots/' + ws_name[3] + '.png')
plt.close(fig)

