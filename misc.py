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
start_date = '1981-01-01'
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
# 1. Check the completeness of downloaded GLDAS files
os.chdir(path_ltdr)
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
# 4.1 Process the DEM data

grd_folders = sorted(glob.glob('/Users/binfang/Downloads/usgs_dem/*/'))

out_ds_list = []
for ife in range(len(grd_folders)):
    grd_file_path = sorted([f.path for f in os.scandir(grd_folders[ife]) if f.is_dir()])[0]
    src_tf = gdal.Open(grd_file_path + '/w001001.adf')
    src_tf_arr = src_tf.ReadAsArray()
    src_tf_arr[np.where(src_tf_arr <= 0)] = np.nan

    out_ds = gdal.GetDriverByName('MEM').Create('', src_tf.RasterXSize, src_tf.RasterYSize, 1,  # Number of bands
                                                gdal.GDT_Float32)
    out_ds.SetGeoTransform(src_tf.GetGeoTransform())
    out_ds.SetProjection(src_tf.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(src_tf_arr)

    out_ds_list.append(out_ds)
    del(src_tf, out_ds)
    print(ife)

# Open file and warp the target raster dimensions and geotransform
out_dem = gdal.Warp('', out_ds_list, format='MEM', dstSRS='EPSG:4326', warpOptions=['SKIP_NOSOURCE=YES'], errorThreshold=0,
                   resampleAlg=gdal.GRA_NearestNeighbour)
out_dem_shape = out_dem.ReadAsArray().shape

# Create a raster of EASE grid projection at 1 km resolution
out_ds_tiff = gdal.GetDriverByName('GTiff').Create('/Users/binfang/Downloads/se_dem.tif',
     out_dem_shape[1], out_dem_shape[0], 1, gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
out_ds_tiff.SetGeoTransform(out_dem.GetGeoTransform())
out_ds_tiff.SetProjection(out_dem.GetProjection())
out_ds_tiff.GetRasterBand(1).WriteArray(out_dem.ReadAsArray())
out_ds_tiff.GetRasterBand(1).SetNoDataValue(0)
out_ds_tiff = None

del(out_ds_list, out_dem, out_ds_tiff)


# 4.2 Load in watershed shapefile boundaries and generate masks of the watersheds
shpf_list = ['basin_1_vec.shp', 'basin_2_vec.shp', 'basin_3_vec.shp', 'basin_4_vec.shp', 'basin_5_vec.shp',
                 'basin_6_vec.shp', 'basin_7_vec.shp']
crop_shape_ws_list = []
for i in range(len(shpf_list)):
    shapefile_ws = fiona.open('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/se_watershed/' + shpf_list[i], 'r')
    crop_shape_ws = [feature["geometry"] for feature in shapefile_ws]
    crop_shape_ws_list.append(crop_shape_ws)
    # shp_ws_extent = list(shapefile_ws.bounds)

# 4.3 Extract the annual PRISM data betwen 1942 - 2017
bil_folders = sorted(glob.glob('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/Data/*/'))

ds_prism_all = []
for iyr in range(len(bil_folders)):
    # bil_path = sorted(os.listdir(bil_folders[iyr])) # subfolder
    if iyr <= 1980 - 1942:
        bil_file = bil_folders[iyr] + 'PRISM_ppt_stable_4kmM2_' + str(1942+iyr) + '_bil.bil'
        ds_prism = rasterio.open(bil_file)
    else:
        bil_file = bil_folders[iyr] + 'PRISM_ppt_stable_4kmM3_' + str(1942+iyr) + '_bil.bil'
        ds_prism = rasterio.open(bil_file)

    ds_prism_all.append(ds_prism)

    print(bil_file)
    del(bil_file, ds_prism)


# 4.4 Mask the PRISM data by watershed shapefiles
masked_ds_1km_all = []
for isp in range(len(crop_shape_ws_list)):

    masked_ds_1km_1shp = []
    for iyr in range(len(ds_prism_all)):

        masked_ds_1km, mask_transform_ds_1km = mask(dataset=ds_prism_all[iyr], shapes=crop_shape_ws_list[isp], crop=True)
        masked_ds_1km[np.where(masked_ds_1km <= 0)] = np.nan
        masked_ds_1km = masked_ds_1km.squeeze()
        masked_ds_1km_1shp.append(masked_ds_1km)
        del(masked_ds_1km)

    masked_ds_1km_all.append(masked_ds_1km_1shp)
    del(masked_ds_1km_1shp)

ds_avg_all = []
ds_max_all = []
ds_min_all = []
for isp in range(len(crop_shape_ws_list)):
    ds_arr = np.array(masked_ds_1km_all[isp])
    ds_shape = ds_arr.shape
    ds_arr = np.reshape(ds_arr, (ds_shape[0], ds_shape[1]*ds_shape[2]))
    ds_avg = np.nanmean(ds_arr, axis=1)
    ds_max = np.nanmax(ds_arr, axis=1)
    ds_min = np.nanmin(ds_arr, axis=1)
    ds_avg_all.append(ds_avg)
    ds_max_all.append(ds_max)
    ds_min_all.append(ds_min)
    del(ds_arr, ds_shape, ds_avg, ds_max, ds_min)

ds_avg_all = np.array(ds_avg_all)
ds_max_all = np.array(ds_max_all)
ds_min_all = np.array(ds_min_all)


########################################################################################################################
# 4.5 Make the tables
# (4. Ochlock; 1. SR White; 6. CR Bruce; 5. Escambia; 3. Perdido; 2. St Marys; 7. Ap R)

ds_avg = [ds_avg_all[3, :], ds_avg_all[0, :], ds_avg_all[5, :], ds_avg_all[4, :], ds_avg_all[2, :], ds_avg_all[1, :], ds_avg_all[6, :]]
ds_avg = np.array(ds_avg)
ds_avg = np.transpose(ds_avg)

ds_max = [ds_max_all[3, :], ds_max_all[0, :], ds_max_all[5, :], ds_max_all[4, :], ds_max_all[2, :], ds_max_all[1, :], ds_max_all[6, :]]
ds_max = np.array(ds_max)
ds_max = np.transpose(ds_max)

ds_min = [ds_min_all[3, :], ds_min_all[0, :], ds_min_all[5, :], ds_min_all[4, :], ds_min_all[2, :], ds_min_all[1, :], ds_min_all[6, :]]
ds_min = np.array(ds_min)
ds_min = np.transpose(ds_min)

ds_avg_avg = np.nanmean(ds_avg, axis=0)
ds_avg_std = np.nanstd(ds_avg, axis=0)
ds_avg_norm = (ds_avg - ds_avg_avg)/ds_avg_std


rows = np.arange(1942, 2018)
rows = [str(i) for i in rows]
columns = ['Ochlock', 'SR White', 'CR Bruce', 'Escambia', 'Perdido', 'St Marys', 'Ap R']

df_precp_1 = pd.DataFrame(ds_avg, columns=columns, index=rows)
df_precp_2 = pd.DataFrame(ds_avg_norm, columns=columns, index=rows)
df_precp_3 = pd.DataFrame(ds_max, columns=columns, index=rows)
df_precp_4 = pd.DataFrame(ds_min, columns=columns, index=rows)

df_precp_1.to_csv('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/se_watershed/df_precp_1.csv')
df_precp_2.to_csv('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/se_watershed/df_precp_2.csv')
df_precp_3.to_csv('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/se_watershed/df_precp_3.csv')
df_precp_4.to_csv('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/se_watershed/df_precp_4.csv')


# 4.6 Calculate lat/lon tables for PRISM data and find the corresponding indices for the lat/lon
src_tf = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/Data/PRISM_ppt_stable_4kmM2_1942_all_bil/PRISM_ppt_stable_4kmM2_1942_bil.bil')

cellsize = src_tf.GetGeoTransform()[1]
lon_len = src_tf.RasterXSize
lat_len = src_tf.RasterYSize
lon_min = src_tf.GetGeoTransform()[0] + cellsize/2
lon_max = lon_min + cellsize * (lon_len - 1)
lat_max = src_tf.GetGeoTransform()[3] - cellsize/2
lat_min = lat_max - cellsize * (lat_len - 1)
lat_prism = np.linspace(lat_max, lat_min, lat_len)
lon_prism = np.linspace(lon_min, lon_max, lon_len)

stn_lat = [30.82, 32.87, 33.45]
stn_lon = [-84.62, -85.19, -84.82]

stn_row_ind_all = []
stn_col_ind_all = []
for ist in range(len(stn_lat)):
    stn_row_ind = np.argmin(np.absolute(stn_lat[ist] - lat_prism)).item()
    stn_col_ind = np.argmin(np.absolute(stn_lon[ist] - lon_prism)).item()
    stn_row_ind_all.append(stn_row_ind)
    stn_col_ind_all.append(stn_col_ind)

prism_stn_all = []
for iyr in range(len(ds_prism_all)):
    prism_mat = ds_prism_all[iyr].read(1)
    prism_stn = prism_mat[stn_row_ind_all, stn_col_ind_all]
    prism_stn_all.append(prism_stn)
prism_stn_all = np.array(prism_stn_all)

# 4.7 make the Map
timerange = np.arange(1942, 2022, 4)
xtick = [str(timerange[x]) for x in range(len(timerange))]
fig = plt.figure(figsize=(13, 8), facecolor='w', edgecolor='k')
plt.plot(prism_stn_all)
plt.xticks(np.arange(0, len(timerange)*4, 4), xtick)
plt.legend(['Bainbridge Intl Paper, GA', 'West Point, GA', 'Newnan 5N, GA'])
plt.savefig('/Users/binfang/Downloads/Processing/processed_data/prism_prec')
plt.close()

# Make the tables for the three GA stations
rows = np.arange(1942, 2018)
rows = [str(i) for i in rows]
columns = ['Bainbridge Intl Paper, GA', 'West Point, GA', 'Newnan 5N, GA']
df_precp_ga = pd.DataFrame(prism_stn_all, columns=columns, index=rows)
df_precp_ga.to_csv('/Volumes/MyPassport/SMAP_Project/Datasets/PRISM/se_watershed/df_precp_ga.csv')


########################################################################################################################
# 5. Subset the 1 km downscaled SMAP SM data

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





