import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib.ticker as mticker
import datetime
import glob
import gdal
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.crs import CRS
import h5py
import pandas as pd
import gdal
from osgeo import ogr
import fiona
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

########################################################################################################################
# (Function 1) Extract data layers from MODIS HDF5 files, and write to GeoTiff
def hdf_subdataset_extraction(hdf_files, subdataset_id, band_n):

    # Open the dataset
    global band_ds
    hdf_ds = gdal.Open(hdf_files, gdal.GA_ReadOnly)

    # Loop read data of specified bands from subdataset_id
    size_1dim = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly).ReadAsArray().astype(np.int16).shape
    band_array = np.empty([size_1dim[0], size_1dim[1], len(subdataset_id)])
    for idn in range(len(subdataset_id)):
        band_ds = gdal.Open(hdf_ds.GetSubDatasets()[subdataset_id[idn]][0], gdal.GA_ReadOnly)
        band_ds_arr = band_ds.ReadAsArray().astype(np.float32)
        band_ds_arr[np.where((band_ds_arr >= 32700) | (band_ds_arr <= -32767))] = np.nan
        band_ds_arr = band_ds_arr * 0.1
        # Write into numpy array
        band_array[:, :, idn] = band_ds_arr
        del(band_ds_arr)

    # Build output path
    # band_path = os.path.join(path_modis_op, os.path.basename(os.path.splitext(hdf_files)[0]) + "-ctd" + ".tif")
    # Write raster
    out_ds = gdal.GetDriverByName('MEM').Create('', band_ds.RasterXSize, band_ds.RasterYSize, band_n, #Number of bands
                                  gdal.GDT_Float32)
    out_ds.SetGeoTransform(band_ds.GetGeoTransform())
    out_ds.SetProjection(band_ds.GetProjection())

    # Loop write each band to Geotiff file
    for idb in range(len(subdataset_id)):
        out_ds.GetRasterBand(idb+1).WriteArray(band_array[:, :, idb])
        out_ds.GetRasterBand(idb+1).SetNoDataValue(0)
    # out_ds = None  #close dataset to write to disc

    return out_ds


#######################################################################################################################################
# (Function 2) Subset the coordinates table of desired area
def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

########################################################################################################################
# (Function 3) Subset and reproject the Geotiff data to WGS84 projection
def sub_n_reproj(input_mat, kwargs_sub, sub_window, output_crs):
    # Get the georeference and bounding parameters of subset image
    kwargs_sub.update({
        'height': sub_window.height,
        'width': sub_window.width,
        'transform': rasterio.windows.transform(sub_window, kwargs_sub['transform'])})

    # Generate subset rasterio dataset
    input_ds_subset = MemoryFile().open(**kwargs_sub)
    input_mat = np.expand_dims(input_mat, axis=0)
    input_ds_subset.write(input_mat)

    # Reproject a dataset
    transform_reproj, width_reproj, height_reproj = \
        calculate_default_transform(input_ds_subset.crs, output_crs,
                                    input_ds_subset.width, input_ds_subset.height, *input_ds_subset.bounds)
    kwargs_reproj = input_ds_subset.meta.copy()
    kwargs_reproj.update({
            'crs': output_crs,
            'transform': transform_reproj,
            'width': width_reproj,
            'height': height_reproj
        })

    output_ds = MemoryFile().open(**kwargs_reproj)
    reproject(source=rasterio.band(input_ds_subset, 1), destination=rasterio.band(output_ds, 1),
              src_transform=input_ds_subset.transform, src_crs=input_ds_subset.crs,
              dst_transform=transform_reproj, dst_crs=output_crs, resampling=Resampling.nearest)

    return output_ds

########################################################################################################################
# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of source MODIS data
path_modis = '/Users/binfang/Downloads/Processing/MOD16A2_V6_vmd'
# Path of output tiff
path_op = '/Users/binfang/Downloads/Processing/ET_Output'
# Path of Shapefile
path_shp = '/Users/binfang/Downloads/Processing/shp'
# Path of SMAP
path_smap_1km_gldas = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/gldas'
path_smap_1km_nldas = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/nldas'
path_smap_9km = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/9km'
# Path of ISMN
path_ismn = '/Volumes/MyPassport/SMAP_Project/Datasets/ISMN/processed_data'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200810'

# World extent corner coordinates
lat_world_max = 90
lat_world_min = -90
lon_world_max = 180
lon_world_min = -180

# VMD extent corner coordinates
lat_vmd_max = 10.966
lat_vmd_min = 8.562
lon_vmd_max = 106.796
lon_vmd_min = 104.475

# Interdistance of EASE grid projection grids
interdist_ease_1km = 1000.89502334956
interdist_ease_9km = 9009.093602916


subdataset_id = [0] # the band ID to extract (0th for ET_500m)
band_n = 1 # the number of bands

yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]


# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'lat_conus_ease_1km', 'lon_conus_ease_1km',
                'lat_world_ease_1km', 'lon_world_ease_1km', 'row_conus_ease_1km_ind', 'col_conus_ease_1km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2015-04-01'
end_date = '2019-12-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_month = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))
    date_seq_month.append(date_str.timetuple().tm_mon)

date_seq_array = np.array([int(date_seq[x]) for x in range(len(date_seq))])
date_seq_month = np.array(date_seq_month)

########################################################################################################################
# 1. Compare GLDAS V2.1 derived SMAP SM with NLDAS derived SMAP SM

# 1.1 Load the site lat/lon from excel files and Locate the SMAP 1/9 km SM positions by lat/lon of in-situ data
# Find the indices of the days between April - September
# month_list = np.array([int(date_seq[x][4:6]) for x in range(len(date_seq))])
month_list_ind = np.where((date_seq_month >= 4) & (date_seq_month <= 9))[0]
month_list_ind = month_list_ind + 2 #First two columns are lat/lon

ismn_list = sorted(glob.glob(path_ismn + '/[A-Z]*.xlsx'))

coords_all = []
df_table_am_all = []
df_table_pm_all = []
netname_all = []
for ife in range(14, len(ismn_list)):
    df_table_am = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='AM')
    df_table_pm = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='PM')

    netname = os.path.basename(ismn_list[ife]).split('_')[1]
    netname = [netname] * df_table_am.shape[0]
    coords = df_table_am[['lat', 'lon']]
    netname_all.append(netname)
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


# 1.2 Locate the SM pixel positions
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_conus_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_conus_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    del(stn_row_1km_ind, stn_col_1km_ind)


# 1.1 Extract daily GLDAS derived SMAP SM (August, 2019)
for iyr in [4]:  # range(yearname):

    os.chdir(path_smap_1km_gldas + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    smap_gldas_all = []
    smap_gldas_sta_all = []
    for idt in range(178, 209):#range(len(tif_files)):
        src_tf = rasterio.open(tif_files[idt])
        src_tf = src_tf.read()[:, row_conus_ease_1km_ind[0]:row_conus_ease_1km_ind[-1]+1,
                 col_conus_ease_1km_ind[0]:col_conus_ease_1km_ind[-1]+1]
        smap_gldas_1day = np.nanmean(src_tf, axis=0)
        smap_gldas_sta_1day = smap_gldas_1day[stn_row_1km_ind_all, stn_col_1km_ind_all]

        smap_gldas_all.append(smap_gldas_1day)
        smap_gldas_sta_all.append(smap_gldas_sta_1day)
        del(src_tf, smap_gldas_1day, smap_gldas_sta_1day)
        print(tif_files[idt])

smap_gldas_all = np.array(smap_gldas_all)
smap_gldas_sta_all = np.array(smap_gldas_sta_all)


# 1.2 Extract daily NLDAS derived SMAP SM (August, 2019)
for iyr in [4]:  # range(yearname):

    os.chdir(path_smap_1km_nldas + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    smap_nldas_all = []
    smap_nldas_sta_all = []
    for idt in range(210, 241):#range(len(tif_files)):
        src_tf = rasterio.open(tif_files[idt])
        src_tf = src_tf.read()
        smap_nldas_1day = np.nanmean(src_tf, axis=0)
        smap_nldas_sta_1day = smap_nldas_1day[stn_row_1km_ind_all, stn_col_1km_ind_all]

        smap_nldas_all.append(smap_nldas_1day)
        smap_nldas_sta_all.append(smap_nldas_sta_1day)
        del(src_tf, smap_nldas_1day, smap_nldas_sta_1day)
        print(tif_files[idt])

smap_nldas_all = np.array(smap_nldas_all)
smap_nldas_sta_all = np.array(smap_nldas_sta_all)


# Store variable-length type variables to the parameter file
var_name = ['smap_gldas_all', 'smap_gldas_sta_all', 'smap_nldas_all', 'smap_nldas_sta_all']
with h5py.File('/Users/binfang/Downloads/Processing/processed_data/smap_ldas.hdf5', 'a') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()


########################################################################################################################
# 1.2 Single maps
# Load in the data
f = h5py.File("/Users/binfang/Downloads/Processing/processed_data/smap_ldas.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

smap_gldas_all_mean = np.nanmean(smap_gldas_all, axis=0)
# smap_gldas_all_mean = np.expand_dims(smap_gldas_all_mean, axis=0)
smap_nldas_all_mean = np.nanmean(smap_nldas_all, axis=0)
# smap_nldas_all_mean = np.expand_dims(smap_nldas_all_mean, axis=0)
del(smap_gldas_all, smap_nldas_all)

# Keyword arguments of the tiff file to write
output_crs = 'EPSG:4326'
window_1km = Window(0, 0, len(lon_conus_ease_1km), len(lat_conus_ease_1km))
kwargs = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0,
          'width': len(lon_conus_ease_1km), 'height': len(lat_conus_ease_1km),
          'count': 1, 'crs': rasterio.crs.CRS.from_dict(init='epsg:6933'),
          'transform': Affine(1000.89502334956, 0.0, -12060785.03616676, 0.0, -1000.89502334956, 5854236.10968041)
          }

# SMAP SM (GLDAS)
smap_gldas_all_mean_ds = sub_n_reproj(smap_gldas_all_mean, kwargs, window_1km, output_crs)
regrid_size = smap_gldas_all_mean_ds.transform[4]
lat_conus_ease_1km_regrid = np.linspace(lat_conus_ease_1km[0], lat_conus_ease_1km[-1], num=smap_gldas_all_mean_ds.height)
lon_conus_ease_1km_regrid = np.linspace(lon_conus_ease_1km[0], lon_conus_ease_1km[-1], num=smap_gldas_all_mean_ds.width)
smap_gldas_all_mean_ds = smap_gldas_all_mean_ds.read()
smap_gldas_all_mean_ds = np.squeeze(smap_gldas_all_mean_ds, axis=0)
# SMAP SM (NLDAS)
smap_nldas_all_mean_ds = sub_n_reproj(smap_nldas_all_mean, kwargs, window_1km, output_crs)
smap_nldas_all_mean_ds = smap_nldas_all_mean_ds.read()
smap_nldas_all_mean_ds = np.squeeze(smap_nldas_all_mean_ds, axis=0)

xx_wrd, yy_wrd = np.meshgrid(lon_conus_ease_1km_regrid, lat_conus_ease_1km_regrid) # Create the map matrix
shape_conus = ShapelyFeature(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

# SMAP SM (GLDAS)
fig = plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-125, -67, 25, 53], crs=ccrs.PlateCarree())
ax.add_feature(shape_conus)
img = ax.pcolormesh(xx_wrd, yy_wrd, smap_gldas_all_mean_ds, transform=ccrs.PlateCarree(), vmin=0, vmax=0.6, cmap='Spectral')
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
gl.xlocator = mticker.FixedLocator([-125, -110, -95, -80, -65])
gl.ylocator = mticker.FixedLocator([25, 30, 40, 50])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
plt.suptitle('SMAP SM (GLDAS)', y=0.95, fontsize=15)
plt.show()
plt.savefig(path_results + '/smap_sm_gldas.png')

# SMAP SM (NLDAS)
fig = plt.figure(num=None, figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-125, -67, 25, 53], crs=ccrs.PlateCarree())
ax.add_feature(shape_conus)
img = ax.pcolormesh(xx_wrd, yy_wrd, smap_nldas_all_mean_ds, transform=ccrs.PlateCarree(), vmin=0, vmax=0.6, cmap='Spectral')
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
gl.xlocator = mticker.FixedLocator([-125, -110, -95, -80, -65])
gl.ylocator = mticker.FixedLocator([25, 30, 40, 50])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
plt.suptitle('SMAP SM (NLDAS)', y=0.95, fontsize=15)
plt.show()
plt.savefig(path_results + '/smap_sm_nldas.png')


########################################################################################################################
# 1.3 Validation

df_subset_begin_ind = df_table_am_all.columns.get_loc('2019213')
df_subset_end_ind = df_table_am_all.columns.get_loc('2019243')
df_table_am_sub = df_table_am_all.iloc[:, df_subset_begin_ind:df_subset_end_ind+1]
df_table_pm_sub = df_table_pm_all.iloc[:, df_subset_begin_ind:df_subset_end_ind+1]
df_table = (np.array(df_table_am_sub) + np.array(df_table_pm_sub)) / 2.0

smap_gldas_sta_all = smap_gldas_sta_all.T
smap_nldas_sta_all = smap_nldas_sta_all.T

df_split_ind = [52, 9, 140, 9, 188, 404, 119, 113]
df_split_ind_cumsum = np.cumsum(df_split_ind)
df_table_split = np.split(df_table, df_split_ind_cumsum) # split by each month
smap_gldas_sta_split = np.split(smap_gldas_sta_all, df_split_ind_cumsum)
smap_nldas_sta_split = np.split(smap_nldas_sta_all, df_split_ind_cumsum)


# Make the plots
stat_gldas = []
stat_nldas = []
# ind_slc_all = []
for ist in range(len(df_split_ind)):

    x = df_table_split[ist].flatten()
    y1 = smap_gldas_sta_split[ist].flatten()
    y2 = smap_nldas_sta_split[ist].flatten()
    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]

    if len(ind_nonnan) > 5:
        x = x[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]

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


        fig = plt.figure(figsize=(11, 6.5))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        ax = fig.add_subplot(111)

        ax.scatter(x, y1, s=10, c='m', marker='s', label='GLDAS')
        ax.scatter(x, y2, s=10, c='b', marker='o', label='NLDAS')
        ax.plot(x, intercept_1+slope_1*x, '-', color='m')
        ax.plot(x, intercept_2+slope_2*x, '-', color='b')

        plt.xlim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.set_title(netname_all[ist][0], pad=20, fontsize=15, weight='bold')
        plt.grid(linestyle='--')
        plt.legend(loc='upper left')
        # plt.show()
        plt.savefig(path_results + '/plots/' + netname_all[ist][0] + '.png')
        plt.close(fig)
        stat_gldas.append(stat_array_1)
        stat_nldas.append(stat_array_2)
        # ind_slc_all.append(ist)
        print(ist)
        del(stat_array_1, stat_array_2)

    else:
        pass

stat_array_gldas = np.array(stat_gldas)
stat_array_nldas = np.array(stat_nldas)

columns_validation = ['number', 'r_sq', 'ubrmse', 'bias', 'p_value', 'conf_int']
index_validation = [netname_all[0][0], netname_all[3][0], netname_all[4][0], netname_all[5][0], netname_all[6][0], netname_all[7][0]]
# index_validation = df_coords.index[ind_slc_all]

# stat_array_gldas = np.concatenate((id, stat_array_gldas), axis=1)
# stat_array_nldas = np.concatenate((id, stat_array_nldas), axis=1)
df_stat_gldas = pd.DataFrame(stat_array_gldas, columns=columns_validation, index=index_validation)
# df_stat_gldas = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_gldas], axis=1)
df_stat_nldas = pd.DataFrame(stat_array_nldas, columns=columns_validation, index=index_validation)
# df_stat_nldas = pd.concat([df_table_am_all['network'][ind_slc_all], df_stat_nldas], axis=1)
writer_gldas = pd.ExcelWriter(path_results + '/stat_gldas.xlsx')
writer_nldas = pd.ExcelWriter(path_results + '/stat_nldas.xlsx')
df_stat_gldas.to_excel(writer_gldas)
df_stat_nldas.to_excel(writer_nldas)
writer_gldas.save()
writer_nldas.save()