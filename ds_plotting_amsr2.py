import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib.ticker as mticker
import numpy as np
import glob
import h5py
import gdal
from osgeo import ogr
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.crs import CRS
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
# (Function 1) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind


########################################################################################################################
# Function 2. Subset and reproject the Geotiff data to WGS84 projection

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
# Path of Land mask
path_lmask = '/Users/binfang/Downloads/Processing/processed_data'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of downscaled SM
path_amsr2_sm_ds = '/Users/binfang/Downloads/Processing/Downscale'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200115'
# Path of preview
path_preview = '/Users/binfang/Documents/SMAP_Project/results/results_191202/preview'
# Path of AMSR2 data
path_amsr2 = '/Volumes/MyPassport/SMAP_Project/Datasets/AMSR2'
# Path of downscaled SM
path_amsr2_sm_ds = '/Users/binfang/Downloads/Processing/processed_data/AMSR2_ds'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'cellsize_1km', 'cellsize_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_ease_25km', 'lon_world_ease_25km', 'row_world_ease_9km_from_1km_ind',
                'col_world_ease_9km_from_1km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

########################################################################################################################

# 1 AMSR2 SM maps (Worldwide)

# 1.1 Composite the data of the first 16 days of one specific month
# Load in AMSR2 9 km SM
year_plt = [2019]
month_plt = [4, 8]
days_begin = 1
days_end = 16
days_n = days_end - days_begin + 1

matsize_9km = [2, len(lat_world_ease_9km), len(lon_world_ease_9km)]
amsr2_9km_mean_1_all = np.empty(matsize_9km, dtype='float32')
amsr2_9km_mean_1_all[:] = np.nan
amsr2_9km_mean_2_all = np.copy(amsr2_9km_mean_1_all)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_amsr2_9km = path_procdata + '/amsr2_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_amsr2_9km = h5py.File(hdf_file_amsr2_9km, "r")
        varname_list_amsr2_9km = list(f_read_amsr2_9km.keys())

        amsr2_9km_load = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
        amsr2_9km_load[:] = np.nan
        for idt in range(days_begin-1, days_end):
            amsr2_9km_load[:, :, 2*idt+0] = f_read_amsr2_9km[varname_list_amsr2_9km[0]][:, :, idt] # AM
            amsr2_9km_load[:, :, 2*idt+1] = f_read_amsr2_9km[varname_list_amsr2_9km[1]][:, :, idt] # PM
        f_read_amsr2_9km.close()

        amsr2_9km_mean_1 = np.nanmean(amsr2_9km_load[:, :, :days_n], axis=2)
        amsr2_9km_mean_2 = np.nanmean(amsr2_9km_load[:, :, days_n:], axis=2)
        del(amsr2_9km_load)

        amsr2_9km_mean_1_all[imo, :, :] = amsr2_9km_mean_1
        amsr2_9km_mean_2_all[imo, :, :] = amsr2_9km_mean_2
        del(amsr2_9km_mean_1, amsr2_9km_mean_2)
        print(imo)


# Load in AMSR2 1 km SM
amsr2_1km_agg_stack = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
amsr2_1km_agg_stack[:] = np.nan
amsr2_1km_mean_1_all = np.empty(matsize_9km, dtype='float32')
amsr2_1km_mean_1_all[:] = np.nan
amsr2_1km_mean_2_all = np.copy(amsr2_1km_mean_1_all)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        for idt in range(days_n):  # 16 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_amsr2_1km = path_amsr2_sm_ds + '/' + str(iyr) + '/amsr2_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_amsr2_1km)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

            # Aggregate to 9 km
            for ilr in range(2):
                src_tf_arr_1layer = src_tf_arr[ilr, :, :]
                amsr2_sm_1km_agg = np.array\
                    ([np.nanmean(src_tf_arr_1layer[row_world_ease_9km_from_1km_ind[x], :], axis=0)
                        for x in range(len(lat_world_ease_9km))])
                amsr2_sm_1km_agg = np.array\
                    ([np.nanmean(amsr2_sm_1km_agg[:, col_world_ease_9km_from_1km_ind[y]], axis=1)
                        for y in range(len(lon_world_ease_9km))])
                amsr2_sm_1km_agg = np.fliplr(np.rot90(amsr2_sm_1km_agg, 3))
                amsr2_1km_agg_stack[:, :, 2*idt+ilr] = amsr2_sm_1km_agg
                del(amsr2_sm_1km_agg, src_tf_arr_1layer)

            print(str_date)
            del(src_tf_arr)

        amsr2_1km_mean_1 = np.nanmean(amsr2_1km_agg_stack[:, :, :days_n], axis=2)
        amsr2_1km_mean_2 = np.nanmean(amsr2_1km_agg_stack[:, :, days_n:], axis=2)

        amsr2_1km_mean_1_all[imo, :, :] = amsr2_1km_mean_1
        amsr2_1km_mean_2_all[imo, :, :] = amsr2_1km_mean_2
        del(amsr2_1km_mean_1, amsr2_1km_mean_2)


amsr2_data_stack = np.stack((amsr2_1km_mean_1_all, amsr2_9km_mean_1_all, amsr2_1km_mean_2_all, amsr2_9km_mean_2_all))


# 1.2 Maps of April, 2019
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_9km, lat_world_ease_9km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')


title_content = ['1 km (04/01 - 04/08)', '10 km (04/01 - 04/08)', '1 km (04/09 - 04/16)', '10 km (04/09 - 04/16)']
columns = 2
rows = 2
fig = plt.figure(figsize=(13, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(), extent=[-180, 180, -70, 90])
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, amsr2_data_stack[ipt, 0, :, :], vmin=0, vmax=0.5, cmap='gist_earth_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 13}
    gl.ylabel_style = {'size': 13}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=13, x=0.5, labelpad=-0.5)
    ax.set_title(title_content[ipt], pad=30, fontsize=16, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.3, wspace=0.25)
plt.show()
plt.savefig(path_results + '/amsr2_sm_comp_04.png')


# 1.3 Maps of August, 2019
title_content = ['1 km (08/01 - 08/08)', '10 km (08/01 - 08/08)', '1 km (08/09 - 08/16)', '10 km (08/09 - 08/16)']
columns = 2
rows = 2
fig = plt.figure(figsize=(13, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(), extent=[-180, 180, -70, 90])
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, amsr2_data_stack[ipt, 1, :, :], vmin=0, vmax=0.5, cmap='gist_earth_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 13}
    gl.ylabel_style = {'size': 13}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=13, x=0.5, labelpad=-0.5)
    ax.set_title(title_content[ipt], pad=30, fontsize=16, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.3, wspace=0.25)
plt.show()
plt.savefig(path_results + '/amsr2_sm_comp_08.png')


# 1.4 Maps of April, 2019 (CONUS)
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_9km, lat_world_ease_9km) # Create the map matrix
shape_conus = ShapelyFeature(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')


title_content = ['1 km (04/01 - 04/08)', '10 km (04/01 - 04/08)', '1 km (04/09 - 04/16)', '10 km (04/09 - 04/16)']
columns = 2
rows = 2
fig = plt.figure(figsize=(13, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(), extent=[-125, -67, 25, 53])
    ax.add_feature(shape_conus, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, amsr2_data_stack[ipt, 0, :, :], vmin=0, vmax=0.5, cmap='gist_earth_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    # gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    # gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlocator = mticker.MultipleLocator(base=15)
    gl.ylocator = mticker.MultipleLocator(base=9)
    gl.xlabel_style = {'size': 13}
    gl.ylabel_style = {'size': 13}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=13, x=0.5, labelpad=-0.5)
    ax.set_title(title_content[ipt], pad=30, fontsize=16, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.3, wspace=0.25)
plt.show()
plt.savefig(path_results + '/amsr2_sm_comp_04_conus.png')



# 1.5 Maps of August, 2019 (CONUS)
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_9km, lat_world_ease_9km) # Create the map matrix
shape_conus = ShapelyFeature(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')


title_content = ['1 km (08/01 - 08/08)', '10 km (08/01 - 08/08)', '1 km (08/09 - 08/16)', '10 km (08/09 - 08/16)']
columns = 2
rows = 2
fig = plt.figure(figsize=(13, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(), extent=[-125, -67, 25, 53])
    ax.add_feature(shape_conus, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, amsr2_data_stack[ipt, 1, :, :], vmin=0, vmax=0.5, cmap='gist_earth_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    # gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    # gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlocator = mticker.MultipleLocator(base=15)
    gl.ylocator = mticker.MultipleLocator(base=9)
    gl.xlabel_style = {'size': 13}
    gl.ylabel_style = {'size': 13}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=13, x=0.5, labelpad=-0.5)
    ax.set_title(title_content[ipt], pad=30, fontsize=16, weight='bold')
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.3, wspace=0.25)
plt.show()
plt.savefig(path_results + '/amsr2_sm_comp_08_conus.png')


########################################################################################################################

# 3 River Basin maps

# 3.1 Danube RB

path_shp_dan = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_DANUBE'
os.chdir(path_shp_dan)
shp_dan_file = "Aqueduct_river_basins_DANUBE.shp"
shp_dan_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_dan_file, 0)
shp_dan_extent = list(shp_dan_ds.GetLayer().GetExtent())

#Load and subset the region of Danube RB (amsr2 9 km)
[lat_9km_dan, row_dan_9km_ind, lon_9km_dan, col_dan_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

# Load and subset amsr2 9 km SM of Danube RB
year_plt = [2019]
month_plt = [8]
days_begin = 1
days_end = 24
days_n = days_end - days_begin + 1

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_amsr2_9km = path_procdata + '/amsr2_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_amsr2_9km = h5py.File(hdf_file_amsr2_9km, "r")
        varname_list_amsr2_9km = list(f_read_amsr2_9km.keys())

        amsr2_9km_load = np.empty((len(row_dan_9km_ind), len(col_dan_9km_ind), days_n*2))
        amsr2_9km_load[:] = np.nan
        for idt in range(days_begin-1, days_end):
            amsr2_9km_load[:, :, 2*idt+0] = f_read_amsr2_9km[varname_list_amsr2_9km[0]][row_dan_9km_ind[0]:row_dan_9km_ind[-1]+1,
                col_dan_9km_ind[0]:col_dan_9km_ind[-1]+1, idt] # AM
            amsr2_9km_load[:, :, 2*idt+1] = f_read_amsr2_9km[varname_list_amsr2_9km[1]][row_dan_9km_ind[0]:row_dan_9km_ind[-1]+1,
                col_dan_9km_ind[0]:col_dan_9km_ind[-1]+1, idt] # PM
        f_read_amsr2_9km.close()

        amsr2_9km_mean_1 = np.nanmean(amsr2_9km_load[:, :, :days_n//3*2], axis=2)
        amsr2_9km_mean_2 = np.nanmean(amsr2_9km_load[:, :, days_n//3*2:days_n//3*4], axis=2)
        amsr2_9km_mean_3 = np.nanmean(amsr2_9km_load[:, :, days_n//3*4:], axis=2)
        amsr2_9km_data_stack = np.stack((amsr2_9km_mean_1, amsr2_9km_mean_2, amsr2_9km_mean_3))
        amsr2_9km_data_stack = np.float32(amsr2_9km_data_stack)

        del(amsr2_9km_load, amsr2_9km_mean_1, amsr2_9km_mean_2, amsr2_9km_mean_3)



#Load and subset the region of Danube RB (amsr2 1 km)
[lat_1km_dan, row_dan_1km_ind, lon_1km_dan, col_dan_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

amsr2_1km_load = np.empty((len(row_dan_1km_ind), len(col_dan_1km_ind), days_n*2))
amsr2_1km_load[:] = np.nan

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        for idt in range(days_n): # 16 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_amsr2_1km = path_amsr2_sm_ds + '/' + str(iyr) + '/amsr2_sm_1km_ds_' + str(iyr) + str_doy + '.tif'
            src_tf = gdal.Open(tif_file_amsr2_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_dan_1km_ind[0]:row_dan_1km_ind[-1]+1,
                         col_dan_1km_ind[0]:col_dan_1km_ind[-1]+1]
            src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
            amsr2_1km_load[:, :, 2*idt+0] = src_tf_arr[:, :, 0]
            amsr2_1km_load[:, :, 2*idt+1] = src_tf_arr[:, :, 1]

            print(str_date)
            del(src_tf_arr)

        amsr2_1km_mean_1 = np.nanmean(amsr2_1km_load[:, :, :days_n//3*2], axis=2)
        amsr2_1km_mean_2 = np.nanmean(amsr2_1km_load[:, :, days_n//3*2:days_n//3*4], axis=2)
        amsr2_1km_mean_3 = np.nanmean(amsr2_1km_load[:, :, days_n//3*4:], axis=2)

        amsr2_1km_data_stack = np.stack((amsr2_1km_mean_1, amsr2_1km_mean_2, amsr2_1km_mean_3))
        amsr2_1km_data_stack = np.float32(amsr2_1km_data_stack)
        del(amsr2_1km_load)


# Subplot maps
# Load in watershed shapefile boundaries
shapefile_dan = fiona.open(path_shp_dan + '/' + shp_dan_file, 'r')
crop_shape_dan = [feature["geometry"] for feature in shapefile_dan]
shp_dan_extent = list(shapefile_dan.bounds)

output_crs = 'EPSG:4326'

# Subset and reproject the amsr2 SM data at watershed
# 1 km
masked_ds_dan_1km_all = []
for n in range(amsr2_1km_data_stack.shape[0]):
    sub_window_dan_1km = Window(col_dan_1km_ind[0], row_dan_1km_ind[0], len(col_dan_1km_ind), len(row_dan_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    amsr2_sm_dan_1km_output = sub_n_reproj(amsr2_1km_data_stack[n, :, :], kwargs_1km_sub, sub_window_dan_1km, output_crs)

    masked_ds_dan_1km, mask_transform_ds_dan_1km = mask(dataset=amsr2_sm_dan_1km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_1km[np.where(masked_ds_dan_1km == 0)] = np.nan
    masked_ds_dan_1km = masked_ds_dan_1km.squeeze()

    masked_ds_dan_1km_all.append(masked_ds_dan_1km)

masked_ds_dan_1km_all = np.asarray(masked_ds_dan_1km_all)


# 9 km
masked_ds_dan_9km_all = []
for n in range(amsr2_9km_data_stack.shape[0]):
    sub_window_dan_9km = Window(col_dan_9km_ind[0], row_dan_9km_ind[0], len(col_dan_9km_ind), len(row_dan_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    amsr2_sm_dan_9km_output = sub_n_reproj(amsr2_9km_data_stack[n, :, :], kwargs_9km_sub, sub_window_dan_9km, output_crs)

    masked_ds_dan_9km, mask_transform_ds_dan_9km = mask(dataset=amsr2_sm_dan_9km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_9km[np.where(masked_ds_dan_9km == 0)] = np.nan
    masked_ds_dan_9km = masked_ds_dan_9km.squeeze()

    masked_ds_dan_9km_all.append(masked_ds_dan_9km)

masked_ds_dan_9km_all = np.asarray(masked_ds_dan_9km_all)


# Make the subplot maps
title_content = ['1 km (08/01 - 08/08)', '10 km (08/01 - 08/08)', '1 km (08/09 - 08/16)', '10 km (08/09 - 08/16)',
              '1 km (08/17 - 08/24)', '10 km (08/17 - 08/24)']
feature_shp_dan = ShapelyFeature(Reader(path_shp_dan + '/' + shp_dan_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_dan = np.array(amsr2_sm_dan_1km_output.bounds)
extent_dan = extent_dan[[0, 2, 1, 3]]

columns = 2
rows = 3
fig = plt.figure(figsize=(11, 8), facecolor='w', edgecolor='k')
# fig = plt.figure(num=None, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(rows, columns, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='gist_earth_r',
               extent=extent_dan)
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.5, labelpad=-0.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=3)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 9 km
    ax = fig.add_subplot(rows, columns, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_9km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='gist_earth_r',
               extent=extent_dan)
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.5, labelpad=-0.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=3)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.35, wspace=0.3)
plt.savefig(path_results + '/amsr2_sm_comp_danube_1.png')






# 3.2 Ganga-Brahmaputra RB

path_shp_gb = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_GANGES - BRAHMAPUTRA'
os.chdir(path_shp_gb)
shp_gb_file = "Aqueduct_river_basins_GANGES - BRAHMAPUTRA.shp"
shp_gb_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_gb_file, 0)
shp_gb_extent = list(shp_gb_ds.GetLayer().GetExtent())

#Subset the region of Ganga-Brahmaputra (9 km)
[lat_9km_gb, row_gb_9km_ind, lon_9km_gb, col_gb_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_gb_extent[3], shp_gb_extent[2], shp_gb_extent[1], shp_gb_extent[0])

# Load and subset amsr2 9 km SM of GB RB
year_plt = [2019]
month_plt = [4]
days_begin = 1
days_end = 24
days_n = days_end - days_begin + 1

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_amsr2_9km = path_procdata + '/amsr2_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_amsr2_9km = h5py.File(hdf_file_amsr2_9km, "r")
        varname_list_amsr2_9km = list(f_read_amsr2_9km.keys())

        amsr2_9km_load = np.empty((len(row_gb_9km_ind), len(col_gb_9km_ind), days_n*2))
        amsr2_9km_load[:] = np.nan
        for idt in range(days_begin-1, days_end):
            amsr2_9km_load[:, :, 2*idt+0] = f_read_amsr2_9km[varname_list_amsr2_9km[0]][row_gb_9km_ind[0]:row_gb_9km_ind[-1]+1,
                col_gb_9km_ind[0]:col_gb_9km_ind[-1]+1, idt] # AM
            amsr2_9km_load[:, :, 2*idt+1] = f_read_amsr2_9km[varname_list_amsr2_9km[1]][row_gb_9km_ind[0]:row_gb_9km_ind[-1]+1,
                col_gb_9km_ind[0]:col_gb_9km_ind[-1]+1, idt] # PM
        f_read_amsr2_9km.close()

        amsr2_9km_mean_1 = np.nanmean(amsr2_9km_load[:, :, :days_n//3*2], axis=2)
        amsr2_9km_mean_2 = np.nanmean(amsr2_9km_load[:, :, days_n//3*2:days_n//3*4], axis=2)
        amsr2_9km_mean_3 = np.nanmean(amsr2_9km_load[:, :, days_n//3*4:], axis=2)
        amsr2_9km_data_stack = np.stack((amsr2_9km_mean_1, amsr2_9km_mean_2, amsr2_9km_mean_3))
        amsr2_9km_data_stack = np.float32(amsr2_9km_data_stack)

        del(amsr2_9km_load, amsr2_9km_mean_1, amsr2_9km_mean_2, amsr2_9km_mean_3)



#Load and subset the region of GB RB (amsr2 1 km)
[lat_1km_gb, row_gb_1km_ind, lon_1km_gb, col_gb_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_gb_extent[3], shp_gb_extent[2], shp_gb_extent[1], shp_gb_extent[0])

amsr2_1km_load = np.empty((len(row_gb_1km_ind), len(col_gb_1km_ind), days_n*2))
amsr2_1km_load[:] = np.nan

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        for idt in range(days_n): # 16 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_amsr2_1km = path_amsr2_sm_ds + '/' + str(iyr) + '/amsr2_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_amsr2_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_gb_1km_ind[0]:row_gb_1km_ind[-1]+1,
                         col_gb_1km_ind[0]:col_gb_1km_ind[-1]+1]
            src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
            amsr2_1km_load[:, :, 2*idt+0] = src_tf_arr[:, :, 0]
            amsr2_1km_load[:, :, 2*idt+1] = src_tf_arr[:, :, 1]

            print(str_date)
            del(src_tf_arr)

        amsr2_1km_mean_1 = np.nanmean(amsr2_1km_load[:, :, :days_n//3*2], axis=2)
        amsr2_1km_mean_2 = np.nanmean(amsr2_1km_load[:, :, days_n//3*2:days_n//3*4], axis=2)
        amsr2_1km_mean_3 = np.nanmean(amsr2_1km_load[:, :, days_n//3*4:], axis=2)

        amsr2_1km_data_stack = np.stack((amsr2_1km_mean_1, amsr2_1km_mean_2, amsr2_1km_mean_3))
        amsr2_1km_data_stack = np.float32(amsr2_1km_data_stack)
        del(amsr2_1km_load)


# Subplot maps
# Load in watershed shapefile boundaries
shapefile_gb = fiona.open(path_shp_gb + '/' + shp_gb_file, 'r')
crop_shape_gb = [feature["geometry"] for feature in shapefile_gb]
shp_gb_extent = list(shapefile_gb.bounds)

output_crs = 'EPSG:4326'

# Subset and reproject the amsr2 SM data at watershed
# 1 km
masked_ds_gb_1km_all = []
for n in range(amsr2_1km_data_stack.shape[0]):
    sub_window_gb_1km = Window(col_gb_1km_ind[0], row_gb_1km_ind[0], len(col_gb_1km_ind), len(row_gb_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    amsr2_sm_gb_1km_output = sub_n_reproj(amsr2_1km_data_stack[n, :, :], kwargs_1km_sub, sub_window_gb_1km, output_crs)

    masked_ds_gb_1km, mask_transform_ds_gb_1km = mask(dataset=amsr2_sm_gb_1km_output, shapes=crop_shape_gb, crop=True)
    masked_ds_gb_1km[np.where(masked_ds_gb_1km == 0)] = np.nan
    masked_ds_gb_1km = masked_ds_gb_1km.squeeze()

    masked_ds_gb_1km_all.append(masked_ds_gb_1km)

masked_ds_gb_1km_all = np.asarray(masked_ds_gb_1km_all)


# 9 km
masked_ds_gb_9km_all = []
for n in range(amsr2_9km_data_stack.shape[0]):
    sub_window_gb_9km = Window(col_gb_9km_ind[0], row_gb_9km_ind[0], len(col_gb_9km_ind), len(row_gb_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    amsr2_sm_gb_9km_output = sub_n_reproj(amsr2_9km_data_stack[n, :, :], kwargs_9km_sub, sub_window_gb_9km, output_crs)

    masked_ds_gb_9km, mask_transform_ds_gb_9km = mask(dataset=amsr2_sm_gb_9km_output, shapes=crop_shape_gb, crop=True)
    masked_ds_gb_9km[np.where(masked_ds_gb_9km == 0)] = np.nan
    masked_ds_gb_9km = masked_ds_gb_9km.squeeze()

    masked_ds_gb_9km_all.append(masked_ds_gb_9km)

masked_ds_gb_9km_all = np.asarray(masked_ds_gb_9km_all)


# Make the subplot maps
title_content = ['1 km (04/01 - 04/08)', '10 km (04/01 - 04/08)', '1 km (04/09 - 04/16)', '10 km (04/09 - 04/16)',
              '1 km (04/17 - 04/24)', '10 km (04/17 - 04/24)']
feature_shp_gb = ShapelyFeature(Reader(path_shp_gb + '/' + shp_gb_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_gb = np.array(amsr2_sm_gb_1km_output.bounds)
extent_gb = extent_gb[[0, 2, 1, 3]]

columns = 2
rows = 3
fig = plt.figure(figsize=(11, 8), facecolor='w', edgecolor='k')
# fig = plt.figure(num=None, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(rows, columns, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_gb)
    ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_gb_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='gist_earth_r',
               extent=extent_gb)
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.5, labelpad=-0.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=3)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # 9 km
    ax = fig.add_subplot(rows, columns, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_gb)
    ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_gb_9km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='gist_earth_r',
               extent=extent_gb)
    cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.5, labelpad=-0.5)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=3)
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.92, hspace=0.35, wspace=0.3)
plt.savefig(path_results + '/amsr2_sm_comp_gb_1.png')


# #1.2.3.1 Murray-Darling RB
# path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
# os.chdir(path_shp_md)
# shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
# shp_md_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_md_file, 0)
# shp_md_extent = list(shp_md_ds.GetLayer().GetExtent())
#
# #Subset the region of Murray-Darling RB (9 km)
# [lat_9km_md, row_md_9km_ind, lon_9km_md, col_md_9km_ind] = \
#     coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_md_extent[3], shp_md_extent[2], shp_md_extent[1], shp_md_extent[0])
#
# # Load in SMAP 9 km SM of Murray-Darling RB
#
# year_plt = yearname[4]
# month_plt = 1
# days_begin = 1
# days_end = 24
# days_n = days_end - days_begin + 1
#
# for iyr in [year_plt]:  # range(yearname):
#     for imo in [month_plt]:  # range(len(monthname)):
#         hdf_file_smap_9km = path_procdata + '/smap_sm_9km_' + str(iyr) + monthname[imo-1] + '.hdf5'
#         f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
#         varname_list_smap_9km = list(f_read_smap_9km.keys())
#
#
# smap_9km_load = np.empty((len(row_md_9km_ind), len(col_md_9km_ind), days_n*2))
# smap_9km_load[:] = np.nan
# for idt in range(days_n):
#     smap_9km_load[:, :, 2*idt] = f_read_smap_9km[varname_list_smap_9km[0]][row_md_9km_ind[0]:row_md_9km_ind[-1]+1,
#         col_md_9km_ind[0]:col_md_9km_ind[-1]+1, days_begin-1+idt] # AM
#     smap_9km_load[:, :, 2*idt+1] = f_read_smap_9km[varname_list_smap_9km[1]][row_md_9km_ind[0]:row_md_9km_ind[-1]+1,
#         col_md_9km_ind[0]:col_md_9km_ind[-1]+1, days_begin-1+idt] # PM
# f_read_smap_9km.close()
#
# smap_9km_mean_1 = np.nanmean(smap_9km_load[:, :, :days_n//3*2], axis=2)
# smap_9km_mean_2 = np.nanmean(smap_9km_load[:, :, days_n//3*2:days_n//3*4], axis=2)
# smap_9km_mean_3 = np.nanmean(smap_9km_load[:, :, days_n//3*4:], axis=2)
# smap_9km_data_stack = np.stack((smap_9km_mean_1, smap_9km_mean_2, smap_9km_mean_3))
#
# del(smap_9km_load, smap_9km_mean_1, smap_9km_mean_2, smap_9km_mean_3)
#
#
# # Load in SMAP 1 km SM
#
# #Subset the region of Genga-Brahmaputra RB (1 km)
# [lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
#     coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_md_extent[3], shp_md_extent[2], shp_md_extent[1], shp_md_extent[0])
#
# smap_1km_load = np.empty((len(row_md_1km_ind), len(col_md_1km_ind), days_n*2))
# smap_1km_load[:] = np.nan
#
# for iyr in [year_plt]:  # range(yearname):
#     for imo in [month_plt]:  # range(len(monthname)):
#         for idt in range(days_n): # days
#             str_date = str(iyr) + '-' + monthname[imo-1] + '-' + str(days_begin-1+idt+1).zfill(2)
#             str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
#             tif_file_smap_1km = path_smap_sm_ds + '/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
#             src_tf = gdal.Open(tif_file_smap_1km)
#             src_tf_arr = src_tf.ReadAsArray()[:, row_md_1km_ind[0]:row_md_1km_ind[-1]+1,
#                          col_md_1km_ind[0]:col_md_1km_ind[-1]+1]
#             src_tf_arr = np.transpose(src_tf_arr, (1, 2, 0))
#             smap_1km_load[:, :, 2*idt] = src_tf_arr[:, :, 0]
#             smap_1km_load[:, :, 2*idt+1] = src_tf_arr[:, :, 1]
#
#             print(str_date)
#             del(src_tf_arr)
#
#
# smap_1km_mean_1 = np.nanmean(smap_1km_load[:, :, :days_n//3*2], axis=2)
# smap_1km_mean_2 = np.nanmean(smap_1km_load[:, :, days_n//3*2:days_n//3*4], axis=2)
# smap_1km_mean_3 = np.nanmean(smap_1km_load[:, :, days_n//3*4:], axis=2)
#
# smap_1km_data_stack = np.stack((smap_1km_mean_1, smap_1km_mean_2, smap_1km_mean_3))
# del(smap_1km_load)
#
#
# # Subplot maps
# # Build the map file
#
# map_md_9km = Basemap(projection='cea',llcrnrlat=lat_9km_md[-1],urcrnrlat=lat_9km_md[0],
#               llcrnrlon=lon_9km_md[0],urcrnrlon=lon_9km_md[-1],resolution='c')
# x_md_9km = np.linspace(0, map_md_9km.urcrnrx, len(lon_9km_md))
# y_md_9km = np.linspace(0, map_md_9km.urcrnry, len(lat_9km_md))
# y_md_9km = y_md_9km[::-1]
# xx_md_9km, yy_md_9km = np.meshgrid(x_md_9km, y_md_9km)
#
# map_md_1km = Basemap(projection='cea',llcrnrlat=lat_1km_md[-1],urcrnrlat=lat_1km_md[0],
#               llcrnrlon=lon_1km_md[0],urcrnrlon=lon_1km_md[-1],resolution='c')
# x_md_1km = np.linspace(0, map_md_1km.urcrnrx, len(lon_1km_md))
# y_md_1km = np.linspace(0, map_md_1km.urcrnry, len(lat_1km_md))
# y_md_1km = y_md_1km[::-1]
# xx_md_1km, yy_md_1km = np.meshgrid(x_md_1km, y_md_1km)
#
# plot_title = ['1 km (01/01 - 01/08)', '1 km (01/09 - 01/16)', '1 km (01/17 - 01/24)',
#               '9 km (01/01 - 01/08)', '9 km (01/09 - 01/16)', '9 km (01/17 - 01/24)']
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 7), facecolor='w', edgecolor='k')
# for ipt in range(3):
#     # 1 km
#     map_md_1km = Basemap(projection='cea', llcrnrlat=lat_1km_md[-1],urcrnrlat=lat_1km_md[0], ax=axes.flat[ipt],
#               llcrnrlon=lon_1km_md[0],urcrnrlon=lon_1km_md[-1], resolution='c')
#     map_md_1km.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1', 'GSHHS_f_L1', drawbounds=True)
#     map_md_1km.readshapefile(path_shp_md + '/Aqueduct_river_basins_MURRAY - DARLING', 'Aqueduct_river_basins_MURRAY - DARLING',
#                               drawbounds=True, linewidth=1.5)
#     map_md_mesh_1km = map_md_1km.pcolormesh(xx_md_1km, yy_md_1km, smap_1km_data_stack[ipt, :, :],
#                                             vmin=0, vmax=0.5, cmap='gist_earth_r')
#     map_md_1km.drawparallels(np.arange(lat_world_min, lat_world_max, 5), labels=[1, 1, 1, 1], linewidth=0.5)
#     map_md_1km.drawmeridians(np.arange(lon_world_min, lon_world_max, 5), labels=[1, 1, 1, 1], linewidth=0.5)
#     cbar = map_md_1km.colorbar(map_wrd_mesh, extend='both', location='right', pad='15%')
#     cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12)
#     cbar.ax.tick_params(labelsize=12)
#     axes.flat[ipt].set_title(plot_title[ipt], pad=20, fontsize=17, fontweight='bold')
#
#     # 9 km
#     map_md_9km = Basemap(projection='cea', llcrnrlat=lat_9km_md[-1],urcrnrlat=lat_9km_md[0], ax=axes.flat[ipt+3],
#               llcrnrlon=lon_9km_md[0],urcrnrlon=lon_9km_md[-1], resolution='c')
#     map_md_9km.readshapefile(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1', 'GSHHS_f_L1', drawbounds=True)
#     map_md_9km.readshapefile(path_shp_md + '/Aqueduct_river_basins_MURRAY - DARLING', 'Aqueduct_river_basins_MURRAY - DARLING',
#                               drawbounds=True, linewidth=1.5)
#     map_md_mesh_9km = map_md_9km.pcolormesh(xx_md_9km, yy_md_9km, smap_9km_data_stack[ipt, :, :],
#                                             vmin=0, vmax=0.5, cmap='gist_earth_r')
#     map_md_9km.drawparallels(np.arange(lat_world_min, lat_world_max, 5), labels=[1, 1, 1, 1], linewidth=0.5)
#     map_md_9km.drawmeridians(np.arange(lon_world_min, lon_world_max, 5), labels=[1, 1, 1, 1], linewidth=0.5)
#     cbar = map_md_9km.colorbar(map_wrd_mesh, extend='both', location='right', pad='15%')
#     cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12)
#     cbar.ax.tick_params(labelsize=12)
#     axes.flat[ipt+3].set_title(plot_title[ipt+3], pad=20, fontsize=17, fontweight='bold')
#
# plt.tight_layout()
# plt.show()
# plt.savefig(path_results + '/sm_comp_md_1.png')



########################################################################################################################
# 2. Scatter plots
# 2.1 Select geograhical locations by using index tables, and plot delta T - SM relationship lines through each NDVI class

# Lat/lon of the locations in the world:
# Solimoes, Amazon RB: -4.636, -70.266
# Tonzi Ranch, California RB: 38.432,-120.966
# Walnut Gulch, Colorado RB: 31.733, -110.05
# Banloc, Danube RB: 45.383, 21.133
# Krong Kracheh, Mekong: 12.49, 106.029
# Yanco, Murrumbidgee, Murray-Darling: -34.604, 146.41
# Nanchong, Yangtze: 30.838, 106.111
# Kanpur, Ganga-Brahmaputra: 26.521, 80.231

lat_slc = [-4.636, 38.432, 31.733, 45.383, 12.49, -34.604, 30.838, 26.521]
lon_slc = [-70.266, -120.966, -110.05, 21.133, 106.029, 146.41, 106.111, 80.231]
name_slc = ['Amazon', 'California', 'Colorado', 'Danube', 'Mekong', 'Murray-Darling', 'Yangtze', 'Ganga-Brahmaputra']
ndvi_class = np.linspace(0, 1, 11)
viridis_r = plt.cm.get_cmap('viridis_r', 10)

row_25km_ind_sub = []
col_25km_ind_sub = []
for ico in range(len(lat_slc)):
    row_dist = np.absolute(lat_slc[ico] - lat_world_ease_25km)
    row_match = np.argmin(row_dist)
    col_dist = np.absolute(lon_slc[ico] - lon_world_ease_25km)
    col_match = np.argmin(col_dist)
    # ind = np.intersect1d(row_match, col_match)[0]
    row_25km_ind_sub.append(row_match)
    col_25km_ind_sub.append(col_match)
    del(row_dist, row_match, col_dist, col_match)

row_25km_ind_sub = np.asarray(row_25km_ind_sub)
col_25km_ind_sub = np.asarray(col_25km_ind_sub)


hdf_file = path_procdata + '/ds_model_coef_nofill.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())

r_sq_all = []
rmse_all = []
for x in range(len(row_25km_ind_sub)):
    r_sq = f_read[varname_list[30]][row_25km_ind_sub[x], col_25km_ind_sub[x], ::2]
    rmse = f_read[varname_list[30]][row_25km_ind_sub[x], col_25km_ind_sub[x], 1::2]
    r_sq_all.append(r_sq)
    rmse_all.append(rmse)
r_sq_all = np.asarray(r_sq_all)
rmse_all = np.asarray(rmse_all)
metric_all = [r_sq_all, rmse_all]
metric_all = np.asarray(metric_all)
np.savetxt(path_results + '/scatterplots/' + 'regression_metric.csv', r_sq_all, delimiter=",", fmt='%f')

    # linear_ind = np.ravel_multi_index([stn_row_9km_ind_all[ire], stn_col_9km_ind_all[ire]],
    #                                   (var_obj.shape[0], var_obj.shape[1]))
    # var_obj = np.reshape(var_obj, (var_obj.shape[0] * var_obj.shape[1], var_obj.shape[2]))  # Convert from 3D to 2D
    # var_obj = var_obj[linear_ind, :]



# Load in data
os.chdir(path_procdata)
hdf_file = path_procdata + '/ds_model_07.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())

lst_am_delta = np.array([f_read[varname_list[0]][x, :] for x in coord_25km_ind])
lst_pm_delta = np.array([f_read[varname_list[1]][x, :] for x in coord_25km_ind])
ndvi = np.array([f_read[varname_list[2]][x, :] for x in coord_25km_ind])
sm_am = np.array([f_read[varname_list[3]][x, :] for x in coord_25km_ind])
sm_pm = np.array([f_read[varname_list[4]][x, :] for x in coord_25km_ind])


# Subplots of GLDAS SM vs. LST difference
# 2.1.1.
fig = plt.figure(figsize=(11, 6.5))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(1, 5):
    x = sm_am[i-1, :]
    y = lst_am_delta[i-1, :]
    c = ndvi[i-1, :]

    ax = fig.add_subplot(2, 2, i)
    sc = ax.scatter(x, y, c=c, marker='o', s=3, cmap='viridis_r')
    sc.set_clim(vmin=0,vmax=0.7)
    sc.set_label('NDVI')

    # Plot by NDVI classes
    for n in range(len(ndvi_class)-1):
        ndvi_ind = np.array([np.where((c>=ndvi_class[n]) & (c<ndvi_class[n+1]))[0]])
        if ndvi_ind.shape[1] > 10:
            coef, intr = np.polyfit(x[ndvi_ind].squeeze(), y[ndvi_ind].squeeze(), 1)
            ax.plot(x[ndvi_ind].squeeze(), intr + coef * x[ndvi_ind].squeeze(), '-', color=viridis_r.colors[n])
        else:
            pass

    plt.xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    plt.ylim(0, 40)
    ax.set_yticks(np.arange(0, 50, 10))
    ax.text(0.02, 5, name_slc[i-1],fontsize=15)
    plt.grid(linestyle='--')
    cbar = plt.colorbar(sc, extend='both')
    cbar.set_label('NDVI')

fig.text(0.5, 0.01, 'Soil Moisture ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.5, 'Temperature Difference (K)', va='center', rotation='vertical', fontsize=16, fontweight='bold')
plt.savefig(path_results + '/gldas_comp_1.png')


# 2.1.2.
fig = plt.figure(figsize=(11, 6.5))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
for i in range(5, 9):
    x = sm_am[i-1, :]
    y = lst_am_delta[i-1, :]
    c = ndvi[i-1, :]

    ax = fig.add_subplot(2, 2, i-4)
    sc = ax.scatter(x, y, c=c, marker='o', s=3, cmap='viridis_r')
    sc.set_clim(vmin=0,vmax=0.7)

    # Plot by NDVI classes
    for n in range(len(ndvi_class)-1):
        ndvi_ind = np.array([np.where((c>=ndvi_class[n]) & (c<ndvi_class[n+1]))[0]])
        if ndvi_ind.shape[1] > 10:
            coef, intr = np.polyfit(x[ndvi_ind].squeeze(), y[ndvi_ind].squeeze(), 1)
            ax.plot(x[ndvi_ind].squeeze(), intr + coef * x[ndvi_ind].squeeze(), '-', color=viridis_r.colors[n])
        else:
            pass

    plt.xlim(0.1, 0.5)
    ax.set_xticks(np.arange(0.1, 0.6, 0.1))
    plt.ylim(0, 30)
    ax.set_yticks(np.arange(0, 40, 10))
    ax.text(0.12, 25, name_slc[i-1],fontsize=15)
    plt.grid(linestyle='--')
    cbar = plt.colorbar(sc, extend='both')
    cbar.set_label('NDVI')

fig.text(0.5, 0.01, 'Soil Moisture ($\mathregular{m^3/m^3)}$', ha='center', fontsize=16, fontweight='bold')
fig.text(0.04, 0.5, 'Temperature Difference (K)', va='center', rotation='vertical', fontsize=16, fontweight='bold')
plt.savefig(path_results + '/gldas_comp_2.png')




# map_dan_9km_enum = list(enumerate(map_dan_9km.Aqueduct_river_basins_DANUBE))[0][1]
#
# # for nshape,seg in enumerate(map_dan_9km.borders):
# #     if nshape == 1873: #This nshape denotes the large continental body of the USA, which we want
# #         mainseg = seg
# mainpoly = Polygon(map_dan_9km_enum,facecolor='blue',edgecolor='k')
#
# nx, ny = 10, 10
# lons, lats = map_dan_9km.makegrid(nx, ny) # get lat/lons of ny by nx evenly space grid.
# x, y = map_dan_9km(lons, lats) # compute map proj coordinates.
# Z = np.zeros((nx,ny))
# Z[:] = np.NAN
#
# for i in np.arange(len(x)):
#     for j in np.arange(len(y)):
#         Z[i,j] = x[0,i]
#
# ax = plt.gca()
# im = ax.imshow(Z, cmap = plt.get_cmap('coolwarm'))
# im.set_clip_path(mainpoly)
# ax.add_patch(mainpoly)
# plt.show()