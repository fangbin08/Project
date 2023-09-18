import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import glob
import h5py
import calendar
from osgeo import ogr
import fiona
import itertools
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
# (Function 1) Subset the coordinates table of study area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind


########################################################################################################################
# (Function 2) Subset and reproject the Geotiff data to WGS84 projection

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
# 0. Preload variables
path_smap = '/Users/binfang/Downloads/Processing/smap_output/la_plata'
path_code = '/Users/binfang/Documents/SMAP_Project/smap_codes'
path_shp = '/Users/binfang/Downloads/Processing/shapefiles'
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_230116'
path_gpm = '/Volumes/Elements/Datasets/GPM'
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of SMAP output
path_output = '/Users/binfang/Downloads/Processing/smap_output'
path_modis = '/Volumes/Elements/Datasets/MODIS/Model_Input/MYD13A2'

yearname = np.linspace(2010, 2023, 14, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2010-01-01'
end_date = '2023-12-31'
year = 2023 - 2010 + 1

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
yearname = np.linspace(2010, 2023, 14, dtype='int')
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

# daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=0)

# Load in geo-location parameters
os.chdir(path_code)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_ease_1km', 'lon_world_ease_1km', 'lat_world_geo_10km', 'lon_world_geo_10km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


########################################################################################################################
# 1.1 Average the SMAP 1 km SM from daily to monthly
# time_step = 30

smap_file_list_all = []
for iyr in range(5, len(yearname)):
    # smap_file_path = path_output + '/lake_victoria/' + str(yearname[iyr])
    smap_file_path = path_output + '/FeatherBasin/1km/SMAP'
    smap_file_list = sorted(glob.glob(smap_file_path + '/' + str(yearname[iyr]) + '/*'))
    smap_file_list_all.append(smap_file_list)
smap_file_list_all = list(itertools.chain(*smap_file_list_all))

daysofmonth_seq_short = daysofmonth_seq[:, 5:]
monthly_seq = np.reshape(daysofmonth_seq_short, (1, -1), order='F')
monthly_seq = monthly_seq[:, 3:] # Remove the first 3 months in 2015
monthly_seq = monthly_seq[:, :-3] # Remove the last 3 months in 2022
monthly_seq = np.squeeze(monthly_seq)
monthly_seq[-1] = monthly_seq[-1] - 1
monthly_seq_cumsum = np.cumsum(monthly_seq)
monthly_seq_cumsum = np.concatenate(([0], monthly_seq_cumsum))

# Month range: Apr 2015 - Sept 2022
smap_file_list_avg_all = []
for idl in range(len(monthly_seq_cumsum)-1):
    smap_file_list_avg = smap_file_list_all[monthly_seq_cumsum[idl]:monthly_seq_cumsum[idl+1]]

    src_tf_all = []
    for ife in range(len(smap_file_list_avg)):
        src_tf = rasterio.open(smap_file_list_avg[ife]).read()
        src_tf_all.append(src_tf)
        del(src_tf)
    smap_file_list_avg = np.nanmean(np.nanmean(src_tf_all, axis=0), axis=0)

    smap_file_list_avg_all.append(smap_file_list_avg)
    print(idl)
    del(smap_file_list_avg, src_tf_all)

########################################################################################################################
# 1.2 Map 1km SMAP SM data

# Subset and reproject the SMAP SM data at watershed
shp_file = path_shp + '/canada/Aqueduct_river_basins_GRANDE RIVIERE/Aqueduct_river_basins_GRANDE RIVIERE.shp'
shp_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_file, 0)
shp_extent = list(shp_ds.GetLayer().GetExtent())
shapefile_read = fiona.open(shp_file, 'r')
crop_shape = [feature["geometry"] for feature in shapefile_read]
output_crs = 'EPSG:4326'
smap_sm_1day = rasterio.open(smap_file_list_all[0])


# Load and Clip SM data by the profiles of the river basin
masked_ds_1km_all = []
for imo in range(len(smap_file_list_avg_all)):
    sub_window_1km = Window(0, 0, smap_sm_1day.shape[1], smap_sm_1day.shape[0])
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': smap_sm_1day.shape[1],
                        'height': smap_sm_1day.shape[0], 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                        'transform': smap_sm_1day.transform}
    smap_sm_1km_output = sub_n_reproj(smap_file_list_avg_all[imo], kwargs_1km_sub, sub_window_1km, output_crs)

    masked_ds_1km, mask_transform_ds_1km = mask(dataset=smap_sm_1km_output, shapes=crop_shape, crop=True)
    masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
    masked_ds_1km = masked_ds_1km.squeeze()

    masked_ds_1km_all.append(masked_ds_1km)
    del(masked_ds_1km, smap_sm_1km_output)
    print(imo)


# smap_divide_ind = np.arange(0, (len(yearname)-4)*12, 12)[1:]
# smap_divide_ind = smap_divide_ind - 3
# masked_ds_1km_all_divide = np.split(masked_ds_1km_all, smap_divide_ind, axis=0)


#Make the subplot SM maps (monthly)
title_content = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
feature_shp = ShapelyFeature(Reader(path_shp + '/FeatherBasin.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_nc_shp = ShapelyFeature(Reader(path_shp + '/NorthComplex/NorthComplex_reprj.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_di_shp = ShapelyFeature(Reader(path_shp + '/Dixie/Dixie_subset.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_ca_shp = ShapelyFeature(Reader(path_shp + '/Camp/camp_subset.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

smap_empty = np.empty(masked_ds_1km_all[0].shape)
smap_empty[:] = np.nan
smap_divide_ind = np.arange(0, (len(yearname)-4)*12, 12)[1:]
masked_ds_1km_extend = list(([smap_empty for x in range(3)], masked_ds_1km_all, [smap_empty for x in range(3)]))
masked_ds_1km_extend = list(itertools.chain(*masked_ds_1km_extend))
masked_ds_1km = np.split(masked_ds_1km_extend, smap_divide_ind, axis=0)[:-1]


# Make the map of every 4 years
fig = plt.figure(figsize=(7, 10), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.08, right=0.88, bottom=0.05, top=0.95, hspace=0.1, wspace=0.0)
for irow in range(12):
    for icol in range(4):

        ax = fig.add_subplot(12, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp)
        ax.add_feature(feature_nc_shp, linewidth=0.2)
        ax.add_feature(feature_di_shp, linewidth=0.2)
        ax.add_feature(feature_ca_shp, linewidth=0.2)
        img = ax.imshow(masked_ds_1km[icol+4][irow, :, :], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                   extent=shp_extent)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.5)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 5}
        gl.ylabel_style = {'size': 5}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 11:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 11:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.9, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
for i in range(4):
    fig.text(0.15+i*0.2, 0.97, yearname[i+5+4], fontsize=10, fontweight='bold')
for i in range(12):
    fig.text(0.04, 0.9-i*0.075, title_content[i], fontsize=8, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/smap_sm_feather_2_new.png')
plt.close()


########################################################################################################################
# 3. GPM
#Load and subset the watershed region
[lat_10km, row_10km_ind, lon_10km, col_10km_ind] = \
    coordtable_subset(lat_world_geo_10km, lon_world_geo_10km,
                      shp_extent[3], shp_extent[2], shp_extent[1], shp_extent[0])

daysofmonth_seq_short = daysofmonth_seq[:, 13:]
monthly_seq_gpm = np.reshape(daysofmonth_seq_short, (1, -1), order='F')
# monthly_seq_gpm = monthly_seq_gpm[:, : -3] # Remove the last 3 months in 2022
monthly_seq_gpm = np.squeeze(monthly_seq_gpm)
monthly_seq_cumsum_gpm = np.cumsum(monthly_seq_gpm)
monthly_seq_cumsum_gpm = np.concatenate(([0], monthly_seq_cumsum_gpm))

gpm_precip_ext_all = []
for iyr in [13]:#range(5, len(yearname)):

    f_gpm = h5py.File(path_gpm + '/gpm_precip_' + str(yearname[iyr]) + '.hdf5', 'r')
    varname_list_gpm = list(f_gpm.keys())
    gpm_precip_ext = f_gpm[varname_list_gpm[0]][row_10km_ind[0]:row_10km_ind[-1]+1, col_10km_ind[0]:col_10km_ind[-1]+1, :]
    gpm_precip_ext_all.append(gpm_precip_ext)
    print(iyr)
    del(gpm_precip_ext)

# gpm_precip_ext_all[-1] = gpm_precip_ext_all[-1][:, :, :273]
gpm_precip_ext_array = np.concatenate(gpm_precip_ext_all, axis=2)

with h5py.File(path_output + '/la_grande/gpm_prec.hdf5', 'w') as f:
    f.create_dataset('gpm_precip_ext_array', data=gpm_precip_ext_array)
f.close()

# Load in 1 km / 9 km matched data for validation
with h5py.File(path_output + '/la_grande/gpm_prec.hdf5', 'r') as f:
    var_name = list(f.keys())
    gpm_precip_ext_array = f[list(f.keys())[0]][()]
    f.close()
del(var_name)


# Month range: Apr 2015 - Dec 2022
gpm_file_list_avg_all = []
for idl in range(len(monthly_seq_cumsum_gpm)-5): # First 8 months of 2023
    gpm_file_list_avg = np.nansum(gpm_precip_ext_array[:, :, monthly_seq_cumsum_gpm[idl]:monthly_seq_cumsum_gpm[idl+1]], axis=2)
    gpm_file_list_avg_all.append(gpm_file_list_avg)
    del(gpm_file_list_avg)


# Load and Clip GPM data by the profiles of the river basin
masked_ds_1km_all_gpm = []
for imo in range(len(gpm_file_list_avg_all)):
    sub_window_1km = Window(0, 0, len(lon_10km), len(lat_10km))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_10km),
                      'height': len(lat_10km), 'count': 1, 'crs': CRS.from_dict(init='epsg:4326'),
                      'transform': Affine(0.1, 0.0, lon_10km[0]-0.05, 0.0, -0.1, lat_10km[0]+0.05)}
    gpm_sm_1km_output = sub_n_reproj(gpm_file_list_avg_all[imo], kwargs_1km_sub, sub_window_1km, output_crs)

    masked_ds_1km, mask_transform_ds_1km = mask(dataset=gpm_sm_1km_output, shapes=crop_shape, crop=True)
    masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
    masked_ds_1km = masked_ds_1km.squeeze()

    masked_ds_1km_all_gpm.append(masked_ds_1km)
    del(masked_ds_1km, gpm_sm_1km_output)
    print(imo)


#Make the subplot GPM precip maps (monthly)
title_content = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
# feature_shp = ShapelyFeature(Reader(path_shp + '/FeatherBasin.shp').geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black', facecolor='none')
# feature_nc_shp = ShapelyFeature(Reader(path_shp + '/NorthComplex/NorthComplex_reprj.shp').geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black', facecolor='none')
# feature_di_shp = ShapelyFeature(Reader(path_shp + '/Dixie/Dixie_subset.shp').geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black', facecolor='none')
# feature_ca_shp = ShapelyFeature(Reader(path_shp + '/Camp/camp_subset.shp').geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_shp = ShapelyFeature(Reader(path_shp + '/canada/Aqueduct_river_basins_GRANDE RIVIERE/Aqueduct_river_basins_GRANDE RIVIERE.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

# gpm_empty = np.empty(masked_ds_1km_all_gpm[0].shape)
# gpm_empty[:] = np.nan
# gpm_divide_ind = np.arange(0, (len(yearname)-4)*12, 12)[1:]
# masked_ds_1km_gpm_extend = list((masked_ds_1km_all_gpm, [gpm_empty for x in range(3)]))
# masked_ds_1km_gpm_extend = list(itertools.chain(*masked_ds_1km_gpm_extend))
# masked_ds_1km_gpm = np.split(masked_ds_1km_gpm_extend, gpm_divide_ind, axis=0)[:-1]

# Make the GPM map of every 4 years
fig = plt.figure(figsize=(10, 6), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.08, right=0.88, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)
for irow in range(4):
    for icol in range(2):

        ax = fig.add_subplot(4, 2, irow*2+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp)
        # ax.add_feature(feature_nc_shp, linewidth=0.2)
        # ax.add_feature(feature_di_shp, linewidth=0.2)
        # ax.add_feature(feature_ca_shp, linewidth=0.2)
        img = ax.imshow(masked_ds_1km_all_gpm[irow*2+icol], origin='upper', vmin=0, vmax=72, cmap='Blues',
                   extent=shp_extent)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=1)
        gl.xlabel_style = {'size': 5}
        gl.ylabel_style = {'size': 5}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        ax.title.set_text(title_content[irow*2+icol])
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('mm', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
# for i in range(4):
#     fig.text(0.15+i*0.2, 0.97, yearname[i+5+4], fontsize=10, fontweight='bold')
# for i in range(12):
#     fig.text(0.04, 0.9-i*0.075, title_content[i], fontsize=8, fontweight='bold', rotation=90)
plt.savefig(path_results + '/gpm_prec_la_grande.png')
plt.close()


########################################################################################################################
# 4. MODIS NDVI
# shp_file = path_shp + '/FeatherBasin.shp'
# shp_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_file, 0)
# shp_extent = list(shp_ds.GetLayer().GetExtent())

lat_sub_max = shp_extent[3]
lat_sub_min = shp_extent[2]
lon_sub_max = shp_extent[1]
lon_sub_min = shp_extent[0]

output_crs = 'EPSG:4326'
[lat_sub_ease_1km, row_sub_ease_1km_ind, lon_sub_ease_1km, col_sub_ease_1km_ind] = coordtable_subset\
    (lat_world_ease_1km, lon_world_ease_1km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)


modis_file_list_all = []
for iyr in range(5, len(yearname)):
    # modis_file_path = path_output + '/lake_victoria/' + str(yearname[iyr])
    modis_file_path = path_output + '/FeatherBasin/1km_NDVI'
    modis_file_list = sorted(glob.glob(modis_file_path + '/' + str(yearname[iyr]) + '/*'))
    modis_file_list_all.append(modis_file_list)
modis_file_list_all = list(itertools.chain(*modis_file_list_all))

doy_name = modis_file_list_all[0:23]
doy_name = [os.path.basename(doy_name[x]).split('.')[0][-7:] for x in range(len(doy_name))]
doy_name_month = np.array([(datetime.datetime(2015, 1, 1) + datetime.timedelta(int(doy_name[x][-3:]) - 1)).month for x in range(len(doy_name))])
doy_name_month_unique, doy_name_month_ind = np.unique(doy_name_month, return_index=True)
doy_name_month_ind_all = [doy_name_month_ind + x*23 for x in range(len(yearname)-5)]
doy_name_month_ind_all[7] = doy_name_month_ind_all[7][:10]
monthly_seq_cumsum = list(itertools.chain(*doy_name_month_ind_all))


# daysofmonth_seq_short = daysofmonth_seq[:, 5:]
# monthly_seq = np.reshape(daysofmonth_seq_short, (1, -1), order='F')
# monthly_seq = monthly_seq[:, 3:] # Remove the first 3 months in 2015
# monthly_seq = monthly_seq[:, :-3] # Remove the last 3 months in 2022
# monthly_seq = np.squeeze(monthly_seq)
# monthly_seq[-1] = monthly_seq[-1] - 1
# monthly_seq_cumsum = np.cumsum(monthly_seq)
# monthly_seq_cumsum = np.concatenate(([0], monthly_seq_cumsum))

# Month range: Jan 2015 - Dec 2022
modis_file_list_avg_all = []
for idl in range(len(monthly_seq_cumsum)-1):
    modis_file_list_avg = modis_file_list_all[monthly_seq_cumsum[idl]:monthly_seq_cumsum[idl+1]]

    src_tf_all = []
    for ife in range(len(modis_file_list_avg)):
        src_tf = rasterio.open(modis_file_list_avg[ife]).read()
        src_tf_all.append(src_tf)
        del(src_tf)
    modis_file_list_avg = np.nanmean(np.nanmean(src_tf_all, axis=0), axis=0)

    modis_file_list_avg_all.append(modis_file_list_avg)
    print(idl)
    del(modis_file_list_avg, src_tf_all)

########################################################################################################################
# 1.2 Map 1km modis ndvi data

# Subset and reproject the modis ndvi data at watershed
# shp_file = path_shp + '/FeatherBasin.shp'
# shp_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_file, 0)
# shp_extent = list(shp_ds.GetLayer().GetExtent())
# shapefile_read = fiona.open(shp_file, 'r')
# crop_shape = [feature["geometry"] for feature in shapefile_read]
# output_crs = 'EPSG:4326'
modis_ndvi_1day = rasterio.open(modis_file_list_all[0])


# Load and Clip ndvi data by the profiles of the river basin
masked_ds_1km_all_modis = []
for imo in range(len(modis_file_list_avg_all)):
    sub_window_1km = Window(0, 0, modis_ndvi_1day.shape[1], modis_ndvi_1day.shape[0])
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': modis_ndvi_1day.shape[1],
                        'height': modis_ndvi_1day.shape[0], 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                        'transform': modis_ndvi_1day.transform}
    modis_ndvi_1km_output = sub_n_reproj(modis_file_list_avg_all[imo], kwargs_1km_sub, sub_window_1km, output_crs)

    masked_ds_1km_modis, mask_transform_ds_1km = mask(dataset=modis_ndvi_1km_output, shapes=crop_shape, crop=True)
    masked_ds_1km_modis[np.where(masked_ds_1km_modis == 0)] = np.nan
    masked_ds_1km_modis = masked_ds_1km_modis.squeeze()

    masked_ds_1km_all_modis.append(masked_ds_1km_modis)
    del(masked_ds_1km_modis, modis_ndvi_1km_output)
    print(imo)


# modis_divide_ind = np.arange(0, (len(yearname)-4)*12, 12)[1:]
# modis_divide_ind = modis_divide_ind - 3
# masked_ds_1km_all_divide = np.split(masked_ds_1km_all, modis_divide_ind, axis=0)


#Make the subplot ndvi maps (monthly)
title_content = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
feature_shp = ShapelyFeature(Reader(path_shp + '/FeatherBasin.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_nc_shp = ShapelyFeature(Reader(path_shp + '/NorthComplex/NorthComplex_reprj.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_di_shp = ShapelyFeature(Reader(path_shp + '/Dixie/Dixie_subset.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_ca_shp = ShapelyFeature(Reader(path_shp + '/Camp/camp_subset.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
modis_empty = np.empty(masked_ds_1km_all_modis[0].shape)
modis_empty[:] = np.nan
modis_divide_ind = np.arange(0, (len(yearname)-4)*12, 12)[1:]
masked_ds_1km_modis_extend = list((masked_ds_1km_all_modis, [modis_empty for x in range(3)]))
masked_ds_1km_modis_extend = list(itertools.chain(*masked_ds_1km_modis_extend))
masked_ds_1km_modis = np.split(masked_ds_1km_modis_extend, modis_divide_ind, axis=0)[:-1]


# Make the map of every 4 years
fig = plt.figure(figsize=(7, 10), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.08, right=0.88, bottom=0.05, top=0.95, hspace=0.1, wspace=0.0)
for irow in range(12):
    for icol in range(4):

        ax = fig.add_subplot(12, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp)
        ax.add_feature(feature_nc_shp, linewidth=0.2)
        ax.add_feature(feature_di_shp, linewidth=0.2)
        ax.add_feature(feature_ca_shp, linewidth=0.2)
        img = ax.imshow(masked_ds_1km_modis[icol][irow, :, :], origin='upper', vmin=0, vmax=1, cmap='BrBG',
                   extent=shp_extent)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.5)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 5}
        gl.ylabel_style = {'size': 5}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 11:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 11:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.9, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
for i in range(4):
    fig.text(0.15+i*0.2, 0.97, yearname[i+5], fontsize=10, fontweight='bold')
for i in range(12):
    fig.text(0.04, 0.9-i*0.075, title_content[i], fontsize=8, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/modis_ndvi_feather_1_new.png')
plt.close()

########################################################################################################################
# 5. Make the subplot maps by each fire event

# Camp Fire
# November 8, 2018 – November 25, 2018
# 8-11, 2018
#
# North Complex Fire
# August 17, 2020 – December 3, 2020
# 8-12, 2020
#
# Dixie Fire
# July 13, 2021 – October 25, 2021
# 7-10, 2021

title_content = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
var_name = ['SMAP SM', 'GPM PCPN', 'MODIS NDVI']
shp_file = path_shp + '/FeatherBasin.shp'
shp_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_file, 0)
shp_extent = list(shp_ds.GetLayer().GetExtent())
shapefile_read = fiona.open(shp_file, 'r')
crop_shape = [feature["geometry"] for feature in shapefile_read]
output_crs = 'EPSG:4326'

feature_shp = ShapelyFeature(Reader(path_shp + '/FeatherBasin.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_nc_shp = ShapelyFeature(Reader(path_shp + '/NorthComplex/NorthComplex_reprj.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_di_shp = ShapelyFeature(Reader(path_shp + '/Dixie/Dixie_subset.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
feature_ca_shp = ShapelyFeature(Reader(path_shp + '/Camp/camp_subset.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')



# Camp Fire
masked_cf_data = list((masked_ds_1km_all[40:44], masked_ds_1km_all_gpm[43:47], masked_ds_1km_all_modis[43:47]))

fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
for irow in range(4):
    for icol in range(3):

        ax = fig.add_subplot(4, 3, irow*3+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp)
        ax.add_feature(feature_ca_shp, edgecolor='red')
        if icol == 0:
            img1 = ax.imshow(masked_cf_data[icol][irow], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                       extent=shp_extent)
        elif icol == 1:
            img2 = ax.imshow(masked_cf_data[icol][irow], origin='upper', vmin=0, vmax=20, cmap='Blues',
                       extent=shp_extent)
        else:
            img3 = ax.imshow(masked_cf_data[icol][irow], origin='upper', vmin=0, vmax=1, cmap='BrBG',
                       extent=shp_extent)

        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.5)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False

cbar_ax1 = fig.add_axes([0.14, 0.06, 0.2, 0.01])
cbar1 = fig.colorbar(img1, cax=cbar_ax1, extend='both', orientation='horizontal')
cbar1.ax.locator_params(nbins=4)
cbar1.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=1)
cbar_ax2 = fig.add_axes([0.43, 0.06, 0.2, 0.01])
cbar2 = fig.colorbar(img2, cax=cbar_ax2, extend='both', orientation='horizontal')
cbar2.ax.locator_params(nbins=4)
cbar2.set_label('(mm)', fontsize=10, x=0.95, labelpad=1)
cbar_ax3 = fig.add_axes([0.73, 0.06, 0.2, 0.01])
cbar3 = fig.colorbar(img3, cax=cbar_ax3, extend='both', orientation='horizontal')
cbar3.ax.locator_params(nbins=4)

for i in range(3):
    fig.text(0.18+i*0.285, 0.96, var_name[i], fontsize=11, fontweight='bold')
for i in range(4):
    fig.text(0.03, 0.84-i*0.225, title_content[i+7] + "'18", fontsize=11, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/camp_fire_new_1.png')
plt.close()


# Camp Fire (zoomed-in)
masked_cf_data_subbasin = list((masked_ds_1km_all_subbasin[0][40:44], masked_ds_1km_all_gpm_subbasin[0][43:47],
                                masked_ds_1km_all_modis_subbasin[0][43:47]))

fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
for irow in range(4):
    for icol in range(3):

        ax = fig.add_subplot(4, 3, irow*3+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp)
        ax.add_feature(feature_ca_shp, edgecolor='red')
        if icol == 0:
            img1 = ax.imshow(masked_cf_data_subbasin[icol][irow], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                       extent=shp_extent_all[0])
        elif icol == 1:
            img2 = ax.imshow(masked_cf_data_subbasin[icol][irow], origin='upper', vmin=0, vmax=20, cmap='Blues',
                       extent=shp_extent_all[0])
        else:
            img3 = ax.imshow(masked_cf_data_subbasin[icol][irow], origin='upper', vmin=0, vmax=1, cmap='BrBG',
                       extent=shp_extent_all[0])

        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.1)
        gl.ylocator = mticker.MultipleLocator(base=0.1)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False

cbar_ax1 = fig.add_axes([0.14, 0.06, 0.2, 0.01])
cbar1 = fig.colorbar(img1, cax=cbar_ax1, extend='both', orientation='horizontal')
cbar1.ax.locator_params(nbins=4)
cbar1.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=1)
cbar_ax2 = fig.add_axes([0.43, 0.06, 0.2, 0.01])
cbar2 = fig.colorbar(img2, cax=cbar_ax2, extend='both', orientation='horizontal')
cbar2.ax.locator_params(nbins=4)
cbar2.set_label('(mm)', fontsize=10, x=0.95, labelpad=1)
cbar_ax3 = fig.add_axes([0.73, 0.06, 0.2, 0.01])
cbar3 = fig.colorbar(img3, cax=cbar_ax3, extend='both', orientation='horizontal')
cbar3.ax.locator_params(nbins=4)

for i in range(3):
    fig.text(0.18+i*0.285, 0.96, var_name[i], fontsize=11, fontweight='bold')
for i in range(4):
    fig.text(0.03, 0.84-i*0.225, title_content[i+7] + "'18", fontsize=11, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/camp_fire_new_2.png')
plt.close()



# North Complex Fire
masked_ncf_data = list((masked_ds_1km_all[64:69], masked_ds_1km_all_gpm[67:72], masked_ds_1km_all_modis[67:72]))

fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
for irow in range(5):
    for icol in range(3):

        ax = fig.add_subplot(5, 3, irow*3+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp)
        ax.add_feature(feature_nc_shp, edgecolor='red')
        if icol == 0:
            img1 = ax.imshow(masked_ncf_data[icol][irow], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                       extent=shp_extent)
        elif icol == 1:
            img2 = ax.imshow(masked_ncf_data[icol][irow], origin='upper', vmin=0, vmax=20, cmap='Blues',
                       extent=shp_extent)
        else:
            img3 = ax.imshow(masked_ncf_data[icol][irow], origin='upper', vmin=0, vmax=1, cmap='BrBG',
                       extent=shp_extent)

        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.5)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 4:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 4:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False

cbar_ax1 = fig.add_axes([0.14, 0.06, 0.2, 0.01])
cbar1 = fig.colorbar(img1, cax=cbar_ax1, extend='both', orientation='horizontal')
cbar1.ax.locator_params(nbins=4)
cbar1.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=1)
cbar_ax2 = fig.add_axes([0.43, 0.06, 0.2, 0.01])
cbar2 = fig.colorbar(img2, cax=cbar_ax2, extend='both', orientation='horizontal')
cbar2.ax.locator_params(nbins=4)
cbar2.set_label('(mm)', fontsize=10, x=0.95, labelpad=1)
cbar_ax3 = fig.add_axes([0.73, 0.06, 0.2, 0.01])
cbar3 = fig.colorbar(img3, cax=cbar_ax3, extend='both', orientation='horizontal')
cbar3.ax.locator_params(nbins=4)

for i in range(3):
    fig.text(0.18+i*0.285, 0.97, var_name[i], fontsize=11, fontweight='bold')
for i in range(5):
    fig.text(0.03, 0.84-i*0.17, title_content[i+7] + "'20", fontsize=11, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/north_camp_fire_new_1.png')
plt.close()

# North Complex Fire (Zoomed-in)
masked_ncf_data = list((masked_ds_1km_all_subbasin[1][64:69], masked_ds_1km_all_gpm_subbasin[1][67:72], masked_ds_1km_all_modis_subbasin[1][67:72]))

fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
for irow in range(5):
    for icol in range(3):

        ax = fig.add_subplot(5, 3, irow*3+icol+1, projection=ccrs.PlateCarree())
        # ax.add_feature(feature_shp)
        ax.add_feature(feature_nc_shp, edgecolor='red')
        if icol == 0:
            img1 = ax.imshow(masked_ncf_data[icol][irow], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                       extent=shp_extent_all[1])
        elif icol == 1:
            img2 = ax.imshow(masked_ncf_data[icol][irow], origin='upper', vmin=0, vmax=20, cmap='Blues',
                       extent=shp_extent_all[1])
        else:
            img3 = ax.imshow(masked_ncf_data[icol][irow], origin='upper', vmin=0, vmax=1, cmap='BrBG',
                       extent=shp_extent_all[1])

        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.3)
        gl.ylocator = mticker.MultipleLocator(base=0.3)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 4:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 4:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False

cbar_ax1 = fig.add_axes([0.14, 0.06, 0.2, 0.01])
cbar1 = fig.colorbar(img1, cax=cbar_ax1, extend='both', orientation='horizontal')
cbar1.ax.locator_params(nbins=4)
cbar1.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=1)
cbar_ax2 = fig.add_axes([0.43, 0.06, 0.2, 0.01])
cbar2 = fig.colorbar(img2, cax=cbar_ax2, extend='both', orientation='horizontal')
cbar2.ax.locator_params(nbins=4)
cbar2.set_label('(mm)', fontsize=10, x=0.95, labelpad=1)
cbar_ax3 = fig.add_axes([0.73, 0.06, 0.2, 0.01])
cbar3 = fig.colorbar(img3, cax=cbar_ax3, extend='both', orientation='horizontal')
cbar3.ax.locator_params(nbins=4)

for i in range(3):
    fig.text(0.18+i*0.285, 0.97, var_name[i], fontsize=11, fontweight='bold')
for i in range(5):
    fig.text(0.03, 0.84-i*0.17, title_content[i+7] + "'20", fontsize=11, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/north_camp_fire_new_2.png')
plt.close()

# Dixie Fire
masked_df_data = list((masked_ds_1km_all[76:80], masked_ds_1km_all_gpm[79:83], masked_ds_1km_all_modis[79:83]))

fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
for irow in range(4):
    for icol in range(3):

        ax = fig.add_subplot(4, 3, irow*3+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp)
        ax.add_feature(feature_di_shp, edgecolor='red')
        if icol == 0:
            img1 = ax.imshow(masked_df_data[icol][irow], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                       extent=shp_extent)
        elif icol == 1:
            img2 = ax.imshow(masked_df_data[icol][irow], origin='upper', vmin=0, vmax=80, cmap='Blues',
                       extent=shp_extent)
        else:
            img3 = ax.imshow(masked_df_data[icol][irow], origin='upper', vmin=0, vmax=1, cmap='BrBG',
                       extent=shp_extent)

        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.5)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False

cbar_ax1 = fig.add_axes([0.14, 0.06, 0.2, 0.01])
cbar1 = fig.colorbar(img1, cax=cbar_ax1, extend='both', orientation='horizontal')
cbar1.ax.locator_params(nbins=4)
cbar1.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=1)
cbar_ax2 = fig.add_axes([0.43, 0.06, 0.2, 0.01])
cbar2 = fig.colorbar(img2, cax=cbar_ax2, extend='both', orientation='horizontal')
cbar2.ax.locator_params(nbins=4)
cbar2.set_label('(mm)', fontsize=10, x=0.95, labelpad=1)
cbar_ax3 = fig.add_axes([0.73, 0.06, 0.2, 0.01])
cbar3 = fig.colorbar(img3, cax=cbar_ax3, extend='both', orientation='horizontal')
cbar3.ax.locator_params(nbins=4)

for i in range(3):
    fig.text(0.18+i*0.285, 0.96, var_name[i], fontsize=11, fontweight='bold')
for i in range(4):
    fig.text(0.03, 0.84-i*0.22, title_content[i+7] + "'21", fontsize=11, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/dixie_fire_new_1.png')
plt.close()


# Dixie Fire (Zoomed-in)
masked_df_data = list((masked_ds_1km_all_subbasin[2][76:80], masked_ds_1km_all_gpm_subbasin[2][79:83], masked_ds_1km_all_modis_subbasin[2][79:83]))

fig = plt.figure(figsize=(12, 8), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.12, right=0.95, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
for irow in range(4):
    for icol in range(3):

        ax = fig.add_subplot(4, 3, irow*3+icol+1, projection=ccrs.PlateCarree())
        # ax.add_feature(feature_shp)
        ax.add_feature(feature_di_shp, edgecolor='red')
        if icol == 0:
            img1 = ax.imshow(masked_df_data[icol][irow], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                       extent=shp_extent_all[2])
        elif icol == 1:
            img2 = ax.imshow(masked_df_data[icol][irow], origin='upper', vmin=0, vmax=80, cmap='Blues',
                       extent=shp_extent_all[2])
        else:
            img3 = ax.imshow(masked_df_data[icol][irow], origin='upper', vmin=0, vmax=1, cmap='BrBG',
                       extent=shp_extent_all[2])

        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.5)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False

cbar_ax1 = fig.add_axes([0.14, 0.06, 0.2, 0.01])
cbar1 = fig.colorbar(img1, cax=cbar_ax1, extend='both', orientation='horizontal')
cbar1.ax.locator_params(nbins=4)
cbar1.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95, labelpad=1)
cbar_ax2 = fig.add_axes([0.43, 0.06, 0.2, 0.01])
cbar2 = fig.colorbar(img2, cax=cbar_ax2, extend='both', orientation='horizontal')
cbar2.ax.locator_params(nbins=4)
cbar2.set_label('(mm)', fontsize=10, x=0.95, labelpad=1)
cbar_ax3 = fig.add_axes([0.73, 0.06, 0.2, 0.01])
cbar3 = fig.colorbar(img3, cax=cbar_ax3, extend='both', orientation='horizontal')
cbar3.ax.locator_params(nbins=4)

for i in range(3):
    fig.text(0.18+i*0.285, 0.96, var_name[i], fontsize=11, fontweight='bold')
for i in range(4):
    fig.text(0.03, 0.84-i*0.22, title_content[i+7] + "'21", fontsize=11, fontweight='bold', rotation=90)
plt.savefig(path_results + '/feather_basin/dixie_fire_new_2.png')
plt.close()



########################################################################################################################
# 6. Make Monthly time-series plot

# tif_files = sorted(glob.glob(path_smap + '/la_plata_10km_prec_monthly/' + '*.tif'))
# gpm_monthly_all = []
# for ife in range(len(tif_files)):
#     gpm_monthly = rasterio.open(tif_files[ife]).read()
#     gpm_monthly_all.append(gpm_monthly)
#     del(gpm_monthly)
#     print(tif_files[ife])
# gpm_monthly_all = np.stack(gpm_monthly_all, axis=0).squeeze()
#
# gpm_monthly_all_avg = np.nanmean(np.nanmean(gpm_monthly_all, axis=1), axis=1)

masked_ds_1km_smap_avg = np.nanmean(np.nanmean(np.array(masked_ds_1km_extend), axis=1), axis=1)
masked_ds_1km_gpm_avg = np.nanmean(np.nanmean(np.array(masked_ds_1km_gpm_extend), axis=1), axis=1)
masked_ds_1km_modis_avg = np.nanmean(np.nanmean(np.array(masked_ds_1km_modis_extend), axis=1), axis=1)

fig = plt.figure(figsize=(15, 4), dpi=200)
plt.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.85, hspace=0.3, wspace=0.25)
x = masked_ds_1km_smap_avg
z = masked_ds_1km_gpm_avg
ax = fig.add_subplot(1, 1, 1)
lns1 = ax.plot(x, c='r', marker='s', label='SMAP', markersize=3, linestyle='--', linewidth=1)
# ax.axvspan(1.25, 1.55, facecolor='g', alpha=0.5)
# lns2 = ax.plot(x, c='k', marker='s', label='MODIS', markersize=3, linestyle='--', linewidth=1)
plt.show()
plt.xlim(0, len(x))
ax.set_xticks(np.arange(0, len(masked_ds_1km_smap_avg)+12, 12))
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
mticks = ax.get_xticks()
ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xticklabels(labels, minor=True)

plt.ylim(0, 0.5)
ax.set_yticks(np.arange(0, 0.6, 0.1))
ax.tick_params(axis='y', labelsize=10)
ax.grid(linestyle='--')

ax2 = ax.twinx()
ax2.set_ylim(0, 30)
ax2.invert_yaxis()
ax2.set_yticks(np.arange(0, 95, 16))
lns3 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
ax2.tick_params(axis='y', labelsize=10)
handles = lns1 + [lns3]
labels = [l.get_label() for l in handles]

plt.legend(handles, labels, loc=(0, 1.01), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
fig.text(0.5, 0.01, 'Year', ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.2, 'SMAP SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=14, fontweight='bold')
fig.text(0.96, 0.2, 'GPM Precipitation (mm)', rotation='vertical', fontsize=14, fontweight='bold')
plt.suptitle('Feather Basin', fontsize=16, y=0.98, fontweight='bold')
plt.savefig(path_results + '/smap_gpm_tseries' + '.png')
plt.close(fig)


fig = plt.figure(figsize=(15, 4), dpi=200)
plt.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.85, hspace=0.3, wspace=0.25)
x = masked_ds_1km_modis_avg
z = masked_ds_1km_gpm_avg
ax = fig.add_subplot(1, 1, 1)
lns1 = ax.plot(x, c='k', marker='s', label='MODIS NDVI', markersize=3, linestyle='--', linewidth=1)
# lns2 = ax.plot(x, c='k', marker='s', label='MODIS', markersize=3, linestyle='--', linewidth=1)
plt.show()
plt.xlim(0, len(x))
ax.set_xticks(np.arange(0, len(masked_ds_1km_smap_avg)+12, 12))
ax.set_xticklabels([])
labels = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
mticks = ax.get_xticks()
ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
ax.tick_params(axis='x', which='minor', length=0)
ax.set_xticklabels(labels, minor=True)

plt.ylim(0, 0.5)
ax.set_yticks(np.arange(0, 1.2, 0.2))
ax.tick_params(axis='y', labelsize=10)
ax.grid(linestyle='--')

ax2 = ax.twinx()
ax2.set_ylim(0, 30)
ax2.invert_yaxis()
ax2.set_yticks(np.arange(0, 95, 16))
lns3 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
ax2.tick_params(axis='y', labelsize=10)
handles = lns1 + [lns3]
labels = [l.get_label() for l in handles]

plt.legend(handles, labels, loc=(0, 1.01), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
fig.text(0.5, 0.01, 'Year', ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.2, 'MODIS NDVI', rotation='vertical', fontsize=14, fontweight='bold')
fig.text(0.96, 0.2, 'GPM Precipitation (mm)', rotation='vertical', fontsize=14, fontweight='bold')
plt.suptitle('Feather Basin', fontsize=16, y=0.98, fontweight='bold')
plt.savefig(path_results + '/gpm_ndvi_tseries' + '.png')
plt.close(fig)




########################################################################################################################
# 6.1 Time-series plot (by subbasin)

shp_file_ca = path_shp + '/Camp/camp_subset.shp'
shp_file_nc = path_shp + '/NorthComplex/NorthComplex_reprj.shp'
shp_file_di = path_shp + '/Dixie/Dixie_subset.shp'
shp_file_subbasin = list((shp_file_ca, shp_file_nc, shp_file_di))

crop_shape_all = []
shp_extent_all = []
for i in range(len(shp_file_subbasin)):
    shp_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_file_subbasin[i], 0)
    shapefile_read = fiona.open(shp_file_subbasin[i], 'r')
    shp_extent = list(shp_ds.GetLayer().GetExtent())
    crop_shape = [feature["geometry"] for feature in shapefile_read]
    crop_shape_all.append(crop_shape)
    shp_extent_all.append(shp_extent)
    del(shp_extent, crop_shape)

output_crs = 'EPSG:4326'
smap_sm_1day = rasterio.open(smap_file_list_all[0])

# SMAP
masked_ds_1km_all_subbasin = []
for ife in range(len(crop_shape_all)):
    masked_ds_1km_all_1basin = []
    for imo in range(len(smap_file_list_avg_all)):
        sub_window_1km = Window(0, 0, smap_sm_1day.shape[1], smap_sm_1day.shape[0])
        kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': smap_sm_1day.shape[1],
                            'height': smap_sm_1day.shape[0], 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                            'transform': smap_sm_1day.transform}
        smap_sm_1km_output = sub_n_reproj(smap_file_list_avg_all[imo], kwargs_1km_sub, sub_window_1km, output_crs)

        masked_ds_1km, mask_transform_ds_1km = mask(dataset=smap_sm_1km_output, shapes=crop_shape_all[ife], crop=True)
        masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
        masked_ds_1km = masked_ds_1km.squeeze()

        masked_ds_1km_all_1basin.append(masked_ds_1km)
        del(masked_ds_1km, smap_sm_1km_output)
    print(ife)
    masked_ds_1km_all_subbasin.append(masked_ds_1km_all_1basin)
    del(masked_ds_1km_all_1basin)


# GPM
masked_ds_1km_all_gpm_subbasin = []
for ife in range(len(crop_shape_all)):
    masked_ds_1km_all_gpm_1basin = []
    for imo in range(len(gpm_file_list_avg_all)):
        sub_window_1km = Window(0, 0, len(lon_10km), len(lat_10km))
        kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_10km),
                          'height': len(lat_10km), 'count': 1, 'crs': CRS.from_dict(init='epsg:4326'),
                            'transform': Affine(0.1, 0.0, lon_10km[0]-0.05, 0.0, -0.1, lat_10km[0]+0.05)}
        gpm_sm_1km_output = sub_n_reproj(gpm_file_list_avg_all[imo], kwargs_1km_sub, sub_window_1km, output_crs)

        masked_ds_1km, mask_transform_ds_1km = mask(dataset=gpm_sm_1km_output, shapes=crop_shape_all[ife], crop=True)
        masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
        masked_ds_1km = masked_ds_1km.squeeze()

        masked_ds_1km_all_gpm_1basin.append(masked_ds_1km)
        del(masked_ds_1km, gpm_sm_1km_output)
    print(ife)
    masked_ds_1km_all_gpm_subbasin.append(masked_ds_1km_all_gpm_1basin)
    del(masked_ds_1km_all_gpm_1basin)

# NDVI
masked_ds_1km_all_modis_subbasin = []
for ife in range(len(crop_shape_all)):
    masked_ds_1km_all_modis_1basin = []
    for imo in range(len(modis_file_list_avg_all)):
        sub_window_1km = Window(0, 0, smap_sm_1day.shape[1], smap_sm_1day.shape[0])
        kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': smap_sm_1day.shape[1],
                            'height': smap_sm_1day.shape[0], 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                            'transform': smap_sm_1day.transform}
        smap_sm_1km_output = sub_n_reproj(modis_file_list_avg_all[imo], kwargs_1km_sub, sub_window_1km, output_crs)

        masked_ds_1km, mask_transform_ds_1km = mask(dataset=smap_sm_1km_output, shapes=crop_shape_all[ife], crop=True)
        masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
        masked_ds_1km = masked_ds_1km.squeeze()

        masked_ds_1km_all_modis_1basin.append(masked_ds_1km)
        del(masked_ds_1km, smap_sm_1km_output)
    print(ife)
    masked_ds_1km_all_modis_subbasin.append(masked_ds_1km_all_modis_1basin)
    del(masked_ds_1km_all_modis_1basin)



masked_ds_1km_smap_avg_subbasin = [np.nanmean(np.nanmean(np.array(masked_ds_1km_all_subbasin[x]), axis=1), axis=1)
                                   for x in range(len(masked_ds_1km_all_subbasin))]
masked_ds_1km_gpm_avg_subbasin = [np.nanmean(np.nanmean(np.array(masked_ds_1km_all_gpm_subbasin[x]), axis=1), axis=1)
                                   for x in range(len(masked_ds_1km_all_gpm_subbasin))]
masked_ds_1km_modis_avg_subbasin = [np.nanmean(np.nanmean(np.array(masked_ds_1km_all_modis_subbasin[x]), axis=1), axis=1)
                                   for x in range(len(masked_ds_1km_all_modis_subbasin))]
masked_ds_1km_smap_avg_subbasin = np.array(masked_ds_1km_smap_avg_subbasin)
masked_ds_1km_gpm_avg_subbasin = np.array(masked_ds_1km_gpm_avg_subbasin)
masked_ds_1km_modis_avg_subbasin = np.array(masked_ds_1km_modis_avg_subbasin)

smap_empty_subbasin = np.empty((3, 3))
smap_empty_subbasin[:] = np.nan
masked_ds_1km_smap_avg_subbasin = np.concatenate((smap_empty_subbasin, masked_ds_1km_smap_avg_subbasin, smap_empty_subbasin), axis=1)
masked_ds_1km_gpm_avg_subbasin = np.concatenate((masked_ds_1km_gpm_avg_subbasin, smap_empty_subbasin), axis=1)
masked_ds_1km_modis_avg_subbasin = np.concatenate((masked_ds_1km_modis_avg_subbasin, smap_empty_subbasin), axis=1)


subbasin_name = ['Camp Fire', 'North Complex', 'Dixie']
file_name = ['camp_fire', 'north_complex', 'dixie']
time_start = [40, 61, 72]
time_end = [53, 78, 88]
x_label = []
for x in range(5, len(yearname)):
    for y in range(len(monthname)):
        single_label = monthname[y] + '-' + str(yearname[x])
        x_label.append(single_label)
        del(single_label)


for i in range(3):
    fig = plt.figure(figsize=(10, 10), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.1, top=0.9, hspace=0.3, wspace=0.25)
    # SMAP SM
    x = masked_ds_1km_smap_avg_subbasin[i, time_start[i]:time_end[i]]
    z = masked_ds_1km_gpm_avg_subbasin[i, time_start[i]:time_end[i]]
    ax = fig.add_subplot(2, 1, 1)
    lns1 = ax.plot(x, c='r', marker='s', label='SMAP', markersize=3, linestyle='--', linewidth=1)
    ax.axvspan(6, len(x)-6, facecolor='r', alpha=0.5)
    # lns2 = ax.plot(x, c='k', marker='s', label='MODIS', markersize=3, linestyle='--', linewidth=1)
    # plt.show()
    plt.xlim(0, len(x)-1)
    # ax.set_xticks(np.arange(0, masked_ds_1km_smap_avg_subbasin.shape[1]+12, 12))
    ax.set_xticklabels([])
    # labels = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    # mticks = ax.get_xticks()
    mticks = np.arange(len(x))
    ax.set_xticks(mticks, minor=False)
    # ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xticklabels(x_label[time_start[i]:time_end[i]], minor=False, rotation=45)

    plt.ylim(0, 0.6)
    ax.set_yticks(np.arange(0, 0.7, 0.1))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(linestyle='--')

    ax2 = ax.twinx()
    ax2.set_ylim(0, 30)
    ax2.invert_yaxis()
    ax2.set_yticks(np.arange(0, 140, 20))
    lns3 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=10)
    handles = lns1 + [lns3]
    labels = [l.get_label() for l in handles]

    plt.legend(handles, labels, loc=(0, 1.01), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
    fig.text(0.02, 0.65, 'SMAP SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=14, fontweight='bold')
    fig.text(0.96, 0.62, 'GPM Precipitation (mm)', rotation='vertical', fontsize=14, fontweight='bold')

    # MODIS NDVI
    x = masked_ds_1km_modis_avg_subbasin[i, time_start[i]:time_end[i]]
    z = masked_ds_1km_gpm_avg_subbasin[i, time_start[i]:time_end[i]]
    ax = fig.add_subplot(2, 1, 2)
    lns1 = ax.plot(x, c='k', marker='s', label='MODIS NDVI', markersize=3, linestyle='--', linewidth=1)
    ax.axvspan(6, len(x)-6, facecolor='r', alpha=0.5)
    # lns2 = ax.plot(x, c='k', marker='s', label='MODIS', markersize=3, linestyle='--', linewidth=1)
    # plt.show()
    plt.xlim(0, len(x)-1)
    ax.set_xticks(mticks, minor=False)
    # ax.set_xticks(np.arange(0, masked_ds_1km_smap_avg_subbasin.shape[1]+12, 12))
    ax.set_xticklabels([])
    # labels = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
    # mticks = ax.get_xticks()
    mticks = np.arange(len(x))
    # ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
    ax.set_xticks(mticks, minor=False)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xticklabels(x_label[time_start[i]:time_end[i]], minor=False, rotation=45)
    # ax.set_xticklabels(labels, minor=True)

    plt.ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(linestyle='--')

    ax2 = ax.twinx()
    ax2.set_ylim(0, 30)
    ax2.invert_yaxis()
    ax2.set_yticks(np.arange(0, 140, 20))
    lns3 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=10)
    handles = lns1 + [lns3]
    labels = [l.get_label() for l in handles]

    plt.legend(handles, labels, loc=(0, 1.01), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
    fig.text(0.5, 0.01, 'Month', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.2, 'MODIS NDVI', rotation='vertical', fontsize=14, fontweight='bold')
    fig.text(0.96, 0.18, 'GPM Precipitation (mm)', rotation='vertical', fontsize=14, fontweight='bold')
    plt.suptitle(subbasin_name[i], fontsize=16, y=0.98, fontweight='bold')
    plt.savefig(path_results + '/feather_basin/' + file_name[i] + '_tseries' + '_new.png')
    plt.close(fig)

gpm_file_list_avg_all_stack = np.stack(gpm_file_list_avg_all, axis=2)
gpm_file_list_avg_all_stack = np.nanmean(np.nanmean(gpm_file_list_avg_all_stack, axis=0), axis=0)
df_gpm_file_list_avg_all_stack = pd.DataFrame(gpm_file_list_avg_all_stack)
df_gpm_file_list_avg_all_stack.to_csv(path_output + '/FeatherBasin/gpm_monthly.csv')

