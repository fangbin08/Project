import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import datetime
import glob
import gdal
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

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


########################################################################################################################
# 0. Input variables
# Specify file paths
# Path of source MODIS data
path_modis = '/Users/binfang/Downloads/Processing/MOD16A2_V6_vmd'
# Path of output tiff
path_op = '/Users/binfang/Downloads/Processing/ET_Output'
# Path of Shapefile
path_shp = '/Users/binfang/Downloads/Processing/shp'

subdataset_id = [0] # the band ID to extract (0th for ET_500m)
band_n = 1 # the number of bands
yearname = np.linspace(2001, 2018, 18, dtype='int')
# monthnum = np.linspace(1, 12, 12, dtype='int')
# monthname = np.arange(1, 13)
# monthname = [str(i).zfill(2) for i in monthname]


# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2001-01-01'
end_date = '2018-12-31'

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

########################################################################################################################

# Load in watershed shapefile boundaries
shapefile = fiona.open(path_shp + '/VMD_WGS.shp', 'r')
crop_shape = [feature["geometry"] for feature in shapefile]
shp_extent = list(shapefile.bounds)

# Extract the MODIS data from hdf files
# for iyr in range(0, len(yearname)):

# path_year = path_modis + '/' + str(yearname[iyr])
os.chdir(path_modis)
hdf_files = sorted(glob.glob('*.hdf'))

hdf_file_name = [hdf_files[x].split('.')[1] for x in range(len(hdf_files))]
hdf_file_name_unique = sorted(list(set(hdf_file_name)))

# Group the MODIS tile files by each day
modis_mat_all = []
for idt in range(len(hdf_file_name_unique)):

    hdf_files_toBuild_ind = [hdf_files.index(i) for i in hdf_files if hdf_file_name_unique[idt] in i]
    hdf_files_toBuild = [hdf_files[i] for i in hdf_files_toBuild_ind]

    hdf_files_list = []
    for idf in range(len(hdf_files_toBuild)):
        extr_file = hdf_subdataset_extraction(path_modis + '/' + hdf_files_toBuild[idf],
                                                          subdataset_id, band_n)
        hdf_files_list.append(extr_file)  # Append the processed hdf to the file list being merged
        print(hdf_files_toBuild[idf])  # Print the file being processed
        del(extr_file)

    # Open file and warp the target raster dimensions and geotransform
    out_ds = gdal.Warp('', hdf_files_list, format='MEM', xRes=0.005, yRes=0.005, dstSRS='EPSG:4326', outputBounds=shp_extent,
                           warpOptions=['SKIP_NOSOURCE=YES'], errorThreshold=0, resampleAlg=gdal.GRA_NearestNeighbour)

    modis_mat = out_ds.ReadAsArray()
    modis_mat_all.append(modis_mat)

modis_mat_all_array = np.stack(modis_mat_all, axis=2)


# Group and average monthly data
date_seq_month_ind = np.array([int(hdf_file_name_unique[x][1:5])*100 + date_seq_month[np.where(int(hdf_file_name_unique[x][1:]) == date_seq_array)[0][0]] for x in
                          range(len(hdf_file_name_unique))])
date_seq_month_ind_unique = np.array(sorted(list(set(date_seq_month_ind))))
date_seq_month_ind_group = [np.where(date_seq_month_ind == date_seq_month_ind_unique[x])
                                for x in range(len(date_seq_month_ind_unique))]
transform = out_ds.GetGeoTransform()

for imo in range(len(date_seq_month_ind_group)):

    modis_mat_1month = np.nanmean(modis_mat_all_array[:, :, date_seq_month_ind_group[imo]], axis=3)
    modis_mat_1month[modis_mat_1month == 0] = np.nan
    modis_mat_1month = np.transpose(modis_mat_1month, (2, 0, 1))

    # Generate subset rasterio dataset
    input_ds_subset = MemoryFile().open(driver='GTiff', count=1, dtype='float32', crs='EPSG:4326',
                                            height=modis_mat_1month.shape[1], width=modis_mat_1month.shape[2],
                                            transform=from_origin(transform[0], transform[3], transform[1], transform[1]))
    input_ds_subset.write(modis_mat_1month)

    masked_ds, mask_transform_ds_1km = mask(dataset=input_ds_subset, shapes=crop_shape, crop=True)
    masked_ds[masked_ds == 0] = np.nan


    kwargs = input_ds_subset.meta.copy()
    kwargs.update({'width': masked_ds.shape[2], 'height': masked_ds.shape[1], 'transform': mask_transform_ds_1km})
    with rasterio.open(path_op + '/' + str(date_seq_month_ind_unique[imo])[0:4] + '_' +
                       str(date_seq_month_ind_unique[imo])[4:] + '.tif', 'w', **kwargs) as output_ds:
        output_ds.write(masked_ds)

    del(modis_mat_1month)
    print(str(date_seq_month_ind_unique[imo]))

