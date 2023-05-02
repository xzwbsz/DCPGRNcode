from netCDF4 import Dataset
import numpy as np
import os

# path = '/data/haojy/2022/04/2022_04_01-30.nc'
# dst = Dataset(path, mode='r', format="netCDF4")
# print(dst.variables.keys())
# lon = dst.variables['longitude'][:]
# lat = dst.variables['latitude'][:]
# time = dst.variables['time']
# print()
#
# path1 = '/data/haojy/2022_6v/04/2022_04_01-05.nc'
# dst1 = Dataset(path1, mode='r', format="netCDF4")
# lon1 = dst1.variables['longitude'][:]
# lat1 = dst1.variables['latitude'][:]
# time1 = dst1.variables['time']
# print()

import timestampToTime as t2ts
time = np.load('/public/141pres_npyData16-22/time_all_year.npy')
print(t2ts.timestampToDate(time[0]))
print(t2ts.timestampToDate(time[-1]))
