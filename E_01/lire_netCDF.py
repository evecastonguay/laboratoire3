# Packages a importer
import os
import cartopy.crs as ccrs
import sys
import glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import xarray as xr

# Repertoire ou le fichier se trouve
path_file='/Users/evecastonguay/Desktop/Labo III/E_01/imerg_pr_201911_3h.nc4'

# Nom de la variable
var_name='precipitationCal'

# Pour lire le fichier
print('Reading file: ',path_file)
ds_i = xr.open_dataset(path_file)
ds_i.close()
print('Reading file: DONE')

precipitation = ds_i[var_name]
lons = ds_i['lon']
lats = ds_i['lat']

point_hasard = precipitation
print(point_hasard)

