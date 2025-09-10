
# packages
import os
import cartopy.crs as ccrs
import sys
import glob
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from scipy.spatial import cKDTree, KDTree

# répertoire
path_file='/Users/evecastonguay/Desktop/Labo/E_01/imerg_pr_201911_3h.nc4'

# nom de la variable
var_name='precipitationCal'

# lire fichier
print('Reading file: ',path_file)
ds = xr.open_dataset(path_file)
ds.close()
print('Reading file: DONE')

pcpts = ds[var_name]
lons = ds['lon'] # longitude est un DataArray
lats = ds['lat'] # or, ds.lat

# exemple de lecture avec longitude
"""
print("-> métadonnées:")
print(ds) # affiche les métadonnées

*Dimensions*:           (time: 240, bnds: 2, lat: 1800, lon: 3600)
Coordinates:
  * time              (time) datetime64[ns] 2kB 2019-11-01 ... 2019-11-30T21:...
  * lon               (lon) float32 14kB -179.9 -179.9 -179.8 ... 179.9 179.9
  * lat               (lat) float32 7kB -89.95 -89.85 -89.75 ... 89.85 89.95
Dimensions without coordinates: bnds
Data *variables*:
    time_bnds         (time, bnds) datetime64[ns] 4kB ...
    precipitationCal  (time, lat, lon) float32 6GB ...
*Attributes*:
    CDI:          Climate Data Interface version 1.9.5 (http://mpimet.mpg.de/...
    history:      Mon Sep 27 17:58:00 2021: cdo --timestat_date first -L -f n...
    Conventions:  CF-1.6
    FileHeader:   DOI=10.5067/GPM/IMERG/3B-HH/06;\nDOIauthority=http://dx.doi...
    FileInfo:     DataFormatVersion=6a;\nTKCodeBuildVersion=0;\nMetadataVersi...
    GridHeader:   BinMethod=ARITHMETIC_MEAN;\nRegistration=CENTER;\nLatitudeR...
    CDO:          Climate Data Operators version 1.9.5 (http://mpimet.mpg.de/...

print("-> valeurs:")
print(lons.values) # affiche le vecteur de données (float32)
print(lons.values[0:3]) # affiche les 3 premières valeurs du vecteur
"""

# 1) coordonnées choisies
kl_lat = 3.1
kl_lon = 101.6

mtl_lat = 45.5
mtl_lon = -73.5

oce_lat = 5
oce_lon = 106.0

wal_lat = 47.1 # walenstadt, suisse
wal_lon = 9.2

# 2.1) point de grille le plus proche
# index: accessing a specific element using it's index // index label: value at a certain index number or coordinate
# kuala lumpur
kl_ds = ds.sel(lon=kl_lon, lat=kl_lat, method='nearest')
kl_nearest_lon = kl_ds.lon.values
kl_nearest_lat = kl_ds.lat.values
print(f"Point de grille le plus proche pour Kuala Lumpur: ({kl_nearest_lon:.2f}°; {kl_nearest_lat:.2f}°)")


# 2.2) variation du taux de précipitation en fonction du temps

# kuala lumpur
kl_pcpts = kl_ds[var_name].values
print(type(kl_pcpts))
kl_var = np.diff(kl_pcpts) # (239,)


# 3a) accumulation totale de précipitations

print(f"l'accumulation totale [mm] de précipitations en novembre 2019 pour Kuala Lumpur est: {kl_pcpts.sum():.2f}")

#%%
x, y = np.random.rand(2, 5)
print(x)
print(y)

