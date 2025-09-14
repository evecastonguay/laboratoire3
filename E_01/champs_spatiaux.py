#%%
"""
Ève Castonguay, UQAM
Laboratoire 3
Partie 2: analyse des champs spatiaux

V.0: 13/09/2025
"""

# packages
import os
import cartopy.crs as ccrs
import sys
import glob
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
from dateutil import tz
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from matplotlib.pyplot import title, xlabel
from scipy.spatial import cKDTree, KDTree
import pandas as pd

# accès aux données
path_file='/Users/evecastonguay/Desktop/Labo3/E_01/imerg_pr_201911_3h.nc4'
var_name='precipitationCal'
print('Reading file: ',path_file)
ds = xr.open_dataset(path_file)
ds.close()
print('Reading file: DONE')
pcpts = ds[var_name]
lons = ds['lon']
lats = ds['lat']

#%% 1) distribution spatiale du taux de précipitation pour le 1er novembre
nov_01 = ds.sel(time='2019-11-01T12:00:00.00')
nov_01_pcpts = nov_01[var_name].values

# projection cylindrique Lambert (code de https://scitools.org.uk/cartopy/docs/v0.10/crs/projections.html)
plt.figure(figsize=(9.42477796077, 3))
ax = plt.axes(projection=ccrs.LambertCylindrical())
ax.coastlines(resolution='110m',linewidths=0.5)
ax.gridlines(draw_labels={"bottom": "x", "left": "y"}, dms=True, x_inline=False, y_inline=False)

# précipitations sur la carte
im = ax.pcolormesh(lons,lats,nov_01_pcpts,vmin=0,vmax=20,cmap='Blues')
plt.colorbar(im)
ax.set(title=f"Distribution spatiale du taux de précipitation pour le 01/11/2019 à 12 UTC")

plt.show()
