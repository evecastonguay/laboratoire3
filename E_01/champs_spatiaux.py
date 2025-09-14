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
from matplotlib.pyplot import title, xlabel, yscale, ylabel, xticks
from scipy.spatial import cKDTree, KDTree
import pandas as pd
from scipy.stats import alpha

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
nov_01_pcpts = nov_01[var_name].values # (1800, 3600) numpy.ndarray

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

#%% 2) histogramme des précipitations du 01/11/2019 à 12 UTC
# trouver la valeur max de précipitation dans le monde
np.amax(nov_01_pcpts) # 180.42758 mm/3h

# transformer le tableau de taille 1800 x 3600 en un tableau 1D (comme une liste)
nov_01_pcpts = nov_01_pcpts.flatten()

# créer le graphique
my_bins = [0,0.2,0.4,0.6,0.8,1,2,3,4,5,6,7,8,9,10,15,20,50,100,150,200]
fig, ax = plt.subplots()
ax.bar(nov_01_pcpts,bins=my_bins,color='mediumseagreen',edgecolor="white", linewidth=0.3,alpha=0.8) #range=(0,10)
ax.set(title="Histogramme des précipitations du 01/11/2019 à 12 UTC",
       yscale='log',xlabel='taux de précipitation [mm/3h]',ylabel='nombre de valeurs',
       xticks=[0,10,20,50,100,150,200])
plt.show()

'''
HISTOGRAMME:
my_bins = [0,0.2,0.4,0.6,0.8,1,2,3,4,5,6,7,8,9,10,15,20,50,100,150,200]
fig, ax = plt.subplots()
ax.hist(nov_01_pcpts,bins=my_bins,color='mediumseagreen',edgecolor="white", linewidth=0.3,alpha=0.8) #range=(0,10)
ax.set(title="Histogramme des précipitations du 01/11/2019 à 12 UTC",
       yscale='log',xlabel='taux de précipitation [mm/3h]',ylabel='nombre de valeurs',
       xticks=[0,10,20,50,100,150,200])
plt.show()
BAR:
counts, edges = np.histogram(nov_01_pcpts, bins=[0,1,2,3,4,5,6,7,8,9,10,20,50,100,200])
positions = np.arange(len(counts))  # positions régulières
ax.bar(positions, counts, width=1, edgecolor="white")
ax.set_xticks(positions)
ax.set_xticklabels([f"{edges[i]}-{edges[i+1]}" for i in range(len(counts))], rotation=45)
'''