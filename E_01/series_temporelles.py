"""
Ève Castonguay, UQAM
Laboratoire 3
Partie 1: séries temporelles

V.0: 05/09/2025
"""

# packages
import os
import cartopy.crs as ccrs
import sys
import glob
import numpy as np
import datetime as dt
from dateutil import tz
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
kl_ds = ds.sel(lon=kl_lon, lat=kl_lat, method='nearest') # renvoie un datastore avec valeurs uniquement pour kl
kl_nearest_lon = kl_ds.lon.values
kl_nearest_lat = kl_ds.lat.values
print(f"Point de grille le plus proche pour Kuala Lumpur: ({kl_nearest_lon:.2f}°; {kl_nearest_lat:.2f}°)")
# montréal
mtl_ds = ds.sel(lon=mtl_lon, lat=mtl_lat, method='nearest')
mtl_nearest_lon = mtl_ds.lon.values
mtl_nearest_lat = mtl_ds.lat.values
print(f"Point de grille le plus proche pour Montréal: ({mtl_nearest_lon:.2f}°; {mtl_nearest_lat:.2f}°)")
# océan
oce_ds = ds.sel(lon=oce_lon, lat=oce_lat, method='nearest')
oce_nearest_lon = oce_ds.lon.values
oce_nearest_lat = oce_ds.lat.values
print(f"Point de grille le plus proche pour la coordonnée dans l'océan (mer de Chine): ({oce_nearest_lon:.2f}°; {oce_nearest_lat:.2f}°)")
# walenstadt
wal_ds = ds.sel(lon=wal_lon, lat=wal_lat, method='nearest')
wal_nearest_lon = wal_ds.lon.values
wal_nearest_lat = wal_ds.lat.values
print(f"Point de grille le plus proche pour Walenstadt: ({wal_nearest_lon:.2f}°; {wal_nearest_lat:.2f}°)")

#%%
# 2.2) variation du taux de précipitation en fonction du temps
# kuala lumpur
kl_pcpts = kl_ds[var_name].values # type np.ndarray, (240,)
kl_var = np.diff(kl_pcpts) # (239,)
# montréal
mtl_pcpts = mtl_ds[var_name].values
mtl_var = np.diff(mtl_pcpts)
# océan
oce_pcpts = oce_ds[var_name].values
oce_var = np.diff(oce_pcpts)
# walenstadt
wal_pcpts = wal_ds[var_name].values
wal_var = np.diff(wal_pcpts)
wal_time = wal_ds['time'].values[:-1]

#%%
# making the plot
fig, ax = plt.subplots(4, figsize=[7,8])
ax[0].plot(kl_var, 'b')
ax[0].set_ylabel('Variation taux pcpt \n [mm/3h]', size=12)
ax[0].axhline(y=0,linewidth=0.5, color = 'gray')
ax[1].plot(mtl_var, 'b')
ax[1].set_ylabel('Variation taux pcpt \n [mm/3h]', size=12)
ax[1].axhline(y=0,linewidth=0.5, color = 'gray')
ax[2].plot(oce_var, 'b')
ax[2].set_ylabel('Variation taux pcpt \n [mm/3h]', size=12)
ax[2].axhline(y=0,linewidth=0.5, color = 'gray')
ax[3].plot(wal_time, wal_var, 'b')
ax[3].set_ylabel('Variation taux pcpt \n [mm/3h]', size=12)
ax[3].axhline(y=0,linewidth=0.5, color = 'gray')
#ax[0].set_xlim([start,end])
for i in range(4):
    ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax[i].tick_params(labelsize=12)
    if i!=4:
        ax[i].set_xticks([])
ax[3].set_xlabel('Date et heure',size=12)
plt.subplots_adjust(hspace=0)
#ax[3].set_xlim([start,end])
ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.show()
# 2019-11-01T00:00:00.000000000

#%%
# 3a) accumulation totale de précipitations
print(f"Accumulation totale [mm] de précipitations en novembre 2019 pour Kuala Lumpur: {kl_pcpts.sum():.2f}")
print(f"Accumulation totale [mm] de précipitations en novembre 2019 pour Montréal: {mtl_pcpts.sum():.2f}")
print(f"Accumulation totale [mm] de précipitations en novembre 2019 en mer de Chine: {oce_pcpts.sum():.2f}")
print(f"Accumulation totale [mm] de précipitations en novembre 2019 pour Walenstadt: {wal_pcpts.sum():.2f}")

#%%
# 3b) nombre de mesures avec un taux supérieur à 0mm/3h & fraction du temps qu'il pleut
kl_jours_pluie = np.sum(kl_pcpts > 0)
mtl_jours_pluie = np.sum(mtl_pcpts > 0)
oce_jours_pluie = np.sum(oce_pcpts > 0)
wal_jours_pluie = np.sum(wal_pcpts > 0)
print(f"Nombre de mesures avec un taux supérieur à 0mm/3h pour Kuala Lumpur: {kl_jours_pluie}")
print(f"Nombre de mesures avec un taux supérieur à 0mm/3h pour Montréal: {mtl_jours_pluie}")
print(f"Nombre de mesures avec un taux supérieur à 0mm/3h en mer de Chine: {oce_jours_pluie}")
print(f"Nombre de mesures avec un taux supérieur à 0mm/3h pour Walenstadt: {wal_jours_pluie}")
kl_ftp = kl_jours_pluie/240
mtl_ftp = mtl_jours_pluie/240
oce_ftp = oce_jours_pluie/240
wal_ftp = wal_jours_pluie/240
print(f"F.t.p. pour Kuala Lumpur: {kl_ftp:.2f}")
print(f"f.t.p. pour Montréal: {mtl_ftp:.2f}")
print(f"F.t.p. en mer de Chine: {oce_ftp:.2f}")
print(f"F.t.p. pour Walenstadt: {wal_ftp:.2f}")

#%% 3c) précipitation moyenne & intensité de pcpt moyenne
kl_pcpts_moy = kl_pcpts.mean()
kl_pcpts_int = kl_pcpts[kl_pcpts>0].mean()
print(f"Moyennes de précipitation et d'intensité de précipitation [mm/3h] pour Kuala Lumpur: {kl_pcpts_moy:.2f} et {kl_pcpts_int:.2f}")
mtl_pcpts_moy = mtl_pcpts.mean()
mtl_pcpts_int = mtl_pcpts[mtl_pcpts>0].mean()
print(f"Moyennes de précipitation et d'intensité de précipitation [mm/3h] pour Montréal: {mtl_pcpts_moy:.2f} et {mtl_pcpts_int:.2f}")
oce_pcpts_moy = oce_pcpts.mean()
oce_pcpts_int = oce_pcpts[oce_pcpts>0].mean()
print(f"Moyennes de précipitation et d'intensité de précipitation [mm/3h] pour la mer de Chine: {oce_pcpts_moy:.2f} et {oce_pcpts_int:.2f}")
wal_pcpts_moy = wal_pcpts.mean()
wal_pcpts_int = wal_pcpts[wal_pcpts>0].mean()
print(f"Moyennes de précipitation et d'intensité de précipitation [mm/3h] pour Walenstadt: {wal_pcpts_moy:.2f} et {wal_pcpts_int:.2f}")

#%% 3d) durée maximale des évènements de précipitation
# kl
zero_one_vector = kl_pcpts > 0
counts = []
n = 0
for i in zero_one_vector:
    if i==0:
        counts.append(n)
        n = 0
    if i==1:
        n+=1
print(f"Durée maximale des évènements de précipitation pour Kuala Lumpur: {max(counts)*3} heures")
# mtl
zero_one_vector = mtl_pcpts > 0
counts = []
n = 0
for i in zero_one_vector:
    if i==0:
        counts.append(n)
        n = 0
    if i==1:
        n+=1
print(f"Durée maximale des évènements de précipitation pour Montréal: {max(counts)*3} heures")
# oce
zero_one_vector = oce_pcpts > 0
counts = []
n = 0
for i in zero_one_vector:
    if i==0:
        counts.append(n)
        n = 0
    if i==1:
        n+=1
print(f"Durée maximale des évènements de précipitation pour la mer de Chine: {max(counts)*3} heures")
# wal
zero_one_vector = wal_pcpts > 0
counts = []
n = 0
for i in zero_one_vector:
    if i==0:
        counts.append(n)
        n = 0
    if i==1:
        n+=1
print(f"Durée maximale des évènements de précipitation pour Walenstadt: {max(counts)*3} heures")

#%% 3e) valeur maximale du taux de précipitation
print(f"Valeur maximale du taux de précipitation [mm/3h] pour Kuala Lumpur: {max(kl_pcpts):.2f}")
print(f"Valeur maximale du taux de précipitation [mm/3h] pour Montréal: {max(mtl_pcpts):.2f}")
print(f"Valeur maximale du taux de précipitation [mm/3h] pour la mer de Chine: {max(oce_pcpts):.2f}")
print(f"Valeur maximale du taux de précipitation [mm/3h] pour Walenstadt: {max(wal_pcpts):.2f}")

#%% 3f) points de grille supplémentaires à 50 km du point original
place = ["kl","mtl","oce","wal"] #

for p in place:
    # créer les coordonnées des 4 points à 50 km (0.45°) du point central
    globals()[f"{p}_1_lon"] = globals()[f"{p}_lon"]
    globals()[f"{p}_1_lat"] = globals()[f"{p}_lat"]+0.45
    globals()[f"{p}_2_lon"] = globals()[f"{p}_lon"]
    globals()[f"{p}_2_lat"] = globals()[f"{p}_lat"]-0.45
    globals()[f"{p}_3_lon"] = globals()[f"{p}_lon"]-0.45
    globals()[f"{p}_3_lat"] = globals()[f"{p}_lat"]
    globals()[f"{p}_4_lon"] = globals()[f"{p}_lon"]+0.45
    globals()[f"{p}_4_lat"] = globals()[f"{p}_lat"]

    print(globals()[f"{p}_4_lon"])

    for i in range(1,5):
        # créer un datastore contenant uniquement les valeurs de variable pour la coordonnée choisie
        globals()[f"{p}_{i}_ds"] = ds.sel(lon=globals()[f"{p}_{i}_lon"], lat=globals()[f"{p}_{i}_lat"], method='nearest')

        # valeurs (240) de précipitation pour la coordonnée choisie
        globals()[f"{p}_{i}_pcpts"] = globals()[f"{p}_{i}_ds"][var_name].values
        if  globals()[f"{p}_{i}_pcpts"].shape != (240,):
            raise ValueError(f"Le format attendu est (240,)")

        # redimensionner
        globals()[f"{p}_{i}_pcpts"] = globals()[f"{p}_{i}_pcpts"].reshape(1,240)

    # créer une matrice par endroit (5, 240). les lignes représentent les différentes coordonnées et les colonnes sont les observations
    globals()[f"{p}_pcpts_matrix"] = globals()[f"{p}_pcpts"].reshape(1,240)
    for i in range(1, 5):
        globals()[f"{p}_pcpts_matrix"] = np.append(globals()[f"{p}_pcpts_matrix"],globals()[f"{p}_{i}_pcpts"],axis=0)

    # calculer la corrélation
    globals()[f"{p}_corr_matrix"] = np.corrcoef(globals()[f"{p}_pcpts_matrix"])
    print(globals()[f"{p}_corr_matrix"])
    print(f'Corrélation entre la série temporelle de {p} et son point à 50 km au nord: {globals()[f"{p}_corr_matrix"][1,0]:.2f}')
    print(f'Corrélation entre la série temporelle de {p} et son point à 50 km au sud: {globals()[f"{p}_corr_matrix"][2, 0]:.2f}')
    print(f'Corrélation entre la série temporelle de {p} et son point à 50 km à l\'ouest: {globals()[f"{p}_corr_matrix"][3, 0]:.2f}')
    print(f'Corrélation entre la série temporelle de {p} et son point à 50 km à l\'est: {globals()[f"{p}_corr_matrix"][4, 0]:.2f}')

'''
Résultats pour kl:
Corrélation entre la série temporelle de kl et son point à 50 km au nord: 0.54
Corrélation entre la série temporelle de kl et son point à 50 km au sud: 0.50
Corrélation entre la série temporelle de kl et son point à 50 km à l'ouest: 0.23
Corrélation entre la série temporelle de kl et son point à 50 km à l'est: 0.25
'''

#%% 3f) points de grille supplémentaires à 50 km du point original - scatter plot
# Kuala Lumpur
fig, ax = plt.subplots()
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
label = ['kl-north', 'kl-south', 'kl-west', 'kl-east']
'''
# -> x-axis (the same for all)
x = kl_ds['time'].values

# -> central coordinate
ax.scatter(x, kl_pcpts, c='tab:gray', label='kl', alpha=0.3, edgecolors='none')

# -> 50 km coordinates
'''
for i in range(1,5):
    ax.scatter(kl_pcpts, globals()[f"kl_{i}_pcpts"], c=color[i-1], label=label[i-1], alpha=0.5, edgecolors='none')

ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.legend()
ax.grid(True)
plt.show()

#%% cycle journalier de précipitation
'''
différence entre UTC et heure locale = longitude/15
-> Montreal 
*malgré changement d'heure le 3 novembre, assumé heure standard d'hiver pour pas créer 
un double dans les données au moment du changement d'heure*
'''
# calcul de l'ajustement à l'heure UTC pour chaque ville
kl_local = kl_lon/15
mtl_local = mtl_lon/15
oce_local = oce_lon/15
wal_local = wal_lon/15

# convertir UTC (datetime string) en heure locale
# https://stackoverflow.com/questions/4770297/convert-utc-datetime-string-to-local-datetime
utc = datetime.strptime('2011-01-21 02:37:21', '%Y-%m-%d %H:%M:%S') # crée un datetime string

kl_time = kl_ds['time'].values # class np.datetime64,
print(kl_time)

for i in kl_time:
    # replace
# essayer de voir ça fait quoi soustraire deux datetime
# test commit















