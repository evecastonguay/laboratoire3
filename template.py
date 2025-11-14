#%%
#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib.pyplot as plt
import climlab
from climlab import constants as constants
import sys
import xarray as xr
import pdb
import copy as cp
import seaborn as sns

## liens utiles
# https://climlab.readthedocs.io/en/latest/courseware/Spectral_OLR_with_RRTMG.html [permis comprendre erreur]
# https://climlab.readthedocs.io/en/latest/courseware/RCE_with_CAM3_radiation.html [obtenir concentrations des gaz]

# Pour fixer la taille de la police sur les graphes
import matplotlib as matplotlib
font = {'size'   : 13}
matplotlib.rc('font', **font)
plt.rcParams["text.usetex"] = True
sns.set_style("darkgrid", {"axes.facecolor": "0.9"})

outpath='/Users/evecastonguay/Desktop/Labo/E_03/Figures/' # repertoire pour les figures, il faut le creer dans votre repertoire

units=r'W m$^{-2}$' # Unités puisance
alb=.25 # Albedo surface
Nz=30 # nombre de niveaux verticaux

# Load the reference vertical temperature profile from the NCEP reanalysis
ncep_lev=np.load('npy/ncep_lev.npy') # levels
ncep_T=np.load('npy/ncep_T.npy')+273.15 # temperature

#  State variables (Air and surface temperature) for the single-column Radiative-Convective model
state = climlab.column_state(num_lev=30)

#  Fixed relative humidity (calculer HR à partir de l'état de la colonne (state))
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)

#  Couple water vapor to radiation (RRTMG = code de transfert radiatif)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)

# Creation d'un modele couplé avec rayonemment et vapour d'eau
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm2 = climlab.process_like(rcm) # creation d'un clone du modele rcm


print('\n','\n','********************************************')
print('Control simulation ') # fig1,2
print('********************************************')
# Make the initial state isothermal
rcm.state.Tatm[:] = rcm.state.Ts
T=[]
q=[]
tr=[]
print(state) # Ts = 288K, et Tatm=288K pour tous les 30 niveaux car atmosphère est isothermale
# Plot temperature
for t in range(1000): # 1000 time steps de 86400 s (1000 jours = 2.7 ans) -> print(modele.param)
    T.append(cp.deepcopy(rcm.Tatm))
    q.append(cp.deepcopy(rcm.q))
    plt.plot(rcm.Tatm,rcm.lev)
    rcm.step_forward() #run the model forward one time step
    if abs(rcm.ASR - rcm.OLR)<1: # energy balance, in W/m2 (Absorbed Shortwave Radiation, Outgoing Longwave Radiation)
        tr.append(t)
print('Équilibre atteint au temps t ='+str(tr[0]) + "(après t jours)")
plt.xlabel('Température (K)')
plt.ylabel('Pression (hPa)')
plt.gca().invert_yaxis()
plt.plot(ncep_T, ncep_lev, marker='x',color='k',label="réanalyse NCEP") # ajout du profil NCEP
fig_name=outpath+'fig1.png'
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()
print('output figure: ', fig_name)

#Plot humidity
for t in range(1000):
    plt.plot(q[t],rcm.lev)
plt.xlabel('Humidité spécifique (kg/kg)')
plt.ylabel('Pression (hPa)')
fig_name=outpath+'fig2.png'
plt.gca().invert_yaxis()
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()
print('output figure: ', fig_name)

# Quel est la sortie du modèle (état de l'atmosphère une fois l'équilibre atteint)?
print('diagnostics: ',rcm.diagnostics,'\n') # see https://climlab.readthedocs.io/en/latest/api/climlab.radiation.Radiation.html
print('tendencies',rcm.tendencies,'\n')
print('Tair: ',rcm.Tatm,'\n')
print('albedo',rcm.SW_flux_up[-1]/rcm.SW_flux_down[-1],'\n')
print('co2',rad.absorber_vmr['CO2'],'\n') #volumetric mixing ratio
print('ch4',rad.absorber_vmr['CH4'],'\n') #volumetric mixing ratio
# deux manières d'accéder aux niveaux de pression
# 1) print(rcm.lev)
# 2) print(state['Tatm'].domain.axes['lev'].points)

#%%
# toujours rouler la simu de contrôle avant !!!
print('\n','\n','********************************************')
print('Sensitivity to the concentration of gases in the atmosphere') # fig3
print('********************************************')
colors=['k','r','g','orange']
plt.plot(rcm.Tatm, rcm.lev, marker='s', color=colors[0],label='contrôle') # [::-1] sur T°
plt.plot(rcm.Ts, 1000, marker='s',color=colors[0]) # à 1000 hPa on met la Ts
print('control',rcm.Ts)
for gi,gg in enumerate(['O3','CO2','CH4']):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr[gg] = 0 # on enlève le gaz en particulier
    rcm.integrate_years(2) # Run the model for two years
    plt.plot(rcm.Tatm, rcm.lev, marker='s', label='non-'+gg,color=colors[gi+1])
    plt.plot(rcm.Ts, 1000, marker='s',color=colors[gi+1])
    print(rcm.Ts,gg)
plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='réanalyse NCEP')
plt.gca().invert_yaxis()
#plt.title('Sensitivity: gases')
plt.ylabel('Pression (hPa)')
plt.xlabel('Température (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig3.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()

#%%
#### BLOC: figure [3,1] des variations (*2,*4,*6) de concentration des 3 ges étudiés
print('********************************************')
print('Variation de la concentration de CO2')
print('********************************************')

## MANIPULATION DES CONCENTRATIONS DE GAZ
# pour changer la valeur: rcm.absorber_vmr[nom_du_gaz] = X, où X est en fraction décimale
# du pourcentage de la molécule dans l'atmosphère.
# exemple: O2 = 21%, donc 0.21.
# exemple pour le CO2: 422 ppm = 0.000422
# print(rcm.subprocess.Radiation.input) # afficher les concentrations de chaque gaz

fig1, ax1 = plt.subplots(3, 1, figsize=(6.5, 12), sharex=True)

## Graphique: variation de la concentration de CO2
# concentration par défaut = 0.000348
# 1) simu de contrôle
alb=.25
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.integrate_years(2)
ax1[0].plot(rcm.Tatm, rcm.lev, color='k',label='contrôle')
ax1[0].scatter(rcm.Ts, 1000,color='k')
# 2) plot les différentes simulation qui font varier la [C] de gaz
# multiplier la concentration par défaut du modèle par 2, 4, 6
conc = [2,4,6]
for i in conc:
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = rcm.absorber_vmr['CO2']*i # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    # plot model
    ax1[0].plot(rcm.Tatm, rcm.lev, label='CO2*'+str(i)) # ,color='r',alpha=0.5
    ax1[0].scatter(rcm.Ts, 1000) # pour faire un point représentant température de surface
    print(rcm.Ts,'CO2')
    print(rcm.subprocess.Radiation.input['absorber_vmr']['CO2'])
# 3) plot réanalyse NCEP
#plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='réanalyse NCEP')
# 4) détails en lien avec figure
ax1[0].invert_yaxis()
ax1[0].set(title=r"a) CO2")
ax1[0].set_ylabel('Pression (hPa)')
ax1[0].legend()
'''plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig3-1.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()'''

print('********************************************')
print('Variation de la concentration de CH4')
print('********************************************')

# 1) simu de contrôle
alb=.25
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.integrate_years(2)
ax1[1].plot(rcm.Tatm, rcm.lev, color='k',label='contrôle') # [::-1] sur T°
ax1[1].scatter(rcm.Ts, 1000, color='k') # à 1000 hPa on met la Ts
# 2) plot les différentes simulation qui font varier la [C] de gaz
conc = [2,4,6]
for i in conc:
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CH4'] = rcm.absorber_vmr['CH4']*i # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    # plot model
    ax1[1].plot(rcm.Tatm, rcm.lev, label='CH4*'+str(i)) # ,color='r',alpha=0.5
    ax1[1].scatter(rcm.Ts, 1000) # pour faire un point représentant température de surface
    print(rcm.Ts, 'CH4')
    # afficher les valeurs de concentration
    print(rcm.subprocess.Radiation.input['absorber_vmr']['CH4'])
# 3) plot réanalyse NCEP
#plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='réanalyse NCEP')
# 4) détails en lien avec figure
ax1[1].invert_yaxis()
ax1[1].set(title=r"b) CH4")
ax1[1].set_ylabel('Pression (hPa)')
ax1[1].legend()
'''plt.gca().invert_yaxis()
plt.ylabel('Pression (hPa)')
plt.xlabel('Température (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig3-2.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()'''

print('********************************************')
print('Variation de la concentration de O3')
print('********************************************')

# 1) simu de contrôle
alb=.25
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.integrate_years(2)
ax1[2].plot(rcm.Tatm, rcm.lev, color='k',label='contrôle') # [::-1] sur T°
ax1[2].scatter(rcm.Ts, 1000, color='k') # à 1000 hPa on met la Ts
# 2) plot les différentes simulation qui font varier la [C] de gaz
conc = [2,4,6]
for i in conc:
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['O3'] = rcm.absorber_vmr['O3']*i # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    # plot model
    ax1[2].plot(rcm.Tatm, rcm.lev, label='O3*'+str(i)) # ,color='r',alpha=0.5
    ax1[2].scatter(rcm.Ts, 1000) # pour faire un point représentant température de surface
    print(rcm.Ts, 'O3')
    # afficher les valeurs de concentration
    print(rcm.subprocess.Radiation.input['absorber_vmr']['O3'])
# 3) plot réanalyse NCEP
#plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='réanalyse NCEP')
# 4) détails en lien avec figure
ax1[2].invert_yaxis()
ax1[2].set(title=r"c) O3")
ax1[2].set_ylabel('Pression (hPa)')
ax1[2].set_xlabel('Température (K)')
ax1[2].legend()
'''plt.gca().invert_yaxis()
plt.ylabel('Pression (hPa)')
plt.xlabel('Température (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig3-3.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()'''
fig_name=outpath+'fig3x1.png'
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()

#%%
print('********************************************')
print('Valeurs de CO2 et CH4 de 1850 (préindustrielles), 1979, aujourd''hui')
print('********************************************')

c_CO2 = [0.000280,0.000334,0.000425]
c_CH4 = [0.00000071020,0.00000153055,0.00000193354]
labelling = ["1750","1979", "2025"]
alb=.25
# 1) simu de contrôle
#plt.plot(rcm.Tatm, rcm.lev, marker='s', color='k',label='contrôle') # [::-1] sur T°
#plt.plot(rcm.Ts, 1000, marker='s',color='k') # à 1000 hPa on met la Ts
# 2) plot les différentes simulation qui font varier la [C] de gaz
for i in range(3):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c_CO2[i] # modifier [C]
    rcm.absorber_vmr['CH4'] = c_CH4[i]  # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    # plot model
    plt.plot(rcm.Tatm, rcm.lev, label=labelling[i]) # ,color='r',alpha=0.5
    plt.scatter(rcm.Ts, 1000) # pour faire un point représentant température de surface
    # afficher les valeurs de concentration
    print(rcm.subprocess.Radiation.input['absorber_vmr']['CO2'])
    print(rcm.subprocess.Radiation.input['absorber_vmr']['CH4'])
    # afficher la température de surface
    print("la température de surface (K) pour "+labelling[i]+" est "+str(np.round(rcm.Ts.item(), 2)))
# 3) plot réanalyse NCEP
#plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='réanalyse NCEP')
# 4) détails en lien avec figure
plt.gca().invert_yaxis()
plt.ylabel('Pression (hPa)')
plt.xlabel('Température (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig3-4.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()

#%%
print('********************************************')
print('Sensitivity to albedo') # fig4
print('********************************************')
albedos=np.arange(.1,.4,.1)
colors=['maroon','firebrick','indianred','darksalmon']
rcms={}
for alb in albedos:
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcms['rcm'+str(alb)]=rcm

for ai,alb in enumerate(albedos):
    rcms['rcm'+str(alb)].integrate_years(2)
    plt.plot(rcms['rcm'+str(alb)].Tatm[::-1], rcm.lev[::-1], marker='s', label=r'$\alpha$='+str(np.round(alb,1)),color=colors[ai])
    plt.plot(rcms['rcm'+str(alb)].Ts, 1000, marker='s',color=colors[ai])
plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='réanalyse NCEP')
plt.gca().invert_yaxis()
#plt.title('Sensitivity: albedo')
plt.ylabel('Pression (hPa)')
plt.xlabel('Température (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig4.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()

#%%
print('********************************************')
print('Problème de géo-ingénérie')
print('********************************************')

alb=.3
co2_pre_indu = 0.000280

## 1) Température de surface associée au profil contrôle
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.absorber_vmr['CO2'] = co2_pre_indu
rcm.integrate_years(2)
T_ctr = rcm.Ts
print(f"La température de surface pour la simu de contrôle est de {T_ctr.item():.2f} K")

## 2) Température associée à doublement de CO2
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.absorber_vmr['CO2'] = co2_pre_indu*2
rcm.integrate_years(2)
T_2_CO2 = rcm.Ts
print(f"La température de surface pour la simu de contrôle est de {T_2_CO2.item():.2f} K")


## 3) Calculer l'augmentation de température
augm = T_2_CO2.item()-T_ctr.item()
print(f"Doubler les concentrations de CO2 fait augmenter la température de surface de {augm:.2f} K")

## 4) Graphique de la variation de température de surface en fonction de l'albedo
'''vals_alb = np.arange(0,1,0.1)
vals_Ts = []
for a in vals_alb:
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=a)
    rcm = climlab.couple([rad, h2o], name='Radiative-Equilibrium Model')
    rcm.integrate_years(2)
    vals_Ts.append(rcm.Ts)
plt.plot(vals_alb, vals_Ts)
plt.ylabel('Température (K)')
plt.xlabel('Albédo')
fig_name=outpath+'albedo_temp.png'
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()'''
# Code commenté pour diminuer le temps d'exécution.
# On remarque que la relation n'est pas parfaitement linéaire entre la température et la valeur de l'albédo.
# On pourrait toutefois approximer une droite linéaire pour déterminer la variation d’albédo qui permettrait de compenser le réchauffement.
# Puisque l'albédo sur la Terre est de 0.29 [the albedo of the earth] et que dans la simu on prend 0.25, on va faire une régression linéaire entre 0.2 et 0.3 (0.25 +- 0.05)

## 5) Régression linéaire
from sklearn import linear_model
vals_alb = np.arange(0.27,0.33,0.01)
vals_Ts_regression = []
for a in vals_alb:
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=a)
    rcm = climlab.couple([rad, h2o], name='Radiative-Equilibrium Model')
    rcm.integrate_years(2)
    vals_Ts_regression.append(rcm.Ts)
# combiner
vals_Ts_regression = np.array(vals_Ts_regression)
vals_alb = np.reshape(vals_alb, (6,1))
#combined = np.stack((vals_alb,vals_Ts_regression))
reg = linear_model.LinearRegression()
lin_reg = reg.fit(vals_alb,vals_Ts_regression)
print('a: ',reg.coef_,"b: ",reg.intercept_) # renvoie la fonction y = -111.95x + 306.83

## 6) Calculer la variation d'albédo nécessaire
# sachant que ∆T = ∆y = 1.63, et sachant que ∆x = ∆y/a, alors
variation_albedo_necessaire = -augm/reg.coef_ # 0.01437508. crée une baisse de température de même grandeur que l'augmentation causée par le CO2
print("Variation d''albédo nécessaire est: ",variation_albedo_necessaire)

#%%
print('********************************************')
print('Deux profils d''albédo') # rouler section précédente
print('********************************************')

alb=.3
co2_pre_indu = 0.000280

# 1) avant le doublement du co2
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.absorber_vmr['CO2'] = co2_pre_indu
rcm.integrate_years(2)
plt.plot(rcm.Tatm, rcm.lev, color='c',label='état initial') # [::-1] sur T°
plt.scatter(rcm.Ts, 1000, marker='D', color='c') # à 1000 hPa on met la Ts

# 2) doublement du co2, avant géo-ingénierie
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.absorber_vmr['CO2'] = co2_pre_indu*2
rcm.integrate_years(2)
plt.plot(rcm.Tatm, rcm.lev, color='m', label='doublement du CO2') # [::-1] sur T°
plt.scatter(rcm.Ts, 1000, color='m') # à 1000 hPa on met la Ts

# 1) modification sur l'albédo
alb = alb + variation_albedo_necessaire
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcm = climlab.couple([rad,h2o], name='Radiative-Equilibrium Model')
rcm.absorber_vmr['CO2'] = co2_pre_indu*2
rcm.integrate_years(2)
plt.plot(rcm.Tatm, rcm.lev, color='y', label='après géo-ingénierie') # [::-1] sur T°
plt.scatter(rcm.Ts, 1000, color='y') # à 1000 hPa on met la Ts

plt.gca().invert_yaxis()
plt.ylabel('Pression (hPa)')
plt.xlabel('Température (K)')
plt.legend()
fig_name=outpath+'2profils.png'
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()

#%%
print('\n','\n','********************************************')
print('Sensitivity to convection') # fig5
print('********************************************')
# https://climlab.readthedocs.io/en/latest/api/climlab.convection.convadj.html#convectiveadjustment
# https://climlab.readthedocs.io/en/latest/_modules/climlab/convection/convadj.html# lapse rate

alb=.25
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcms={}
rcms['rcm0'] = climlab.couple([rad,h2o], name='Radiative-Convective Model') # Adjustment includes the surface if ``'Ts'`` is included in the ``state``dictionary (and Ts IS included)

conv = climlab.convection.ConvectiveAdjustment(name='Convection', state=state, adj_lapse_rate=6.5) # moist adiabatic lapse rate -> lapse rate in K per km (decrease of T per km)
rcms['rcm1'] = climlab.couple([rad,conv,h2o], name='Radiative-Convective Model')

conv = climlab.convection.ConvectiveAdjustment(name='Convection', state=state, adj_lapse_rate=9.8) # dry adiabatic lapse rate
rcms['rcm2'] = climlab.couple([rad,conv,h2o], name='Radiative-Convective Model')

mod_name=['contrôle','conv-6.5','conv-9.8']
couleurs=['k','teal','mediumpurple']
for ai in range(3):
    rcms['rcm'+str(ai)].integrate_years(2)
    plt.plot(rcms['rcm'+str(ai)].Tatm[::-1], rcm.lev[::-1],  label=mod_name[ai],color=couleurs[ai])
    plt.scatter(rcms['rcm'+str(ai)].Ts, 1000,color=couleurs[ai])
plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='réanalyse NCEP')
plt.gca().invert_yaxis()
#plt.title('Sensitivity: convection')
plt.ylabel('Pression (hPa)')
plt.xlabel('Température (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig5.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,dpi=500,bbox_inches='tight')
plt.close()

#%%
print('\n','\n','********************************************')
print('Comparaison avec GIEC')
print('********************************************')

alb=.25
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
conv = climlab.convection.ConvectiveAdjustment(name='Convection', state=state, adj_lapse_rate=6.5) # moist adiabatic lapse rate -> lapse rate in K per km (decrease of T per km)
rcms = climlab.couple([rad,conv,h2o], name='Radiative-Convective Model')
rcms.integrate_years(2)

## print model ouputs of model with lapse rate = 6.5
print('diagnostics: ',rcms.diagnostics,'\n') # see https://climlab.readthedocs.io/en/latest/api/climlab.radiation.Radiation.html
print('tendencies',rcms.tendencies,'\n')
print('Tair: ',rcms.Tatm,'\n')
print('albedo',rcms.SW_flux_up[-1]/rcms.SW_flux_down[-1],'\n')
print('co2',rad.absorber_vmr['CO2'],'\n') #volumetric mixing ratio
print('ch4',rad.absorber_vmr['CH4'],'\n') #volumetric mixing ratio

#%%
print('\n','\n','********************************************')
print('Figure en extra')
print('********************************************')

# figure combinées
fig, ax = plt.subplots(2,1, figsize=(9, 5))

## partie 1
alb=.29
co2_values_pre = np.arange(280,1100,10)
co2_values = co2_values_pre*1e-6
print(co2_values)
t_sf = []

for i, c in enumerate(co2_values):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    t_sf.append(rcm.Ts[0])

ax[0].plot(co2_values_pre,t_sf,  color='k')
ax[0].set_yticks([274,275,276,277])
ax[0].set_ylabel(r'$T_s$ (K)')
ax[0].set_xlabel('Concentration de CO2 (ppm)')
ax[0].set_title('a)')

'''fig_name=outpath+'extra_co2_p1.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,dpi=500,bbox_inches='tight')'''

## partie 2
co2_historique0 = np.array([280,315,334,370,425])*1e-6
annees_histo = [1750,1958,1979,2000,2025]

co2_historique1 = np.array([280,315,334,370,425,425,400])*1e-6
co2_historique2 = np.array([280,315,334,370,425,480,450])*1e-6
co2_historique3 = np.array([280,315,334,370,425,500,600])*1e-6
co2_historique4 = np.array([280,315,334,370,425,550,850])*1e-6
co2_historique5 = np.array([280,315,334,370,425,575,1100])*1e-6
annees = [1750,1958,1979,2000,2025,2050,2100]

alb=.29
t_sf0 = []
t_sf1 = []
t_sf2 = []
t_sf3 = []
t_sf4 = []
t_sf5 = []

for i, c in enumerate(co2_historique0):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    t_sf0.append(rcm.Ts[0])

for i, c in enumerate(co2_historique1):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    t_sf1.append(rcm.Ts[0])

for i, c in enumerate(co2_historique2):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    t_sf2.append(rcm.Ts[0])

for i, c in enumerate(co2_historique3):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    t_sf3.append(rcm.Ts[0])

for i, c in enumerate(co2_historique4):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    t_sf4.append(rcm.Ts[0])

for i, c in enumerate(co2_historique5):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr['CO2'] = c # modifier [C]
    rcm.integrate_years(2) # Run the model for two years
    t_sf5.append(rcm.Ts[0])

print(f"Écart entre pré-indu et i=15: {t_sf[15]-t_sf[0]:.2f}")
print(f"Écart entre pré-indu et i=16: {t_sf[16]-t_sf[0]:.2f}")
print(f"Écart pour 1: {t_sf1[-1]-t_sf1[0]:.2f}")
print(f"Écart pour 2: {t_sf2[-1]-t_sf2[0]:.2f}")
print(f"Écart pour 3: {t_sf3[-1]-t_sf3[0]:.2f}")
print(f"Écart pour 4: {t_sf4[-1]-t_sf4[0]:.2f}")
print(f"Écart pour 5: {t_sf5[-1]-t_sf5[0]:.2f}")

ax[1].plot(annees,t_sf1,  color='lightskyblue', label='SSP1-1.9')
ax[1].plot(annees,t_sf2,  color='tab:blue', label='SSP1-2.6')
ax[1].plot(annees,t_sf3,  color='orange', label='SSP2-4.5')
ax[1].plot(annees,t_sf4,  color='lightcoral', label='SSP3-7.0')
ax[1].plot(annees,t_sf5,  color='firebrick', label='SSP5-8.5')
ax[1].plot(annees_histo,t_sf0,  color='darkgrey', label='historique')

ax[1].set_ylabel(r'$T_s$ (K)')
ax[1].set_xlabel('Année')
ax[1].set_title('b)')
ax[1].set_yticks([274,275,276,277])
plt.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.5),   # sous le graphe
    ncol=6,                       # en ligne (ajuste le nombre)
    fontsize=8                    # plus petit
)
plt.subplots_adjust(hspace=0.4)
fig_name=outpath+'extra_planche2.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,dpi=500,bbox_inches='tight')
plt.close()
