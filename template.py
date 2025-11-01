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

# Pour fixer la taille de la police partout
import matplotlib as matplotlib
font = {'size'   : 13}
matplotlib.rc('font', **font)
plt.rcParams["text.usetex"] = True
sns.set_style("darkgrid", {"axes.facecolor": "0.9"})

outpath='/Users/evecastonguay/Desktop/Labo/E_03/Figures/' # repertoire pour les figures, il faut le creer dans votre repertoire

units=r'W m$^{-2}$' # Unités puisance
alb=.25 # Albedo surface
Nz=30 # nombre de niveaux verticaux
levels=np.arange(200,330,20) # pas utilisé dans code
levels=[298] # pas utilisé dans code
Tlims=[180,310] # pas utilisé dans code

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
    plt.plot(rcm.Tatm,rcm.lev[::-1])
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
    plt.plot(q[t],rcm.lev[::-1])
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

print('\n','\n','********************************************')
print('Sensitivity to the concentration of gases in the atmosphere') # fig3
print('********************************************')
colors=['k','r','g','orange']
plt.plot(rcm.Tatm[::-1], rcm.lev[::-1], marker='s', color=colors[0],label='control')
plt.plot(rcm.Ts, 1000, marker='s',color=colors[0])

for gi,gg in enumerate(['O3','CO2','CH4']):
    state = climlab.column_state(num_lev=30)
    h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
    rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
    rcm = climlab.couple([rad,h2o], name='Radiative-Convective Model')
    rcm.absorber_vmr[gg] = 0
    rcm.integrate_years(2) # Run the model for two years
    plt.plot(rcm.Tatm[::-1], rcm.lev[::-1], marker='s', label='non-'+gg,color=colors[gi+1])
    plt.plot(rcm.Ts, 1000, marker='s',color=colors[gi+1])
plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='NCEP reanalysis')
plt.gca().invert_yaxis()
plt.title('Sensitivity: gases')
plt.ylabel('Pression (hPa)')
plt.xlabel('Temperature (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig3.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()

print('\n','\n','********************************************')
print('Sensitivity to albedo') # fig4
print('********************************************')
albedos=np.arange(.1,.4,.1)
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
plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='NCEP reanalysis')
plt.gca().invert_yaxis()
plt.title('Sensitivity: albedo')
plt.ylabel('Pression (hPa)')
plt.xlabel('Temperature (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig4.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,bbox_inches='tight',dpi=500)
plt.close()

print('\n','\n','********************************************')
print('Sensitivity to convection') # fig5
print('********************************************')
alb=.25
state = climlab.column_state(num_lev=30)
h2o = climlab.radiation.ManabeWaterVapor(name='WaterVapor', state=state)
rad = climlab.radiation.RRTMG(name='Radiation', state=state, specific_humidity=h2o.q, albedo=alb)
rcms={}
rcms['rcm0'] = climlab.couple([rad,h2o], name='Radiative-Convective Model')
conv = climlab.convection.ConvectiveAdjustment(name='Convection', state=state, adj_lapse_rate=6.5)
rcms['rcm1'] = climlab.couple([rad,conv,h2o], name='Radiative-Convective Model')
conv = climlab.convection.ConvectiveAdjustment(name='Convection', state=state, adj_lapse_rate=9.8) #lapse rate in degC per km
rcms['rcm2'] = climlab.couple([rad,conv,h2o], name='Radiative-Convective Model')

mod_name=['control','conv-6.5','conv-9.8']
for ai in range(3):
    rcms['rcm'+str(ai)].integrate_years(2)
    plt.plot(rcms['rcm'+str(ai)].Tatm[::-1], rcm.lev[::-1], marker='s', label=mod_name[ai],color=colors[ai])
    plt.plot(rcms['rcm'+str(ai)].Ts, 1000, marker='s',color=colors[ai])
plt.plot(ncep_T, ncep_lev, marker='x',color='k',label='NCEP reanalysis')
plt.gca().invert_yaxis()
plt.title('Sensitivity: convection')
plt.ylabel('Pression (hPa)')
plt.xlabel('Temperature (K)')
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig_name=outpath+'fig5.png'
print('output figure: ', fig_name)
plt.savefig(fig_name,dpi=500,bbox_inches='tight')
plt.close()
