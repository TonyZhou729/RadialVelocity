#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:23:45 2018

@author: Tony Zhou
"""

import matplotlib.pyplot as plt
import numpy as np
from thejoker.data import RVData
from thejoker.sampler import JokerParams, TheJoker
from thejoker.plot import plot_rv_curves
import astropy.units as u
import schwimmbad
from astropy.table import Table

FILENAME = ''
#The String name of the .vels file to be read in. Insert between the single quotes.

def readAndGenerateData():
    """
    Reads a .vels data file and creates three array objects storing the data within the first three columns of the 
    .vels file
    
    Parameters
    ----------
    None
    
    Returns
    -------
    t: list of floats
        Stores the BMJD data
    rv: list of Astropy.units
        Stores the radial velocity in km/s
    err: list of Astropy.units
        Stores the standard deviation of the radial velocity in km/s
    """
    names = ['jd', 'rv', 'rv_err', 'a', 'b', 'c', 'd']
    data = Table.read(FILENAME, names = names, format='ascii.basic')
    data = data.as_array()
    # Converts data from astropy.Table object to np.array object.
    t = []
    rv = []
    err = []
    # Initialize three list objects for generating RVData.
    for x in range(len(data)):
        # Assign values in corresponding columns to individual lists.
        t.append(data[x][0])
        rv.append(data[x][1])
        err.append(data[x][2])
    rv = rv * u.km/u.s
    err = err * u.km/u.s
    return(t,rv,err)

t,rv,err = readAndGenerateData()

data = RVData(t=t, rv=rv, stddev=err)
params = JokerParams(P_min=8*u.day, P_max=512*u.day)
pool = schwimmbad.MultiPool()
joker = TheJoker(params, pool=pool)

samples = joker.rejection_sample(data, n_prior_samples=100000)

fig, ax = plt.subplots(1, 1, figsize=(6,6)) # doctest: +SKIP
ax.scatter(samples['P'].value, samples['K'].to(u.km/u.s).value,
           marker='.', color='k', alpha=0.45) # doctest: +SKIP
ax.set_xlabel("$P$ [day]")
ax.set_ylabel("$K$ [km/s]")
ax.set_xlim(-5, 128)
ax.set_ylim(0.75, 3.)

ax.scatter(61.942, 1.3959, marker='o', color='#31a354', zorder=-100)

fig, ax = plt.subplots(1, 1, figsize=(8,5)) # doctest: +SKIP
t_grid = np.linspace(t[0]-10, t[len(t)-1]+10, 1024)
plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
               plot_kwargs=dict(color='#888888'))
ax.set_xlim(t[0]-5, t[len(t)-1]+5)
