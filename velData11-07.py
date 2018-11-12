#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 21:23:45 2018

@author: tonyzhou
"""

import matplotlib.pyplot as plt
import numpy as np
from thejoker.data import RVData
from thejoker.sampler import JokerParams, TheJoker
from thejoker.plot import plot_rv_curves
import astropy.units as u
import schwimmbad
from astropy.table import Table

FILENAME = 'G60-06_KECK.vels'

def readAndGenerateData():
    names = ['jd', 'rv', 'rv_err', 'a', 'b', 'c', 'd']
    data = Table.read(FILENAME, names = names, format='ascii.basic')
    print(data)
    data = data.as_array()
    # Converts data from astropy.Table object to np.array object.
    t = []
    rv = []
    err = []
    # Initialize three np.array objects for generating RVData.
    for x in range(len(data)):
        # Assign values in corresponding columns to individual numpy arrays.
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

samples = joker.rejection_sample(data, n_prior_samples=10)

fig, ax = plt.subplots(1, 1, figsize=(6,6)) # doctest: +SKIP
ax.scatter(samples['P'].value, samples['K'].to(u.km/u.s).value,
           marker='.', color='k', alpha=0.45) # doctest: +SKIP
ax.set_xlabel("$P$ [day]")
ax.set_ylabel("$K$ [km/s]")
ax.set_xlim(-5, 128)
ax.set_ylim(0.75, 3.)

ax.scatter(61.942, 1.3959, marker='o', color='#31a354', zorder=-100)

fig, ax = plt.subplots(1, 1, figsize=(8,5)) # doctest: +SKIP
t_grid = np.linspace(-10, 210, 1024)
plot_rv_curves(samples, t_grid, rv_unit=u.km/u.s, data=data, ax=ax,
               plot_kwargs=dict(color='#888888'))
ax.set_xlim(-5, 205)
