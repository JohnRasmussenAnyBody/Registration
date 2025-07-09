# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 19:34:13 2023

@author: jr
"""

import numpy as np
from numpy.polynomial import hermite
import matplotlib.pyplot as plt

mu = 0.0
sigma = 3.0
x = np.linspace(-10, 10, 100)
y = np.exp(-0.5*((x-mu)/sigma)**2)+x

c = hermite.hermfit(x,y,1)

xx = np.linspace(-15, 15, 100)
yy = hermite.hermval(xx, c)

plt.plot(x,y)
plt.plot(xx,yy)
