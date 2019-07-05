# Plots track cuts at 'track_num' and 'track_num_w'

import matplotlib.pyplot as plt
import toolbar
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages

plt.rc('text', usetex = True)
plt.rc('font', size=25, family = 'serif')
plt.rc('legend', fontsize=25)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

import numpy as np
from scipy.optimize import curve_fit
import math
import tkinter.filedialog as fd

pathNS = 'd27m12y2016S014815'
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)
size =  thetaNS.shape

for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  

track_num = 162
track_num_w = 250
fig=plt.figure(figsize = (8,6))

sig = sigNS[:,track_num]
theta = thetaNS[:,track_num]
sig_w = sigNS[:,track_num_w]
theta_W = thetaNS[:,track_num_w]
plt.plot(theta,sig,'k.-',label = 'Ice cover')     
plt.plot(theta_W,sig_w,'r.-',label = 'Water')     
plt.xlabel('$\\theta,^{\\circ}$')
plt.ylabel('$\\sigma^0,dB$')
plt.grid(which = 'major')
plt.legend()
plt.savefig('ice_cuts.png', bbox_inches='tight')

plt.show() 