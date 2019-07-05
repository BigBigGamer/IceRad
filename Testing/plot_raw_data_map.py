import numpy as np
import pandas as pd
import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'

from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', size=25, family = 'serif')
plt.rc('legend', fontsize=16)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')


import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import shapefile
import pyproj
import pycrs
import toolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

pathNS = 'd27m12y2016S014815'
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)
LoNS = LoNS[:,80:300]
LaNS = LaNS[:,80:300]
sigNS = sigNS[:,80:300]
thetaNS = thetaNS[:,80:300]
# extent = [ 135, 45, 165, 63 ] 
extent = [ 138, 54, 149, 60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap(extent)

# draw parallels
m.drawparallels(np.arange(45,65,2),labels=[1,0,0,1],color = 'grey')
# draw meridians
m.drawmeridians(np.arange(135,165,3),labels=[1,0,0,1],color = 'grey')   

xm,ym = m(LoNS,LaNS)
im = plt.scatter(xm,ym,30,sigNS, marker = '.',alpha = 1,cmap = 'BuGn_r',zorder=3)
# im = plt.scatter(xm,ym,5,sigNS, marker = '.',alpha = 1,cmap = 'jet',zorder=3)

# show colorbar 
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')

cbar.ax.set_title('$\sigma^0,dB$')

plt.savefig('imgs/11.png', bbox_inches='tight')

plt.show()