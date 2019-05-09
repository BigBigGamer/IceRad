import numpy as np
import pandas as pd
import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import shapefile
import pyproj
import pycrs
import toolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

pathNS = 'd27m12y2016S014815'
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)

extent = [ 140, 40, 156, 60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap(extent)

# draw parallels
# m.drawparallels(np.arange(40,80,3),labels=[1,1,0,1],color = 'grey')
# draw meridians
# m.drawmeridians(np.arange(130,170,3),labels=[1,1,0,1],color = 'grey')   

xm,ym = m(LoNS,LaNS)
im = plt.scatter(xm,ym,5,sigNS, marker = '.',alpha = 1,cmap = 'jet',zorder=3)

# show colorbar 
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, orientation='horizontal')

plt.show()