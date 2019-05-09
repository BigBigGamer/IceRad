import numpy as np
import pandas as pd
import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
# import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import shapefile
import pyproj
import pycrs
from mpl_toolkits.axes_grid1 import make_axes_locatable

pathNS = 'd27m12y2016S014815'
pathNS = 'd05m04y2017S1411'
LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
LoNS =  np.loadtxt(pathNS+'\LoKu.txt')  
sigNS =  np.loadtxt(pathNS+'\sigKu.txt') 

extent = [ 120,160,50,60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=141.,llcrnrlat=46.,urcrnrlon=146.,urcrnrlat=55.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='h',projection='merc',\
            lat_0=40.,lon_0=-20.,lat_ts=20.)
m.fillcontinents(zorder=0)
m.drawcoastlines(zorder=1)

# draw parallels
# m.drawparallels(np.arange(40,80,3),labels=[1,1,0,1],color = 'grey')
# draw meridians
# m.drawmeridians(np.arange(130,170,3),labels=[1,1,0,1],color = 'grey')   

xm,ym = m(LoNS,LaNS)
im = plt.scatter(xm,ym,5,sigNS, marker = '.',alpha = 1,cmap = 'jet',zorder=3)
# plt.scatter(xm,ym,5,sigNS, marker = '.',alpha = 1,cmap = 'jet',zorder=3)
# im = m.scatter(xm,ym,5,sigNS, marker = '.',alpha =0.9,cmap = 'jet')

# show colorbar 
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("bottom", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax, orientation='horizontal')

plt.show()