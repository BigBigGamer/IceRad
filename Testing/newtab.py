import numpy as np
import pandas as pd
# from tkinter import *
import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
# import tkinter.filedialog as fd
import matplotlib.pyplot as plt

df = pd.read_csv('planet_okh_20180305_pl_a.txt',sep=';')
# create new figure, axes instances.
# Tk().withdraw()
# pathNS = fd.askdirectory() 
pathNS = 'DataProcessing/Hydro/NS/m03y2018/d04m03y2018S133003'
LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
LoNS =  np.loadtxt(pathNS+'\LoKu.txt')  
sigNS =  np.loadtxt(pathNS+'\sigKu.txt') 
extent = [ 120,130,50,60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=130.,llcrnrlat=40.,urcrnrlon=160.,urcrnrlat=65.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lat_0=40.,lon_0=-20.,lat_ts=20.)

m.drawcoastlines()
m.fillcontinents()
# draw parallels
m.drawparallels(np.arange(40,60,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(130,150,10),labels=[1,1,0,1])
x, y = m(df['Lon'].values.tolist(),df['Lat'].values.tolist())
xm,ym = m(LoNS,LaNS)
m.pcolormesh(xm,ym,sigNS,cmap = 'jet')
m.scatter(xm,ym,5,sigNS, marker = '.',alpha =0.9,cmap = 'jet')
# m.imshow(sigNS,extent=extent, alpha=0.6)
m.plot(x,y,'.',markersize = 0.3,color='r')
plt.show()
# fig.savefig('map.png',dpi = 1000)

