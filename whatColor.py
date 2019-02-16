import pandas as pd
import numpy as np
# from tkinter import *
from mpl_toolkits.basemap import Basemap
# import tkinter.filedialog as fd
import matplotlib.pyplot as plt

df = pd.read_csv('planet_okh_20180305_pl_a.txt',sep=';')
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=130.,llcrnrlat=30.,urcrnrlon=160.,urcrnrlat=65.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='stere',\
            lat_0=40.,lon_0=160.,lat_ts=20.)

m.drawcoastlines()
m.fillcontinents()
# draw parallels
m.drawparallels(np.arange(40,60,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(130,150,10),labels=[1,1,0,1])
Lon = df['Lon'].values.tolist()
Lat = df['Lat'].values.tolist()
PolyType = df['POLY_TYPE'].values.tolist()
Id = df['ID'].values.tolist()
ColorSA = df['COLORSA'].values.tolist()
# Colors:
# 3 - Тонкий однолетний белый лед.
# 4 - Нилас?
# 6 - Серый лед 10-15
# 7 - Серо-белый лед 15-30
# 9 - Тонкий лед 30-70 +++
# 12 - Однолетний лед 30-200
# 19 - Однолетний лед средней толщины 70-120
# 40 - Нилас?
variants = [19]
wLat =[]
wLon =[]
ind = 0
for i in ColorSA:
    # if not (i in variants):
    #     if PolyType[ind] == 'I' :
    #         variants.append(i)
    if (i == 9 ):
        wLat.append(Lat[ind])
        wLon.append(Lon[ind])
    ind += 1

x, y = m(wLon,wLat)
m.plot(x,y,'.',markersize = 0.3, color='g')
plt.show()
print(variants)