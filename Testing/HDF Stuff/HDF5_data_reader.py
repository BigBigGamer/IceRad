import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import matplotlib.path as mplPath
import h5py as hdf

def drawMap(Lo,La,Poly_Color):
    extent = [ 120,130,50,60 ] 
    fig=plt.figure(figsize = (8,6))
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    m = Basemap(llcrnrlon=130.,llcrnrlat=40.,urcrnrlon=160.,urcrnrlat=65.,\
                rsphere=(6378137.00,6356752.3142),\
                resolution='l',projection='merc',\
                lat_0=40.,lon_0=-20.,lat_ts=20.)

    m.drawcoastlines()
    m.drawparallels(np.arange(40,60,10),labels=[1,1,0,1])
    m.drawmeridians(np.arange(130,150,10),labels=[1,1,0,1])   


    # create new figure, axes instances.
    # m.pcolormesh(xm,ym,sigNS,cmap = 'jet')
    # m.scatter(xm,ym,5,sigNS, marker = '.',alpha =0.9,cmap = 'jet')
    # m.imshow(sigNS,extent=extent, alpha=0.6)
    x, y = m(Lo[Poly_Color == b'09'],La[Poly_Color == b'09'])
    m.plot(x,y,'.',markersize = 0.3, color='g')
    x, y = m(Lo[Poly_Color == b'07'],La[Poly_Color == b'07'])
    m.plot(x,y,'.',markersize = 0.3, color='b')
    plt.show()


path = 'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\IceMaps\Ice Data.hdf5'

f = hdf.File(path,'r')
group = f['m03y2017']
ds1 = group['d05m03y2017']
ds_id1 = group['d05m03y2017_ID']
La = ds1[0]
Lo = ds1[1]
Sig = ds1[2]
Theta = ds1[3]
Poly_Color = ds_id1[1]
print(Poly_Color)


# drawMap(Lo,La,Poly_Color)

Theta_s = Theta[Theta<10]
Poly_Color_s = Poly_Color[Theta<10]
Sig_s = Sig[Theta<10]
Poly_Color_s = Theta_s[Theta_s>5]
Sig_s = Sig_s[Theta_s>5]
Theta_s = Theta_s[Theta_s>5]

plt.plot(Theta_s[Poly_Color_s != b'09'],Sig_s[Poly_Color_s != b'09'],'k.')
plt.plot(Theta_s[Poly_Color_s == b'09'],Sig_s[Poly_Color_s == b'09'],'g.')
plt.show()
# print( list(group.keys()) )