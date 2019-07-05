# Plots Kurtosis track over map with borders
import numpy as np
import pandas as pd
# import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
# import tkinter.filedialog as fd
import matplotlib.pyplot as plt

from matplotlib import rc
plt.rc('text', usetex = True)
plt.rc('font', size=18, family = 'serif')
plt.rc('legend', fontsize=14)
plt.rc('text.latex', preamble=r'\usepackage[russian]{babel}')

import shapefile
import pyproj
import pycrs
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import toolbar

data_path = 'E:\Work\GitHub\IceRad_Data\PROCESSED_Data'

full_path = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\Planets_files\2016\2016 12 27\planet_okh_20161227_pl_a'
# days = ['19','20','21','22']

extent = [ 138, 55, 148, 60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap(extent)

#reading shapefile
sf = shapefile.Reader(full_path+'.shp')
#reading projection file
crs = pycrs.load.from_file(full_path+'.prj')
shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system        

print(sf)

shapes = sf.shapes()
records = sf.records()
length = len(records)

# draw parallels
m.drawparallels(np.arange(45,65,2),labels=[1,0,0,1],color = 'grey')
# draw meridians
m.drawmeridians(np.arange(135,165,2),labels=[1,0,0,1],color = 'grey')   

pathNS = 'd27m12y2016S014815'
sigNS, LaNS, LoNS, thetaNS = toolbar.readFolder(pathNS)
LoNS = LoNS[:,80:]
LaNS = LaNS[:,80:]
sigNS = sigNS[:,80:]
thetaNS = thetaNS[:,80:]

size =  thetaNS.shape

for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  
sigNSun = sigNS # dB
sigNSun_up,sigNSun_down = toolbar.getHalfs(sigNSun)

thetaNSun = thetaNS 
thetaNSun_up,thetaNSun_down = toolbar.getHalfs(thetaNSun)


# normalization 
sigNS = np.power(10, sigNS*0.1)
sigNS_up,sigNS_down = toolbar.getHalfs(sigNS)

thetaNS = np.tan( thetaNS/180 * np.pi)
thetaNS_up,thetaNS_down = toolbar.getHalfs(thetaNS)

mu_up = [[],[],[],[]]
mu_down = [[],[],[],[]]

size_half =  thetaNS_up.shape
colFlag_h = np.zeros((size),dtype = bool)
colFlag_h_up = np.zeros((size_half),dtype = bool)
colFlag_h_down = np.zeros((size_half),dtype = bool)

for i in range(0,size_half[1]):
    mu_up_2,mu_up_3,mu_up_4,mu_downs = 0,0,0,0
    # m2,m3,m4 = 0,0,0
    cut_n = i
    y = sigNS_up[:,cut_n] # sig_0
    x = thetaNS_up[:,cut_n] # tan

    mean_up = np.sum([ x[j] * y[j] * np.cos(thetaNSun_up[j][i])**4  for j in range(0,size_half[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun_up[j][i])**4  for j in range(0,size_half[0]) ]) 

    for j in range(0,size_half[0]):
        mu_up_2 += (x[j]- mean_up)**2 * y[j] * np.cos(thetaNSun_up[j][i])**4
        # mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_4 += (x[j]- mean_up)**4 * y[j] * np.cos(thetaNSun_up[j][i])**4
        mu_downs += y[j] * np.cos(thetaNSun_up[j][i])**4

    dispersion_up = np.sqrt(mu_up_2/mu_downs)
    # skewness = (mu_up_3/mu_down) / (dispersion**3)
    kurtosis_up = (mu_up_4/mu_downs) / (dispersion_up**4) - 3 
    
    mu_up[0].append(mean_up)
    mu_up[1].append(dispersion_up)
    # mu_up[2].append(skewness)
    mu_up[3].append(kurtosis_up)



    mu_up_2,mu_up_3,mu_up_4,mu_downs = 0,0,0,0
    y = sigNS_down[:,cut_n] # sig_0
    x = thetaNS_down[:,cut_n] # tan

    mean_down = np.sum([ x[j] * y[j] * np.cos(thetaNSun_down[j][i])**4  for j in range(0,size_half[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun_down[j][i])**4  for j in range(0,size_half[0]) ]) 

    for j in range(0,size_half[0]):
        mu_up_2 += (x[j]- mean_down)**2 * y[j] * np.cos(thetaNSun_down[j][i])**4
        # mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_4 += (x[j]- mean_down)**4 * y[j] * np.cos(thetaNSun_down[j][i])**4
        mu_downs += y[j] * np.cos(thetaNSun_down[j][i])**4

    dispersion_down = np.sqrt(mu_up_2/mu_downs)
    # skewness = (mu_up_3/mu_downs) / (dispersion**3)
    kurtosis_down = (mu_up_4/mu_downs) / (dispersion_down**4) - 3 
    
    mu_down[0].append(mean_down)
    mu_down[1].append(dispersion_down)
    # mu_down[2].append(skewness)
    mu_down[3].append(kurtosis_down)



# Merge up-half and down-half
kurtosis  = np.empty(size)
kurtosis[0:25,:] = mu_up[3]
kurtosis[24:,:] = mu_down[3]
colFlag_h = kurtosis


toolbar.polyPlotShapeFile(m,ax,sf,crs)

xm,ym = m(LoNS,LaNS)
# im = plt.scatter(xm[colFlag_h>10],ym[colFlag_h>10],30,colFlag_h[colFlag_h>10], marker = '.',alpha = 1,cmap = 'jet_r')
im = plt.scatter(xm,ym,40,colFlag_h, marker = '.',alpha = 1, cmap = 'viridis')




# border plotting
if True:
    for some_shape in list(sf.iterShapes()):
        npoints=len(some_shape.points)
        nparts=len(some_shape.parts)

        if nparts == 1:
            length = len(some_shape.points)
            x_lon = np.zeros((length,1))
            y_lat = np.zeros((length,1))
            for i in range(0,length):
                x_lon[i] = some_shape.points[i][0]
                y_lat[i] = some_shape.points[i][1]
            plt.plot(x_lon,y_lat)
        else:
            for j in range(nparts):
                i0 = some_shape.parts[j]
                if j < nparts-1:
                    i1 = some_shape.parts[j+1]-1
                else:
                    i1 = npoints
                seg = some_shape.points[i0:i1+1]
                x_lon = np.zeros((len(seg),1))
                y_lat = np.zeros((len(seg),1))
                for i in range(0,len(seg)):
                    x_lon[i] = some_shape.points[i][0]
                    y_lat[i] = some_shape.points[i][1]
                x_lon,y_lat = pyproj.transform(shp_proj,std_proj,x_lon,y_lat)
                x_lon,y_lat = m(x_lon,y_lat)
                c = 'b'
                m.scatter(x_lon,y_lat,0.5,marker = '.',color = c)

# colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.set_xlabel('$\gamma_2$')

plt.savefig('imgs/51.png', bbox_inches='tight',dpi=900)


plt.show()
