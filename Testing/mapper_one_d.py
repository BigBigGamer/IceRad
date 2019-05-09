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
from tqdm import tqdm
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_path = 'C:\Work\Python\IceRad_Data\PROCESSED_Data'

full_path = r'C:\Work\Python\IceRad_Data\PROCESSED_Data\Planets_files\2016\2016 12 27\planet_okh_20161227_pl_a'
# days = ['19','20','21','22']

extent = [ 120,130,50,60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = Basemap(llcrnrlon=135.,llcrnrlat=41.,urcrnrlon=160.,urcrnrlat=63.,\
            rsphere=(6378137.00,6356752.3142),\
            resolution='l',projection='merc',\
            lat_0=40.,lon_0=-20.,lat_ts=20.)


#reading shapefile
sf = shapefile.Reader(full_path+'.shp')
#reading projection file
crs = pycrs.load.from_file(full_path+'.prj')
ice_shapes_ind = []
shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system        

print(sf)

shapes = sf.shapes()
records = sf.records()
length = len(records)

print('Coordinates re-calculation:')
for i in range(0,length):
    if records[i]['POLY_TYPE'] == 'I':
        if records[i]['COLORSA'] == '09':
            x,y = shapes[i].points[0]
            lo,la = pyproj.transform(shp_proj,std_proj,x,y)
        #     print('Old: ', x,y)
        #     print('New: ',la,lo)
            ice_shapes_ind.append(i)

print('Shapes indeces: ',ice_shapes_ind)

m.drawcoastlines()
m.fillcontinents()
# draw parallels
m.drawparallels(np.arange(40,80,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(130,170,10),labels=[1,1,0,1])   
pathNS = 'd27m12y2016S014815'
sigNS = np.loadtxt(pathNS+'\SigKu.txt')
LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
LoNS =  np.loadtxt(pathNS+'\LoKu.txt')    
thetaNS = np.loadtxt(pathNS+'\IncKu.txt')
size =  thetaNS.shape
for i in range(0,size[1]):
    for j in range(0,size[0]):
        if j < math.floor(size[0]/2):
            thetaNS[j][i] *= -1  
sigNSun = sigNS # dB
thetaNSun = thetaNS 
# normalization 
sigNS = np.power(10, sigNS*0.1)
thetaNS = np.tan( thetaNS/180 * np.pi)
mu = [[],[],[],[]]
# colFlag_h = np.zeros((size),dtype = bool)
colFlag_h = np.zeros((size),dtype = float)
for i in range(0,size[1]):
    mu_up_2,mu_up_3,mu_up_4,mu_down = 0,0,0,0
    m2,m3,m4 = 0,0,0
    cut_n = i
    y = sigNS[:,cut_n] # sig_0
    x = thetaNS[:,cut_n] # tan
    # moments calculation 

    # mean
    mean = np.sum([ x[j] * y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) 

    for j in range(0,size[0]):
        mu_up_2 +=(x[j]- mean)**2 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_3 += (x[j]- mean)**3 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_4 += (x[j]- mean)**4 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_down += y[j] * np.cos(thetaNSun[j][i])**4

    dispersion = np.sqrt(mu_up_2/mu_down)
    # print(dispersion)
    skewness = mu_up_3/mu_down / (dispersion)**3
    kurtosis = mu_up_4/mu_down / (dispersion)**4 - 3 
    
    mu[0].append(mean)
    mu[1].append(dispersion)
    mu[2].append(skewness)
    mu[3].append(kurtosis)
    colFlag_h[:,i] = kurtosis
    # if kurtosis > 2*10**8:
    #     # print(kurtosis)
    #     colFlag_h[:,i] = True
    # else:
    #     colFlag_h[:,i] = False

xm,ym = m(LoNS,LaNS)
# print(colFlag_h[:,150])
im = plt.scatter(xm,ym,25,colFlag_h, marker = '.',alpha = 1,cmap = 'jet_r')

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
                # print('plottoing',x_lon)

divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, orientation='horizontal')

plt.show()

# create new figure, axes instances.
# Tk().withdraw()
# pathNS = fd.askdirectory() 
# pathNS = 'DataProcessing/Hydro/NS/m03y2018/d04m03y2018S133003'
# LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
# LoNS =  np.loadtxt(pathNS+'\LoKu.txt')  
# sigNS =  np.loadtxt(pathNS+'\sigKu.txt') 
# x, y = m(df['Lon'].values.tolist(),df['Lat'].values.tolist())
# # xm,ym = m(LoNS,LaNS)
# # m.pcolormesh(xm,ym,sigNS,cmap = 'jet')
# # m.scatter(xm,ym,5,sigNS, marker = '.',alpha =0.9,cmap = 'jet')
# # m.imshow(sigNS,extent=extent, alpha=0.6)
# m.plot(x,y,'.',markersize = 0.3,color='r')
# plt.title(days[0]+'.'+ month + '.' + year  )
# plt.show()