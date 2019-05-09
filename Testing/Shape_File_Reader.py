# This program reads a .shp file, finds needed shapes by specific attributes and prints out list of indeces of needed
# shapes. Also the script shows how to use pycrs and pyproj to work with different map projections
# Additionaly, you can set show_map parameter to "Yes"? which will show the map of .shp file

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import shapefile
import matplotlib.path as mplPath
import pyproj
import pycrs

show_map = 'Yes'    

#reading shapefile
sf = shapefile.Reader('2017 03 28\planet_okh_20170328_pl_a.shp') # указать путь к shape файлу
#reading projection file
crs = pycrs.load.from_file('2017 03 28\planet_okh_20170328_pl_a.prj') # указать путьь к файлу проекции
shp_proj = pyproj.Proj(crs.to_proj4()) #.shp file projection
std_proj = pyproj.Proj(init = 'epsg:4326') # Lat/Lon system

print(sf)

shapes = sf.shapes()
records = sf.records()
length = len(records)

ice_shapes_ind = []

print('Coordinates re-calculation:')
for i in range(0,length):
    if records[i]['POLY_TYPE'] == 'I':
        if records[i]['COLORSA'] == '09':
            x,y = shapes[i].points[0]
            lo,la = pyproj.transform(shp_proj,std_proj,x,y)
            print('Old: ', x,y)
            print('New: ',la,lo)
            ice_shapes_ind.append(i)


print('Shapes indeces: ',ice_shapes_ind)

if show_map=='Yes':
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
                plt.plot(x_lon,y_lat)
    plt.show()

print('Program end')