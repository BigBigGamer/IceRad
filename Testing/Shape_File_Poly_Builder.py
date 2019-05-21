import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import shapefile
import matplotlib.path as mplPath
import pycrs
import toolbar

show_map = 'Yes'    

#reading shapefile
full_path = r'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\Planets_files\2016\2016 12 27\planet_okh_20161227_pl_a'
sf = shapefile.Reader(full_path+'.shp')
#reading projection file
crs = pycrs.load.from_file(full_path+'.prj')

shapes = sf.shapes()
records = sf.records()
length = len(records)

extent = [ 135, 45, 165, 63 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap(extent)

toolbar.polyPlotShapeFile(m,ax,sf,crs)
plt.show()

print('Program end')
