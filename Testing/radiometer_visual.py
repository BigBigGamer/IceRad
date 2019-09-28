
# Radiometer-data visualizer script. Takes raw data and plots it

import numpy as np
import matplotlib.pyplot as plt 
import os
# os.environ['PROJ_LIB'] = r'E:/Anaconda/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
import toolbar
# import tkinter.filedialog as fd
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time

start = time.time()
print('Start')
path = 'E:\Work\GitHub\IceRad_Data\Radiometer_data\m12y2016\d27m12y2016S014815'

areaS1=np.loadtxt(path+r'\areaS1.txt')
areaS2=np.loadtxt(path+r'\areaS2.txt')

LaS1=areaS1[:,0]                        
LoS1=areaS1[:,1]  

LaS2=areaS2[:,0]                        
LoS2=areaS2[:,1]

TcS1 = areaS1[:,2:10]
TcS2 = areaS1[:,2:5]

Boundries=[40, 140, 20, 115]

# %Boundries=[64, 168, 40, 132]; %define boundries of the needed area

# S1_titles={'10.65 GHz V-Pol','10.65 GHz H-Pol','18.7 GHz V-Pol',
#            '18.7 GHz H-Pol','23.8 GHz V-Pol','36.64 GHz V-Pol',
#            '36.64 GHz H-Pol','89.0 GHz V-Pol ','89.0 GHz H-Pol'};

# S2_titles={'166.0 GHz V-Pol','166.0 GHz H-Pol','183.31 +/-3 GHz V-Pol','183.31 +/-7 GHz V-Pol'};

la1 = 40
la2 = 64
lo1 = 132
lo2 = 168

extent = [ 120,130,50,60 ] 
fig=plt.figure(figsize = (8,6))
ax=fig.add_axes([0.1,0.1,0.8,0.8])
# setup mercator map projection.
m = toolbar.makeMap([135,45,160,63])
# draw parallels
m.drawparallels(np.arange(40,80,10),labels=[1,1,0,1])
# draw meridians
m.drawmeridians(np.arange(130,170,10),labels=[1,1,0,1])   


# print(TcS1.shape)
# print(LaS1.shape)

xm,ym = m(LoS1,LaS1)
im=plt.scatter(xm,ym,5,TcS1[:,6],cmap = 'jet')
# plt.title(1)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("bottom", size="5%", pad=0.05)
plt.colorbar(im, ax=ax, orientation='vertical')
end = time.time()
print(end-start)
plt.show()
        #    % for S1 
        #    % 1) 10.65 GHz V-Pol 
        #    % 2) 10.65 GHz H-Pol
        #    % 3) 18.7 GHz V-Pol
        #    % 4) 18.7 GHz H-Pol
        #    % 5) 23.8 GHz V-Pol 
        #    % 6) 36.64 GHz V-Pol
        #    % 7) 36.64 GHz H-Pol
        #    % 8) 89.0 GHz V-Pol 
        #    % 9) 89.0 GHz H-Pol
                    #   
        #    % for S2
        #    % 1) 166.0 GHz V-Pol 
        #    % 2) 166.0 GHz H-Pol
        #    % 3) 183.31 +/-3 GHz V-Pol and 
        #    % 4) 183.31 +/-7 GHz V-Pol
