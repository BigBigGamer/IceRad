import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import matplotlib.path as mplPath
import h5py as hdf

big_ds = np.array([[],[],[],[]])
big_ds_id = np.array([[],[],[],[],[],[],[]])

path = 'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\IceMaps\Ice Data01.hdf5'

f = hdf.File(path,'r+')
group = f['m01y2017']
for ds_name in group:
    day = np.string_(ds_name[1:3])
    month = np.string_(ds_name[4:6])
    year = np.string_(ds_name[7:11])
    if '_ID' in ds_name:
        shape = group[ds_name].shape
        date_ = np.array([[day,month,year]]*shape[1]).T
        ids = np.concatenate((group[ds_name],date_),axis = 0)
        big_ds_id = np.concatenate((big_ds_id,ids),axis = 1)
        print(big_ds_id.shape)
    else:
        big_ds = np.concatenate((big_ds,group[ds_name]),axis = 1)
        print(big_ds.shape)


d = f.create_dataset('data01',data = big_ds,dtype = 'f')
idss = f.create_dataset('data01_id',data = big_ds_id,dtype = 'S10')
# ds_id1 = group['d01m01y2017_ID']
# Data = ds1
# ID = ds_id1
