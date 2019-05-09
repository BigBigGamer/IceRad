import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os
import matplotlib.path as mplPath
import h5py as hdf
import pandas as pd

path = 'E:\Work\GitHub\IceRad_Data\PROCESSED_Data\IceMaps\Ice Data01.hdf5'

f = hdf.File(path,'r')
data = f['data01']
ids = f['data01_id']

f_pd = pd.read_hdf(path,mode='r')
