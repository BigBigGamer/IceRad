# file  = open('names2018_03.txt', 'r') 
# comfile = open('comfile2018_03.txt','w')
# lines = file.readlines()
# path = 'E:/Work/GitHub/iceRad/Data/2018_03/'
# for i in lines:
#     comfile.write(path + i )
# file.close()
# comfile.close()
import numpy as np

a = [[1,2,3,4,5],['a','b','c','e','f']]
a = np.array(a)
a = a.transpose().tolist()
print(a)