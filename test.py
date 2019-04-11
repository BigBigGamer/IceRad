import numpy as np
a = np.array([[True,False,False],[True,False,False],[True,False,True]])
b = np.array([[False,True],[True,False],[False,False]])
c = np.concatenate((a,b),axis = 1)

c = np.array([[],[],[],[]])
print( np.any(a) )

