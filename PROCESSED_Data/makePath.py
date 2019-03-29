file  = open('names2018_03.txt', 'r') 
comfile = open('comfile2018_03.txt','w')
lines = file.readlines()
path = 'E:/Work/GitHub/iceRad/Data/2018_03/'
for i in lines:
    comfile.write(path + i )
file.close()
comfile.close()