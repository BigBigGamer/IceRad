import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from mpl_toolkits.mplot3d import Axes3D
import sklearn as sk 
import sklearn.linear_model as lin_mod

# Ice Detecting
def HypApp(xdata,p1,p2,p3):
    return p1 * abs( 1/(abs(xdata) + p2) ) + p3

pathNS = 'd27m12y2016S014815'

# File reading
sigNS = np.loadtxt(pathNS+'\SigKu.txt')
LaNS = np.loadtxt(pathNS+'\LaKu.txt')     
LoNS =  np.loadtxt(pathNS+'\LoKu.txt')    
thetaNS = np.loadtxt(pathNS+'\IncKu.txt')
sigNSun = sigNS 
thetaNSun = thetaNS 
# normalization 
sigNS = np.power(10, sigNS*0.1)
thetaNS = np.tan( thetaNS/180 * np.pi)
parameters = [[],[],[],[]]
mu = [[],[],[],[]]
mu_alt = [[],[],[],[]]
size =  thetaNS.shape
for i in range(0,size[1]):
    mu_up_2,mu_up_3,mu_up_4,mu_down = 0,0,0,0
    m2,m3,m4 = 0,0,0
    # plt.figure(0)
    # plt.imshow(sigNSun,extent=[0,size[1], 0,size[0]],aspect = 'auto',cmap = 'jet')
    # plt.title('Base Data')

    cut_n = i

    y = sigNS[:,cut_n] # sig_0

    x = thetaNS[:,cut_n] # tan
    # moments calculation 

    # mean
    mean = np.sum([ x[j] * y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) / np.sum([ y[j] * np.cos(thetaNSun[j][i])**4  for j in range(0,size[0]) ]) 

    for j in range(0,size[0]):
        mu_up_2 +=(x[j]- mean)**2 * y[j] * np.cos(thetaNSun[j][i])**4
        mu_up_3 += mu_up_2 * (x[j]- mean)
        mu_up_4 += mu_up_3 * (x[j]- mean)
        mu_down += y[j] * np.cos(thetaNSun[j][i])**4

    dispersion = mu_up_2/mu_down
    # print(mu_up_3/mu_down/ dispersion**4)
    skewness = mu_up_3/mu_down / (dispersion)**3
    kurtosis = mu_up_4/mu_down / (dispersion)**4 - 3 
    
    mu[0].append(mean)
    mu[1].append(dispersion)
    mu[2].append(skewness/10**6)
    mu[3].append(kurtosis/10**9)


    # make data positive
    # y_min = np.amin(y)
    # y = y - y_min
    z = x*y
    # print(np.array([x,y]).reshape(49*2,order='F').reshape(49,2))
    regressor = lin_mod.LinearRegression()
    # look up here https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    regressor.fit(np.array([x,y]).reshape(size[0]*2,order='F').reshape(size[0],2),z)
    coeff = regressor.coef_
    intersept = regressor.intercept_

    # print('c,-b = ',coeff,' a+bc = ',intersept)
    # print('So for the parameters we would have:')
    c = coeff[0]
    b = -coeff[1]
    a = intersept - b*c
    mse = ((y - HypApp(x,a,b,c))**2).mean()
    parameters[0].append(a)
    parameters[1].append(b)
    parameters[2].append(c)
    parameters[3].append(mse)
 
    # print('a =',a,' b =',b,' c =',c,' mse =',mse)
# plt.figure(0)

# plt.plot(x,y,'.')
# plt.plot(x,HypApp(x,a,b,c))
# plt.figure(2)
# plt.plot(np.log10(parameters[3]))
# plt.ylim(0,20)
plt.figure(3)
plt.plot(mu[0],label = 'Среднее')
plt.plot(mu[1],label = 'Дисперсия' )
plt.plot(mu[2],label = 'Коэффициент асимметрии')
plt.plot(mu[3],label = 'Коэффициент эксцесса')

plt.legend()
plt.show()

print('Done!')
