import matplotlib.pyplot as plt 
import numpy as np 
import statsmodels.api as sm 
import sklearn.linear_model as lm 
import math 


sigNS = np.loadtxt('SigKu.txt') 

thetaNS = np.loadtxt('IncKu.txt') 
theta1NS = thetaNS 
thetaNS = np.power(np.tan(theta1NS/180*math.pi),2) 
sigNS = np.power(10,sigNS*0.1) * np.power(np.cos(theta1NS/180*math.pi),4) 
tanth = np.tan(theta1NS/180*math.pi) 

nc = 350 

ns = 1 
nf = 49 
npt = nf-ns 

x1 = [[0 for i in range(2)] for j in range(npt)] 
z = [0 for j in range(npt)] 


th = (np.array(thetaNS[ns:nf,nc])) 
sig = (np.array(sigNS[ns:nf,nc])) 
print(np.shape(th)) 
for i in range(npt): 
    x1[i][0] = th[i] 
    x1[i][1] = sig[i] 
    z[i] = sig[i] * th[i] 


x = np.array(x1) 
y = (np.array(z)) 

# добавим фиктивную переменную для расчета intercept'а 
x_ = sm.add_constant(x) 

# создаем модель для метода обычных наименьших квадратов (Ordinary Least Squares) 
smm = sm.OLS(y, x_) 
# запускаем расчет модели 
res = smm.fit() 
# теперь выведем параметры рассчитанной модели 
print(res.params) 

#plt.plot(theta1NS[ns:nf,nc],sigNS[ns:nf,nc],'.')#10*np.log10(sigNS[ns:nf,nc]),'.') 
c = res.params[1] 
b = -res.params[2] 
a = res.params[0] - b*c 
#plt.plot(theta1NS[ns:nf,nc], a/(thetaNS[ns:nf,nc]+b)+c) #10*np.log10(a/(thetaNS[ns:nf,nc]+b)+c)) 

#plt.plot(thetaNS[ns:nf,nc], 1/np.power(c*(np.power(thetaNS[ns:nf,nc],2)+b),1.5)) 


# plt.imshow(sigNS); 
#plt.colorbar()

import numpy as np 
import matplotlib.pyplot as plt 

nsc = np.shape(tanth)[0] 
nal = np.shape(tanth)[1] 
print(nsc) 

A = [[0 for i in range (3)] for j in range (nsc)] 


Bd = [0 for i in range (nal)] 
skewness = [0 for i in range (nal)] 
kurtosis = [0 for i in range (nal)] 

for j in range (nal): 
    for i in range (nsc): 
        A[i][1] = tanth[i][j] 
        A[i][2] = sigNS[i][j] * np.power(np.cos(np.arctan(A[i][1])/180*math.pi),4) 

    S1 = 0 
    SS0 = 0 

    for i in range(nsc): 
        S1=A[i][1]*A[i][2]+S1 
        SS0=SS0+A[i][2] 

    fcm1=S1/SS0 

    S2=0 
    for i in range (nsc): 
        S2=(A[i][1])**2*A[i][2]+S2  

    S200=0 
    for i in range (nsc): 
        S200=(A[i][1]-fcm1)**2*A[i][2]+S200 

    S3=0 
    for i in range (nsc): 
        S3=(A[i][1]-fcm1)**3*A[i][2]+S3 

    S4=0 
    for i in range (nsc): 
        S4=(A[i][1]-fcm1)**4*A[i][2]+S4 

    Bd[j] = np.sqrt(S2/SS0-fcm1**2) 

    M3 = S3/SS0 
    M4 = S4/SS0 
    Bd[j] = np.sqrt(S200/SS0) 
    skewness[j] = M3/Bd[j]**3 
    kurtosis[j] = M4/Bd[j]**4 - 3 
plt.plot(kurtosis)
plt.show()