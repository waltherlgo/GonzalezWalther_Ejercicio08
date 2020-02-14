import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import itertools
import sklearn.linear_model
data = np.loadtxt("data_to_fit.txt", skiprows=1)
x=data[:,0]
def model_A(x,params):
    y=params[0]+x*params[1]+params[2]*x**2
    return y
def model_B(x,params):
    y=params[0]*(np.exp(-0.5*(x-params[1])**2/params[2]**2))
    return y
def prob(y_m):
    P=np.sum(-1/(2*data[:,2]**2) *(y_m-data[:,1])**2)
    return P
def metro(PAR,model):
    PARN=PAR+np.random.normal(loc=0.0,scale=0.02,size=3)
   # while np.sum(PARN<0)>0:
   #     PARN=PAR+np.random.normal(loc=0.0,scale=0.05,size=3)  
    A=np.min([1,np.exp(prob(model(x,PARN))-prob(model(x,PAR)))])
    #print(prob(PARN) , prob(PAR))
    r=np.random.random()
    if r<A:
       # print(PARN)
        PAR=PARN
        Prob=prob(model(x,PARN))
    else:
        Prob=prob(model(x,PAR))
    return PAR,Prob
N=80000
PARA=[-15,0,2] #np.random.random(size=3)
PARB=np.random.random(size=3)
ProbA=np.zeros(N)
ProbB=np.zeros(N)
PARAL=np.zeros((N,3))
PARBL=np.zeros((N,3))
for i in range(N):
    [PARAL[i,:],ProbA[i]]=metro(PARA,model_A) 
    [PARBL[i,:],ProbB[i]]=metro(PARB,model_B) 
    PARA=PARAL[i,:]
    PARB=PARBL[i,:]
MPARAL=PARAL[int(N/2):,:]
MPARBL=PARBL[int(N/2):,:]
plt.figure(figsize=(12,12))
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.hist(MPARAL[:,i])
    plt.title(r'$\beta_%1.0f= $'%i +'%4.3f'%np.mean(MPARAL[:,i])+'$\pm $ %4.3f' %np.std(MPARAL[:,i]))
    plt.xlabel(r'$\beta_%1.0f$'%i)
for i in range(3):
    plt.subplot(2,3,3+i+1)
    plt.hist(MPARBL[:,i])
    plt.title(r'$\beta_%1.0f= $'%i +'%4.3f'%np.mean(MPARBL[:,i])+'$\pm $ %4.3f' %np.std(MPARBL[:,i]))
    plt.xlabel(r'$\beta_%1.0f$'%i)
plt.savefig("Distribuciones.png")
plt.show()
plt.figure()
plt.scatter(x,model_B(x,np.mean(MPARBL,axis=0)))
plt.scatter(x,model_A(x,np.mean(MPARAL,axis=0)))
plt.scatter(x,data[:,1])
plt.legend(['modelo B','modelo A', 'datos'])
plt.savefig("Predicciones.png")
plt.show()
PA=np.mean(np.exp(ProbA))
PB=np.mean(np.exp(ProbB))
print("La evidencia de cada modelo es  A:%4.3e"%PA+" y B:%4.3e" %PB +" \nEntonces el modelo más probable es:  "+NAMES[np.argmax([PA,PB])])