import numpy as np
import scipy.optimize as sp
import itertools 
from itertools import combinations, chain 

S=20
c=0.98

def accinsr(B,f,chi,*args):

    def rose(h):
        return f(rv(np.array(h),np.array(args),B,chi))


    bds=sp.Bounds(np.zeros(np.array(args).size),np.inf)

    opt=sp.minimize(rose,np.zeros(np.array(args).size),method='L-BFGS-B',bounds=bds)
    if opt['x'].size==1:
        opt['x']=[opt['x']] 
    mes=rv(np.array(opt['x']),np.array(args),B,chi)

    res = {
		"Lay Stakes" : np.round(opt['x'],5),
		"Expected Value" : round(E(mes),2),
		"Standard Deviation" : round(Var(mes)**0.5,2),
		"Sortino Ratio" : -round(Sor(mes),5),
		"Min. Return" : round(min(mes[0]),2),
		"Max Return" : round(max(mes[0]),2)
	}

    print("-" * 50)
    print("Results:")
    if opt['success']==True:
        print("Optimisation terminated successfully.")
    else:
        print("Optimisation failed.")
    print("-" * 50)
    for n,r in res.items():
        print(n,"=",r)
    print("-" * 50)
    return res

def rv(h,L,B,chi):
    t=0
    X=np.empty(2**h.size)
    P=np.empty(2**h.size)
    
    def empt(x,y):
        if x.size==0 and y==0:
            return [0]
        if x.size==0 and y==1:
            return [1]
        else:
            return x

    def PP(x):
        if x.size==h.size:
            return B-1
        elif x.size==h.size-1:
            return chi-1
        else:
            return -1

    for i in range(h.size+1):
        for v in np.array(list(itertools.combinations(range(h.size), i)),dtype=int):
            v_dash=np.delete(range(h.size),v)

            X[t]=(np.sum(empt(h[v_dash],0))*c-np.dot(empt(h[v],0),empt(L[v]-1,0))+S*PP(v))
        
            P[t]=np.product(empt(1/L[v],1))*np.product(empt(1-1/L[v_dash],1))

            t+=1
    return [X,P]

def E(X):
        return np.dot(X[0],X[1])

def Var(X):
        return np.dot(np.power(X[0],2),X[1])-E(X)**2

def DR(X):

    return np.dot(np.multiply(np.power(X[0]-E(X),2),X[0]<0),X[1])**0.5

def Sor(X):

    return -E(X)/DR(X)

def Sha(X):

    return -E(X)/Var(X)**0.5

def Cow(X):
    return -min(X[0])

accinsr(2.074,Sor,0.8,1.29,1.27,1.28,1.27)