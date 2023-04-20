# A demonstration script for regression problem of synthetic data with RKHM:
# "Learning in RKHM: a C*-Algebraic Twist for Kernel Machines".

import numpy.random as nr
import random
import numpy.linalg as alg
import scipy.linalg as salg
import numpy as np
import math
import copy
import cmath

n=30 # number of training samples
p=2 # dimension of samples
qq=3 # degree of polynomial kernel
cv=5 # number of folds for a cross validation
sn=6 # number of samples in a fold

def f(x):
    result=np.zeros(p)
    result[0]=np.sin(x[0]+x[1])
    result[1]=np.sin(x[0]+x[1])+np.sin(0.5*(x[0]+x[1]))
    return result

def run_rkhm():
    xdata=np.zeros((n+100,p))
    ydata=np.zeros((n+100,p))
    for i in range(n+100):
        xdata[i,:]=nr.rand(p)
        if n<100:
            ydata[i,:]=f(xdata[i,:]+0.1*nr.randn(p))
        else:
            ydata[i,:]=f(xdata[i,:])
        
    ymat=np.zeros((n*p,p))
    for i in range(n):
        ymat[i*p:(i+1)*p,:]=salg.circulant(ydata[i,:])

    ytest=np.zeros((100*p,p))
    for i in range(100):
        ytest[i*p:(i+1)*p,:]=salg.circulant(ydata[i+n,:])

    xtest=np.zeros((100*p,p))
    for i in range(100):
        xtest[i*p:(i+1)*p,:]=salg.circulant(xdata[i+n,:])

    xmat=np.zeros((n*p,p))
    for i in range(n):
        xmat[i*p:(i+1)*p,:]=salg.circulant(xdata[i,:])


    a=np.eye(p)

    ccmin=10**(-8)
    lammin=10**(-8)
    errmin=10000
    for cvnc in range(16):
        for cvnlam in range(16):
            cc=10**(-8+cvnc)
            lam=10**(-8+cvnlam)
            err=0

            xmat1=np.zeros((n*p,qq*p),dtype=complex)
            for i in range(n):
                q,r=alg.qr(xmat[i*p:(i+1)*p,:].dot(a))
                xmat1[i*p:(i+1)*p,0:p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),1))        
                xmat1[i*p:(i+1)*p,p:2*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),2))
                xmat1[i*p:(i+1)*p,2*p:3*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),3))
                
            for cvnn in range(cv-1):
                index=np.concatenate((np.arange(0,cvnn*sn*p,1,dtype=np.int32),np.arange((cvnn+1)*sn*p,n*p,1,dtype=np.int32)),axis=None)
                G=xmat1[index].dot(xmat1[index].T)+lam*np.eye((n-sn)*p)
                c=alg.solve(G,ymat[index])
                xtest1=np.zeros((sn*p,qq*p),dtype=complex)
                for i in range(sn):
                    q,r=alg.qr(xmat[(cvnn*sn+i)*p:(cvnn*sn+i+1)*p,:].dot(a))
                    xtest1[i*p:(i+1)*p,0:p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),1))
                    xtest1[i*p:(i+1)*p,p:2*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),2))
                    xtest1[i*p:(i+1)*p,2*p:3*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),3))

                y=xtest1.dot(xmat1[index].T.conjugate().dot(c))
                resultall=np.zeros((sn,p))
                for i in range(sn):
                    result=np.zeros(p,dtype=complex)
                    for j in range(p):
                        index=np.concatenate([np.arange(j,p,1),np.arange(0,j,1)],axis=0)
                        result=result+y[i*p+j,index]
                    resultall[i,:]=abs(result/p-ydata[cvnn*sn+i,:])
                err=err+sum(alg.norm(resultall,axis=1))/sn
                
            if err<errmin:
                ccmin=cc
                lammin=lam
                errmin=err                

    cc=ccmin
    lam=lammin
    xmat1=np.zeros((n*p,qq*p),dtype=complex)
    for i in range(n):
        q,r=alg.qr(xmat[i*p:(i+1)*p,:].dot(a))
        xmat1[i*p:(i+1)*p,0:p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),1))
        xmat1[i*p:(i+1)*p,p:2*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),2))
        xmat1[i*p:(i+1)*p,2*p:3*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),3))

    G=xmat1.dot(xmat1.T)+lam*np.eye(n*p)
    c=alg.solve(G,ymat)
        
    xtest1=np.zeros((100*p,qq*p),dtype=complex)
    for i in range(100):
        q,r=alg.qr(xtest[i*p:(i+1)*p,:].dot(a))
        xtest1[i*p:(i+1)*p,0:p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),1))
        xtest1[i*p:(i+1)*p,p:2*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),2))
        xtest1[i*p:(i+1)*p,2*p:3*p]=a.T.conjugate().dot(r.T.conjugate()).dot(alg.matrix_power(np.eye(p)-cc*q.T.conjugate(),3))


    y=xtest1.dot(xmat1.T.conjugate().dot(c))
    resultall=np.zeros((100,p))
    
    for i in range(100):
        result=np.zeros(p,dtype=complex)
        
        for j in range(p):
            index=np.concatenate([np.arange(j,p,1),np.arange(0,j,1)],axis=0)
            result=result+y[i*p+j,index]

        resultall[i,:]=abs(result/p-ydata[i+n,:])

    print("Mean test error:",sum(alg.norm(resultall,axis=1))/100)
    print("c:",cc)
    print("lambda:",lam)
    
if __name__ == '__main__':
    run_rkhm()
