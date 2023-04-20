# A demonstration script for noise reduction of MNIST with RKHM and CNN:
# "Learning in RKHM: a C*-Algebraic Twist for Kernel Machines".

import numpy.random as nr
import random
import numpy.linalg as alg
import scipy.linalg as salg
import tensorflow as tf
from tensorflow.keras import layers, losses, models
from tensorflow.keras.models import Model
import numpy as np
import math
import copy
import cmath
from PIL import Image
import idx2numpy
import os

n=20 # number of training samples
phalf=28 # number of pixels in each column or each row
p=phalf*phalf # number of pixels (phalf x phalf)
lam=10 # regularization parameter for learning in RKHM
qq=2 # number of layers in RKHM
b=0.1 # parameter in RKHM
epochs=1000 # maximum number of epochs for learning CNN

# 1-layer 3 x 3 CNN
class CNN(Model):
  def __init__(self,dim):
    super(CNN, self).__init__()    
    self.model = models.Sequential()
    self.model.add(layers.Conv2D(1, (3, 3), activation='relu', padding="same"))
  
  def call(self, x):
    output = self.model(x)
    
    return output

# optimize parameters of CNN
@tf.function
def train(classifier, data, opt, ylabel):
  with tf.GradientTape(persistent=True) as tape :
    tape.watch(classifier.trainable_variables)
    loss = tf.reduce_mean((ylabel-tf.reshape(classifier.call(data),[n,p]))**2)
  grad = tape.gradient(loss, classifier.trainable_variables)
  opt.apply_gradients(zip(grad, classifier.trainable_variables))
  return loss

# optimize parameters of RKHM
@tf.function
def opti(aa,xmat,ymat,opt):
    with tf.GradientTape(persistent=True) as tape :
        tape.watch(aa)
        
        for i in range(n):
            tmp=tf.matmul(xmat[i*p:(i+1)*p,:],aa[0])+aa[1]
            q,r=tf.linalg.qr(tmp)
            
            if i==0:
                tmpp=tf.matmul(aa[2],tf.transpose(r,(1,0)))
                tmp=tf.matmul(tmpp,tf.eye(p)-b*tf.transpose(q,(1,0)))+aa[3]
                tmp1=tf.matmul(tmpp,tf.matmul(tf.eye(p)-b*tf.transpose(q,(1,0)),tf.matmul(tf.eye(p)-b*tf.transpose(q,(1,0)),tf.eye(p)-b*tf.transpose(q,(1,0)))))
                xmat1=tf.concat([tmp,tmp1],axis=1)
                
            else:
                tmpp=tf.matmul(aa[2],tf.transpose(r,(1,0)))
                tmp=tf.matmul(tmpp,tf.eye(p)-b*tf.transpose(q,(1,0)))+aa[3]
                tmp1=tf.matmul(tmpp,tf.matmul(tf.eye(p)-b*tf.transpose(q,(1,0)),tf.matmul(tf.eye(p)-b*tf.transpose(q,(1,0)),tf.eye(p)-b*tf.transpose(q,(1,0)))))
                xmat1=tf.concat([xmat1,tf.concat([tmp,tmp1],axis=1)],axis=0)

        G1=tf.matmul(xmat1,tf.transpose(xmat1,(1,0)))
        G=G1+lam*tf.eye(n*p)
        GG=tf.matmul(G1,G)
        c=tf.linalg.solve(G,ymat)
        y1=tf.matmul(tf.transpose(c,(1,0)),tf.matmul(G1,ymat))
        y=tf.matmul(tf.transpose(c,(1,0)),tf.matmul(GG,c))-y1-tf.transpose(y1,(0,1))
        loss=tf.norm(y,ord=2)
        
    grad = tape.gradient(loss, aa)
    opt.apply_gradients(zip(grad, aa))
    return loss,xmat1,c

def run_rkhm_cnn():    
    ydata = idx2numpy.convert_from_file('./train-images.idx3-ubyte')/255
    label= idx2numpy.convert_from_file('./train-labels.idx1-ubyte')
    index=[[] for i in range(10)]
    for i in range(10):
        index[i]=list(np.where(label==i))[0]

    nr.seed(0)

    ymat=np.zeros((n*p,p))
    for i in range(n):
        ymat[i*p:(i+1)*p,:]=salg.circulant(ydata[index[i%10][i],:,:].reshape([p]))

    xmat=np.zeros((n*p,p))
    for i in range(n):
        tmp=ydata[index[i%10][i],:,:].reshape([p])
        for k in range(p):
            if True: 
                tmp[k]=tmp[k]+0.01*nr.randn()
        xmat[i*p:(i+1)*p,:]=salg.circulant(tmp)

    xmatt=tf.constant(xmat,dtype=tf.float32)
    ymatt=tf.constant(ymat,dtype=tf.float32)
    nr.seed(0)
    a1=np.zeros(p)
    
    a1[0]=1
    a1[1]=1
    a1[p-1]=1
    
    aa=0.1*salg.circulant(a1)
    aa2=np.zeros((p,p))
    a=tf.Variable(aa,dtype=tf.float32)
    a1=tf.Variable(aa2,dtype=tf.float32)
    a2=tf.Variable(aa,dtype=tf.float32)
    a3=tf.Variable(aa2,dtype=tf.float32)
    aa=[a,a1,a2,a3]
    
    opt=tf.keras.optimizers.Adam(1e-4)

    for kk in range(20):
        loss,xmat11,cc=opti(aa,xmatt,ymatt,opt)
        
        if kk==19:
            xmat1=xmat11.numpy()
            c=cc.numpy()
            xtest1=np.zeros((100*p,qq*p))
            a0=aa[0].numpy()
            a1=aa[1].numpy()
            a2=aa[2].numpy()
            a3=aa[3].numpy()
            
            y=xmat1.dot(xmat1.T.conjugate().dot(c))
            onevec=np.zeros((2,p))
            
            xmat=np.zeros((n,phalf,phalf,1))
            for i in range(n):
                result=np.zeros(p)
                for j in range(round(p)):
                    index1=np.concatenate([np.arange(j,p,1),np.arange(0,j,1)],axis=0)
                    result=result+y[i*p+j,index1]
                
                onevec[0,:]=result/p
                ind=np.where(np.argmax(onevec,axis=0)==1)
                result[ind[0]]=np.zeros(len(ind[0]))
                result=result/np.max(result)
                xmat[i,:,:,0]=result.reshape([phalf,phalf])
            

    ymat=np.zeros((n,p))
    for i in range(n):
        ymat[i,:]=ydata[index[i%10][i],:,:].reshape([p])

    path = "testdata"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    xtest=np.zeros((100,phalf,phalf,1))
    for i in range(100):
        tmp=ydata[index[i%10][i+n],:,:]
        Image.fromarray((255*tmp).astype(np.int8), 'L').save(path+'/original'+str(i)+'.png')
        for k in range(phalf):
            for l in range(phalf):
                if True: 
                    tmp[k,l]=tmp[k,l]+0.01*nr.randn()
        xtest[i,:,:,:]=tmp.reshape([phalf,phalf,1])
        Image.fromarray((255*tmp).astype(np.int8), 'L').save(path+'/noisy'+str(i)+'.png')

    xmat=tf.constant(xmat,dtype=tf.float32)
    xtest=tf.constant(xtest,dtype=tf.float32)
    ymat=tf.constant(ymat,dtype=tf.float32)
    
    cnn = CNN(phalf)
    opt = tf.keras.optimizers.Adam(1e-3)
    cnn.model(tf.zeros((1,phalf,phalf,1), dtype=tf.float32))
    
    path = "results"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        
    for epoch in range(1, epochs + 1):
        loss= train(cnn, xmat, opt, ymat)
        
        if epoch%100==0:
          result=cnn.call(xtest).numpy()
          onevec=np.zeros((2,p))
          err=0

          for i in range(100):
            result1=result[i,:,:,0].reshape([p])
            onevec[0,:]=result1
            ind=np.where(np.argmax(onevec,axis=0)==1)
            result1[ind[0]]=np.zeros(len(ind[0]))
            result1=result1/np.max(result1)
            err=err+alg.norm(result1-ydata[index[i%10][i+n],:,:].reshape([p]))
            yy=ydata[index[i%10][i+n],:,:].reshape([p])
            Image.fromarray((255*(result1.reshape([phalf,phalf]))).astype(np.int8), 'L').save(path+'/result'+str(epoch)+'_'+str(i)+'.png')
            
          print(epoch,"Mean test error:",err/100,flush=True)

if __name__ == '__main__':
    run_rkhm_cnn()
