import numpy as np
from numpy import dtype, shape
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from array import array

from sklearn import cluster
from numpy.core.defchararray import center
from matplotlib.pyplot import axis

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib.numpy_pickle_utils import xrange
from numpy import dtype
from array import array


li=[]
#tmp=np.load('C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\classes\\coVec\\00032_Left_002_Corridor.npy')
#print(len(tmp))
#tmp1=np.load('C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\classes\\coVec\\00035_Right_002_Corridor.npy')

#li.append(tmp)
#li.append(tmp1)
#print(li)
#s=np.array([np.array(tmp),np.array(tmp1)])
#s2=np.vstack((tmp,tmp1))
#print(s)
#print(s2)
d1=[]
d2=[]

d1.append(1)
d1.append(2)
d2+=[0]*3
d2+=[2]*2
train=np.load('training_features.npy')
print(shape(train))
print(train)
k=np.sum((train>0)*1,axis=0)
idf = np.array(np.log((1.0*810+1) / (1.0*k + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(train)
train = stdSlr.transform(train)
'''
    
    #count=0
    #SVM = cv2.ml.SVM_create()
   # SVM.setType(cv2.ml.SVM_C_SVC)
    #SVM.setKernel(cv2.ml.SVM_LINEAR)
    #svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR,
     #svm_type = cv2.ml.SVM_C_SVC,
   #  C=2.67, gamma=5.383 )
    #SVM.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
'''
