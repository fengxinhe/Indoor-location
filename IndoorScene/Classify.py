import numpy as np
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import cluster
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib.numpy_pickle_utils import xrange
import os

test_classes=[
                 "mr",
                 "ra",
                 "so",
                 "to",
                 "co",
                 "lmr",
                 "lo",
                 "pa",
                 "el"
                ]

imgpaths=np.load('data\imgpath.npy')
dictionary=np.load('data\codebook.npy')
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

test_pth=[]

testdir=('C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\testimg\\')

for imgClass in test_classes:
    subDir=testdir+imgClass
    for file in os.listdir(subDir):
         imgfn,ext=os.path.splitext(file)
         imgpth=subDir+'\\'+imgfn+'.jpeg'
         test_pth.append(imgpth)


sift = cv2.xfeatures2d.SIFT_create()
bowDiction = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
bowDiction.setVocabulary(dictionary)

test_desc=[]

   


def feature_extract(pth):
    image=cv2.imread(pth,1)
    gray= cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
    return bowDiction.compute(gray,sift.detect(gray))


for p in test_pth:
    test_desc.extend(feature_extract(p))
    
test_desc=np.array(test_desc)

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_desc > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(test_pth)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_desc = stdSlr.transform(test_desc)

print(len(test_desc))
#predictions= [test_classes[i] for i in clf.predict(test_desc)]
pre=clf.predict(test_desc)

print(pre)





