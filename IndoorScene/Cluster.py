
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import cluster
import numpy as np
import cv2
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib.numpy_pickle_utils import xrange

    

if __name__ == '__main__':
    
    codebook=[]
    img_classes=[
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
    train_labels=[]
    desList=[]
    descriptors=[]

    
    dictSize=800
    BOW=cv2.BOWKMeansTrainer(dictSize)
    
    kpNum=0
    desIndex=0
    index=1
    curdir=('C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\')
    imgdir=('C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\')
    for imgClass in img_classes:
        numForEachClass=0
        dir=curdir+imgClass
        imgCurrDir=imgdir+imgClass
        
        for file in os.listdir(dir):
            fn=os.path.basename(file)
            imgfn,ext=os.path.splitext(file)
            exactPath=os.path.join(dir+'\\'+fn)
            tmp=np.load(exactPath)
            imgpth=imgCurrDir+'\\'+imgfn+'.jpeg'
            desList.append(imgpth)
            BOW.add(tmp)
            numForEachClass+=1
        train_labels+=[index]*numForEachClass
        index+=1
    #descriptors=np.empty((kpNum,128),dtype='int')
    sift=cv2.xfeatures2d.SIFT_create()
    np.save('imgpath',desList)
    print(train_labels)
    np.save('train_labels',train_labels)
    
    
                
    dictionary=BOW.cluster()
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    #flann = cv2.FlannBasedMatcher(index_params,search_params)
    sift2 = cv2.xfeatures2d.SIFT_create()
    bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
    bowDiction.setVocabulary(dictionary)
    print ("bow dictionary", np.shape(dictionary))
    np.save('codebook',np.array(dictionary))

    def feature_extract(pth):
        image=cv2.imread(pth,1)
        gray= cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
        return bowDiction.compute(gray,sift.detect(gray))

    train_desc=[] 
    for p in range(len(desList)):
        train_desc.extend(feature_extract(desList[p]))
    
    np.save('training_features',np.array(train_desc))
    print(len(train_desc))
    print(len(train_labels))
    

    #SVM.train(np.array(train_desc),cv2.ml.ROW_SAMPLE,np.array(train_labels))
    #print(SVM)
    #SVM.save('svm_module.dat')
    #SVM.save('svm_module.xml')

  
  
  
  
  