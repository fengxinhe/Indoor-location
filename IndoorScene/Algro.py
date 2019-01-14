from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib.numpy_pickle_utils import xrange
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from _overlapped import NULL
from pyexpat import model
import os
import cv2
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,\
    QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score


image_paths=np.load('imgpath2.npy')
train_labels=np.load('train_labels2.npy')
img_classes=[
                 "a",
                 "b",
                 "c",
                 "d",
                 "e",
                 "f",
                 "g",
                 "h",
                 "i",
                 "j",
                 "k",
                 "l"
                ]
curdir=('C:\\Users\\qq1\\Desktop\\LibraryScene\\surftrainset\\')
imgdir=('C:\\Users\\qq1\\Desktop\\LibraryScene\\')
testdir=('C:\\Users\\qq1\\Desktop\\LibraryScene\\test\\')
res=[]


def test(spth,classify_flag):
    clf, classes_names, stdSlr  = joblib.load(spth+"/"+classify_flag+"bof.pkl")
    codebook=np.load(spth+'/codebook.npy')
    test_pth=[]


    test_Labels=[]

    cc=1
    for imgClass in img_classes:
        subDir=testdir+imgClass
        imgcount=0
        for file in os.listdir(subDir):
            imgfn,ext=os.path.splitext(file)
            imgpth=subDir+'\\'+imgfn+'.jpg'
            test_pth.append(imgpth)
            imgcount+=1
        test_Labels+=[cc]*imgcount
        cc+=1

    sift = cv2.xfeatures2d.SURF_create()
    bowDiction = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    bowDiction.setVocabulary(codebook)
    #print(len(test_Labels))
    #test_desc=[]   


    def feature_extract(pth):
        image=cv2.imread(pth,1)
        gray= cv2.cvtColor(image,cv2.IMREAD_GRAYSCALE)
        return bowDiction.compute(gray,sift.detect(gray))


    #for p in test_pth:
       # test_desc.extend(feature_extract(p))
    
    #test_desc=np.array(test_desc)
    #np.save(spth+"/test_features",test_desc)
    test_desc=np.load(spth+"/test_features.npy")
    nbr_occurences = np.sum( (test_desc > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(test_pth)+1) / (1.0*nbr_occurences + 1)), 'float32')

    test_desc = stdSlr.transform(test_desc)

    #print(len(test_desc))
    pre=clf.predict(test_desc)
    #print(metrics.classification_report(test_Labels,pre))
    #print(metrics.confusion_matrix(test_Labels,pre))
    #np.save("C:\\Users\\qq1\\Desktop\\predict-report\\"+classify_flag+spth,
    #        metrics.classification_report(test_Labels,pre))
    
   # np.save("C:\\Users\\qq1\\Desktop\\predict-confusion\\"+classify_flag+spth,
    #        metrics.confusion_matrix(test_Labels,pre))

    #print(pre)
    #print(test_Labels)
    
    correct = 0
    for i in range(len(test_desc)):
  
        if test_Labels[i]==pre[i]:
            correct+=1
    scores=cross_val_score(clf,test_desc,test_Labels)
    print(scores.mean())
    res.append(correct/len(test_desc))
    #print(correct/len(test_desc))

def svm(pth):
    train_desc=np.load(pth+'/training_features.npy')
    nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
    idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

    stdSlr = StandardScaler().fit(train_desc)
    train_desc = stdSlr.transform(train_desc)
    clf = LinearSVC()
    clf.fit(train_desc, np.array(train_labels))
    print(clf)
    joblib.dump((clf, img_classes, stdSlr), pth+"/svm-bof.pkl", compress=3)   
    test(pth,"svm-")

    
def bayes(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelBay=GaussianNB()
     modelBay.fit(train_desc, np.array(train_labels)) 
     print(modelBay)
     joblib.dump((modelBay, img_classes, stdSlr), pth+"/bayes-bof.pkl", compress=3)   
     test(pth,"bayes-")
     
def DT(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelDT=DecisionTreeClassifier(max_depth=5)
     modelDT.fit(train_desc,np.array(train_labels))
     print(modelDT)
     joblib.dump((modelDT, img_classes, stdSlr), pth+"/DT-bof.pkl", compress=3)   
     test(pth,"DT-")

def logistic(pth):
     
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modellog=LogisticRegression()
     modellog.fit(train_desc,np.array(train_labels))
     print(modellog)
     joblib.dump((modellog, img_classes, stdSlr), pth+"/Logistics-bof.pkl", compress=3)   
     test(pth,"Logistics-")


def knn(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelNeibor=KNeighborsClassifier()
     modelNeibor.fit(train_desc,np.array(train_labels))
     print(modelNeibor)
     joblib.dump((modelNeibor, img_classes, stdSlr), pth+"/knn-bof.pkl", compress=3) 
     test(pth,"knn-")

def RF(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelRF=RandomForestClassifier(n_estimators=10,
                                    max_depth=5,max_features=1,random_state=0)
     modelRF.fit(train_desc,np.array(train_labels))
     joblib.dump((modelRF, img_classes, stdSlr), pth+"/rf-bof.pkl", compress=3) 
     test(pth, "rf-")
     
def AB(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelAB=AdaBoostClassifier(n_estimators=100)
     modelAB.fit(train_desc,np.array(train_labels))

     joblib.dump((modelAB, img_classes, stdSlr), pth+"/ab-bof.pkl", compress=3) 
     test(pth, "ab-")
 
def GNB(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelGNB=GaussianNB()
     modelGNB.fit(train_desc,np.array(train_labels))
     joblib.dump((modelGNB, img_classes, stdSlr), pth+"/gnb-bof.pkl", compress=3) 
     test(pth, "gnb-")

def GN(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelGN=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,
                                         max_depth=1, random_state=0)
     modelGN.fit(train_desc,np.array(train_labels))
     joblib.dump((modelGN, img_classes, stdSlr), pth+"/gn-bof.pkl", compress=3) 
     test(pth, "gn-")
     
def LD(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelLD=LinearDiscriminantAnalysis()
     modelLD.fit(train_desc,np.array(train_labels))
     joblib.dump((modelLD, img_classes, stdSlr), pth+"/ld-bof.pkl", compress=3) 
     test(pth, "ld-")

def QD(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelQD=QuadraticDiscriminantAnalysis()
     modelQD.fit(train_desc,np.array(train_labels))
     joblib.dump((modelQD, img_classes, stdSlr), pth+"/qd-bof.pkl", compress=3) 
     test(pth, "qd-")
     
def ET(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelET=ExtraTreesClassifier(n_estimators=10,max_depth=None)
     modelET.fit(train_desc,np.array(train_labels))
     joblib.dump((modelET, img_classes, stdSlr), pth+"/et-bof.pkl", compress=3) 
     test(pth, "et-")
     
def svc(pth):
     train_desc=np.load(pth+'/training_features.npy')
     nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
     idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
     stdSlr = StandardScaler().fit(train_desc)
     train_desc = stdSlr.transform(train_desc)
     modelSVC=SVC(gamma=2,C=1)
     modelSVC.fit(train_desc,np.array(train_labels))
     joblib.dump((modelSVC, img_classes, stdSlr), pth+"/svc-bof.pkl", compress=3) 
     test(pth, "svc-")
    
if __name__ == '__main__':
    ss=["500dr","800dr","1000dr","2000dr","3000dr","4000dr","5000dr","6000dr","7000dr",
        "8000dr","9000dr","10000dr","15000dr","20000dr"]
    for x in range(14):
        svm(ss[x])
        logistic(ss[x])
        bayes(ss[x])
        #ET(ss[x])
    
        #QD(ss[x])
    
        #LD(ss[x])
    
        #GN(ss[x])
       # GNB(ss[x])
        #AB(ss[x])
       # RF(ss[x])
   # np.save('res6',np.array(res)) 
    print(res)
    pass