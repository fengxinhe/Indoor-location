
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import shutil


#cv2.waitKey(0)
#cv2.destroyAllWindows()

def getSift (indir,outdir):  
    os.chdir(indir)

    for file in os.listdir(os.curdir):
        filename,ex=os.path.splitext(file)
        vectorFile=outdir+filename
    
        image=cv2.imread(file)
        gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc=sift.detectAndCompute(gray,None)
        np.save(vectorFile,np.array(desc)) 
    #f.close()
    
if __name__ == '__main__':
    inputDir=[#'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\mr',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\ra',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\so',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\to',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\co',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\lmr',
             'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\lo',
             'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\pa',
             'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\el']
    outputDir=[#'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\mr\\',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\ra\\',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\so\\',
             #'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\to\\',
            # 'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\co\\',
            # 'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\lmr\\',
             'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\lo\\',
             'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\pa\\',
             'C:\\Users\\qq1\\Desktop\\IndoorScene\\data\\validation\\resize\\trainSet\\el\\']
    for x in range(len(inputDir)):
        getSift(inputDir[x], outputDir[x])
