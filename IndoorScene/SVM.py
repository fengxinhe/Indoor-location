from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib.numpy_pickle_utils import xrange
import numpy as np


train_desc=np.load('training_features.npy')
image_paths=np.load('imgpath.npy')
train_labels=np.load('train_labels.npy')
codebook=np.load('data/codebook.npy')
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

nbr_occurences = np.sum( (train_desc > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(train_desc)
train_desc = stdSlr.transform(train_desc)

# Train the Linear SVM
clf = LinearSVC()
clf.fit(train_desc, np.array(train_labels))

# Save the SVM
joblib.dump(clf,'svm_module.pkl')

joblib.dump((clf, img_classes, stdSlr, 800, codebook), "bof.pkl", compress=3)   