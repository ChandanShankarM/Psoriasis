#%% TESTING PART

import h5py
import numpy as np
import cv2
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold#, StratifiedKFold
#From sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
import mahotas

from sklearn.metrics import confusion_matrix
import seaborn as sn # heatmap for confusion matrics
import time
import os
import os.path
from os import path
#%% Set Path
#PATH = 'C:\Users\Sumol Suresh\Desktop\plantdisease\random_model.pkl'
#%% SAME PROCESS AS WE DID FOR TRAINING
fixed_size = tuple((500, 500))

# no.of.trees for Random Forests
num_trees = 100
#total number of bins for histogram
bins = 8
# train_test_split size
test_size = 0.10
# is the seed we keep to reproduce
# same results everytime we run this script
seed = 9 # to make sure data is choosen randomly but every time
# we run the process same data is choosen

 
#%%    
def fd_hu_moments(image):
    image =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature=cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
# feature-descriptor-1:haralick
    
def fd_haralick(image):
    #convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    #return the result
    return haralick
#feature-descriptor-3: Color Histogram
    
def fd_histogram(image, mask=None):
        #convert the image to HSV color-space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #compute the color histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        #normalize the histogram
        cv2.normalize(hist, hist)
        #return the histogram
        return hist.flatten()
#%% MACHINE LEARNING MODELS
models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
#%%
test_size = 0.10
seed = 9
results = []
names = []
scoring='accuracy'

#import the feature vector and trained labels
h5f_data = h5py.File('Output/data.h5','r')
h5f_label = h5py.File('Output/labels.h5','r')

global_features_string = h5f_data['dataset_1']
global_labels_string = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

#verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

#split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features), np.array(global_labels), test_size=test_size, random_state = seed)
print("[STATUS] splitted train and test data...")
print("[STATUS] splitted train and test data...")
print("Train data : {}".format(trainDataGlobal.shape))
print("Test data  : {}".format(testDataGlobal.shape))
print("Train labels :{}".format(trainLabelsGlobal.shape))
print("Test labels :{}".format(testLabelsGlobal.shape))


#filter all the warnings
import warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results=cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
print('BREAKING EACH MACHINE LEARNING MODEL')
time.sleep(5)

#%% LETS BREAK DOWN EACH METHOD

# step 1: breaking Logistic regression

classifier_LR=LogisticRegression(random_state=9)
classifier_LR.fit(trainDataGlobal, trainLabelsGlobal)
param = classifier_LR.coef_
Lr_predict=classifier_LR.predict(testDataGlobal)
cm=confusion_matrix(testLabelsGlobal,Lr_predict)
sn.heatmap(cm, annot=True)
pyplot.title('Logistic Regression Confusion matrix')
pyplot.show()
pyplot.boxplot(Lr_predict)
ax.set_xticklabels(names)
pyplot.title('Box plot for Logistic Regression')
pyplot.show()
print('-----------')
total_case=sum(np.diagonal(cm))/sum((cm))*100
#print('Efficency of Logistic Regression in percentage :', total_case)
#print('-----------')
time.sleep(5)
#step 2: Breaking KNN
print('-----------')
Classifier_knn=KNeighborsClassifier(n_neighbors=5,)
Classifier_knn.fit(trainDataGlobal,trainLabelsGlobal)
Knn_predict= Classifier_knn.predict(testDataGlobal)
cm1=confusion_matrix(testLabelsGlobal, Knn_predict)
sn.heatmap(cm1, annot=True)
pyplot.title('KNN Confusion Matrix')
pyplot.show()
pyplot.boxplot(Knn_predict)
ax.set_xticklabels(names)
pyplot.title('Box plot for KNN')
pyplot.show()
print('-----------')
total_case1=sum(np.diagonal(cm1))/sum((cm1))*100
#print('Efficiency of KNN in Percentage :',total_case1)
#print('-----------')
time.sleep(5)
#step 3: Breaking Decision tree
Classifier_tree=DecisionTreeClassifier(random_state=9)
Classifier_tree.fit(trainDataGlobal, trainLabelsGlobal)
tree_predict=Classifier_tree.predict(testDataGlobal)
cm2=confusion_matrix(testLabelsGlobal, tree_predict) 
sn.heatmap(cm2, annot=True)
pyplot.title('Decision tree Confusion matrix')
pyplot.show()
total_case2=sum(np.diagonal(cm2))/sum((cm2))*100
pyplot.boxplot(tree_predict)
ax.set_xticklabels(names)
pyplot.title('Box plot for Decision Tree')
pyplot.show()
#print('-----------')
#print('Efficiency of Decision Tree in Percentage:', total_case2)
#print('-----------')
time.sleep(5)
#step 4: Breaking Random Forest
Classifier_random = RandomForestClassifier(n_estimators=100, random_state=9)
Classifier_random.fit(trainDataGlobal, trainLabelsGlobal)
forest_predict=Classifier_random.predict(testDataGlobal)
cm3=confusion_matrix(testLabelsGlobal, forest_predict)
sn.heatmap(cm3, annot=True)
pyplot.title('Random forest Confusion matrix')
pyplot.show()
total_case3=sum(np.diagonal(cm3))/sum((cm3))*100
pyplot.boxplot(forest_predict)
ax.set_xticklabels(names)
pyplot.title('Box plot for Random forest')
pyplot.show()
print('-----------')
#print('Efficiency of Random forest in Percentage :',total_case3)
#print('-----------')
#time.sleep(5)
#%% Save model
joblib.dump(Classifier_random,'random_model.pkl')


























