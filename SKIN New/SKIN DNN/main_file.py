#%% LIBRARIES
# organize imports
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
# for machine learning we cannot have any data
# or text so we conver it to numbers hance used Label Encoder
from sklearn.preprocessing import MinMaxScaler
#gets feature within a range
import numpy as np
#numerical
import mahotas
# Computer vision toolbox contains algorithm
#for feature extraction in general and its works faster
import cv2
#open CV library
import os
#Operating system
import h5py
#let you store huge amounts of numerical data,
#and easily manipulate that data from NumPy
from pathlib import Path
#handle filesystem paths
import json
#to save in json format
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib 
from dbn.tensorflow import SupervisedDBNClassification
 
#%%TRAINING PATH
#path to training data where data is saved
train_path='Database'
fixed_size=tuple((256,256))
 
#total number of bins for histogram
bins=8
#train_test_split size
test_size=0.10
#is the seed we keep to reproduce
#same results everytime we run this scrpit
seed=9
#%% FOR EXTRACTING IMAGES
def folders_in_path(path): #takes path as input
    if not Path.is_dir(path): #checks if path exsist
        raise ValueError("argument is not directory") #produses error
        #if not in directory
    yield from filter(Path.is_dir,path.iterdir())
def folders_in_depth(path,depth):
    if 0>depth:
        raise ValueError("depth smaller 0")
    if 0==depth:
        yield from folders_in_path(path)
    else:
        for folder in folders_in_path(path):
            yield from folders_in_depth(folder,depth-1)
def files_in_path(path):
    if not Path.is_dir(path):
        raise ValueError("argument is not a directory")
    yield from filter(Path.is_file,path.iterdir())
def sum_file_size(filepaths):
    return sum([filep.stat().st_size for filep in filepaths])
#%% FEATURE EXTRACTION
#Hu moments
# feature descriptor-1: Hu Moments 7 feature
def fd_hu_moments(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    feature=cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
#feature descriptor-1:
def fd_haralick(image): #glcm feature
#convert the image to grayscale
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# compute the haralick texture feature vector
    haralick=mahotas.features.haralick(gray).mean(axis=0)
#return the result
    return haralick

#feature-descriptor-3: Color Histogram

def fd_histogram(image,mask=None):
    #convert the image to HSV color-space
    image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    #compute the color histogram
    hist=cv2.calcHist([image],[0,1,2],None,[bins,bins,bins],
                      [0,256,0,256,0,256])
    #normalize the histogram
    cv2.normalize(hist,hist)
    #return the histogram
    return hist.flatten()

#%%MAIN PROGRAM STARTS HERE

if __name__=='__main__':

    train_labels=os.listdir(train_path) #take training path labels
    train_labels.sort() #sort the labels
    print(train_labels) #primt the lables
    global_features=[] #initialize variable to combine all features
    labels=[] #create label variables so as to decode text to number
    total=0 #initialize
    tot_file=[] #initialize
    count=1 #start count to check number of images
    i=0
    j=0
    k=0
    print(Path.cwd()) #gives the current path
    for folder in folders_in_depth(Path.cwd(),1):
        #first loop will pick the first foldend then next folder
        files=list(files_in_path(folder)) #list all files in folder
        file=len(files) #length of files
        tot_file.append(file) #because we are running for all folder
        # we are appending all files in tot_file at the end we
        #shall get the list of number of files in the folder
        #we are doing this because every folder has different number of files
        #at the end when we are trainig all class of disease have to be
        #trained equally, hence find the least number of images in the folder
        #and then train accordingly
        total_size=sum_file_size(files)
        #total size of files
        count=count+1 #check total number of files executed
        print(f'{folder}:filecount:{len(files)},total size:{total_size}')
        # print
    tot_file.sort() #sort files based on ascending order
    num=tot_file[1] #Index 0 is junkhence extract index 1
    images_per_class=40 #consider number of images per class
    #%%START WITH TRAINING
    #for tr_name in range(0,2):
    count=0    
    while count <=1:
        tr_name=count
        print(tr_name)
        dir=train_path+'/'+train_labels[tr_name]
        current_label=train_labels[tr_name]
        print("[STATUS] processed folder: {}".format(current_label))
        k=1
        #print(dir)
        file_sub_folder=os.listdir(dir) 
        for x in range(0,images_per_class):
            file=os.getcwd()+'//'+dir +'//'+ file_sub_folder[x]
            image= cv2.imread(file) #read image
            image=cv2.resize(image,fixed_size) #resize image
            fv_hu_moments=fd_hu_moments(image)
            fv_haralick=fd_haralick(image)
            fv_histogram=fd_histogram(image)
            global_feature=np.hstack([fv_histogram,fv_haralick,fv_hu_moments])
            global_features.append(global_feature)
            labels.append(current_label)
            i+=1
            k+=1    
        print('inner loop done')
        count=count+1
    print("[STATUS] complete global feature extraction...")
    #print('outer loop ')
    #get the overall feature vector size
    print("[STATUS]featurevector size{}".format(np.array(global_features).shape))
    #get the overall training label size
    print("[STATUS] training labels{}".format(np.array(labels).shape))
#%%LABEL ENCODING

#encode the target labels
    targetNames=np.unique(labels)
    le=LabelEncoder()
    target=le.fit_transform(labels) #this steps sets label form 0 to..n based
#on number of labels for us there are 14 label starting from 0
    print("[STATUS] target labels:{}".format(target))
    print("[STATUS] target labels shape:{}".format(target.shape))
    print("[STATUS] training labels encoded...")
#%%FEATURE NORMALIZATION
#normalize the feature vector in the range(0 -1)
 
    scaler=MinMaxScaler(feature_range=(0,1))
    rescaled_features=scaler.fit_transform(global_features)
    print("[STATUS] feature vector normalized...")
    
#%% SAVE THE DATA FOR TRAINING

    h5f_data=h5py.File('Output/data.h5', 'w')
    h5f_data.create_dataset('dataset_1',data=np.array(rescaled_features))
    h5f_data.close()
    h5f_label=h5py.File('Output/labels.h5', 'w')
    h5f_label.create_dataset('dataset_1',data=np.array(target))
    h5f_label.close()
 

#%% SAVE LABEL IN JSON FORMAT
#do this just to display output when we test
    with open('data.txt','w') as outfile:
         json.dump(train_labels, outfile)
#%% CLOSE
    print("[STATUS] end of feature extraction ...")
    #%% Feature 
    model = LinearRegression()
    x = pd.DataFrame(global_features)
    y = pd.DataFrame(target)
    #Initializing RFE model
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LassoCV
    rfe = RFE(model, 7)
    x1=x.iloc[:,0:10] 
#Transforming data using RFE
    X_rfe = rfe.fit_transform(x1,y)  
#Fitting the data to model
    model.fit(X_rfe,y)
    print(rfe.support_)
    print(rfe.ranking_)
    reg = LassoCV()
    reg.fit(x1, y)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(x1,y))
    coef = pd.Series(reg.coef_, index = x1.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()
    imp_coef.plot(kind = "barh")
    import matplotlib.pyplot as plt 
    plt.title("Feature importance using Lasso Model")
    plt.show()
#%% TRaining  PART
    from keras.models import Sequential
    from keras.layers import Dense
    import pandas as pd
    
    x = pd.DataFrame(global_features)
    y = pd.DataFrame(target)
    complete_feat=pd.concat([x,y],axis=1)
    X=x.iloc[:,0:15] 
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)
    # define the keras model
    model = Sequential()
    model.add(Dense(64, input_dim=15, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(24, activation='tanh'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='tanh'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history=model.fit(X_train, Y_train, epochs=100, batch_size=20)
    _, accuracy = model.evaluate(X_test,Y_test)
    print('Accuracy: %.2f' % (accuracy*100))
    preds = model.predict_classes(X_test, verbose=0)
    cnf=confusion_matrix(Y_test, preds)        
    print(cnf)
    print('-----------------')
    joblib.dump(model,'dnn.pk1')
#%%
     
     # import the necessary packages
    X_train1=X_train.reset_index(drop=True) 
    Y_train1=Y_train.reset_index(drop=True) 
    model = SupervisedDBNClassification(hidden_layers_structure=[200,200],
    learning_rate_rbm=0.01,
    learning_rate=0.1,
    n_epochs_rbm=20,
    n_iter_backprop=20,
    batch_size=30,
    activation_function='relu')
    scale = np.max(X_train)
    X_train /= scale
    X_test /= scale

    mean = np.std(X_train)
    X_train -= mean
    X_test -= mean
    nb_classes= Y_train.shape[1]
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    print("Training...")
    model.fit(X_train, Y_train, nb_epoch=20, batch_size=30)
    
    print("Generating test predictions...")
    preds = model.predict_classes(X_test, verbose=0)
  
    cnf=confusion_matrix(Y_test, preds)        
    print(cnf)
    
    