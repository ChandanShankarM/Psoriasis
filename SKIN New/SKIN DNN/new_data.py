import os.path
from os import path
import cv2
from matplotlib import pyplot
import numpy as np
import mahotas
from sklearn.externals import joblib
import json
model=joblib.load('C:/Users/abhijith.abhi/Desktop/SKIN DNN/random_model.pkl')
fixed_size=tuple((500,500))
bins=8
with open("data.txt") as json_file:
	data=json.load(json_file)
execute=1
while execute==1:
    PATH='C:/Users/abhijith.abhi/Desktop/SKIN DNN/Test/ISIC_0000002.jpg'
    y=path.exists(PATH)
    if y:
        count=1
        print("Data Found")
    else:
        count=0
        print("No data found")
    if count==1:
        image=cv2.imread(PATH)
        image=cv2.resize(image,fixed_size)
        pyplot.imshow(image)
        pyplot.show()
        def fd_hu_moments(image):
            cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            feature=cv2.HuMoments(cv2.moments(image)).flatten()
            return feature
        def fd_haralick(image):
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            haralick=mahotas.features.haralick(gray).mean(axis=0)
            return haralick
        def fd_histogram(image,mask=None):
            image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            hist=cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        fv_hu_moments=fd_hu_moments(image)
        fv_haralick=fd_haralick(image)
        fv_histogram=fd_histogram(image)
        global_feature=np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        count=0
        #predict label of test image
        prediction=model.predict(global_feature.reshape(1,-1))[0]
        print("Prediction Label is: {}".format(data[prediction+1]))
        #show predicted label on image
        cv2.putText(image, data[prediction+1], (20,30), cv2.FONT_HERSHEY_SIMPLE, 1.0, (0, 255, 255), 3)
        pyplot.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pyplot.show()
        global_feature=[]
        fv_hu_moments=[]
        fv_haralick=[]
        fv_histogram=[]
         
        print("Prediction Label is: {}".format(data[prediction+1]))
        execute=0 


