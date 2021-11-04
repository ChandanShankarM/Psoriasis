from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import style
from PIL import Image, ImageEnhance 
import numpy as np
import numpy.core.multiarray
import warnings
import tkinter as tk
from tkinter import ttk
import cv2
from skimage import io
 

style.use('ggplot')
warnings.simplefilter('ignore')

from tkinter import filedialog


global_filename = ""

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
import joblib
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
 
#from dbn.tensorflow import SupervisedDBNClassification
fixed_size=tuple((256,256))
 
bins=8
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

# ===========================================================================================
def get_input(inp):
    print(inp)


# function to browse files
def browsefunc():
    global global_filename
    filename = filedialog.askopenfilename()
    global_filename = filename
    pathlabel.config(text=filename)


# given the path to image, returns its name
def get_img_name(path):
    path_split = path.split("/")
    return path_split[-1]


# save the genrated image
def save_file(image, img_path, scale):
    img_name = get_img_name(img_path)
    save_img_name = img_name[:-4] + "_SR_x{0}".format(scale) + img_name[-4:]

    save_folder =  filedialog.askdirectory()
    save_file = save_folder + "/" + save_img_name

    io.imsave(save_file, image)


# function to Show low resolution image on a new pop up window
def show_lr(path):
    #popup_lr = tk.Tk()
    #popup_lr.wm_title("Low Resolution Image")

    #label = ttk.Label(popup_lr, justify=tk.LEFT, text="""Original Low Resolution Image""", font=("Verdana", 14, "bold"))
    #label.pack(side="top", fill="x", pady=30, padx=30)

    img = io.imread(path)
    if img is None:
        print(path)
        print(type(path))
        print("IMG IS NONE")
        
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()    
    plt.imshow(img)
    #fig, ax = plt.subplots()
    #im = ax.imshow(img, origin='upper')
    #plt.grid("off")

    # canvas = FigureCanvasTkAgg(fig, popup_lr)
    # canvas.show()
    # canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # toolbar = NavigationToolbar2Tk(canvas, popup_lr)
    # toolbar.update()
    # canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # label = ttk.Label(popup_lr, justify=tk.CENTER, text="")
    # label.pack(side="top", pady=2, padx=30)

    # B1 = ttk.Button(popup_lr, text="SELECT FOLDER TO SAVE THIS IMAGE", command=lambda: save_file(img, path, scale=1))
    # B1.pack(side="top")

    # label = ttk.Label(popup_lr, justify=tk.CENTER, text="")
    # label.pack(side="top", pady=2, padx=30)

    # B2 = ttk.Button(popup_lr, text="CLOSE THIS WINDOW", command=popup_lr.destroy)
    # B2.pack(side="top")

    #popup_lr.mainloop()


# function to Show super resolved image on a new pop up window
def show_sr(path, scale=-2.0):
    from PIL import Image, ImageEnhance
    img = io.imread(path)
    #Im_x=ImageEnhance.Color(img)  
    #im1 = ImageEnhance.Sharpness(img)    
    #im4=Im_x.enhance(0.0) 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    plt.figure()    
    plt.imshow(image_yuv)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
   
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    plt.figure()    
    plt.imshow(image_rgb)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))
 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    
    plt.figure()    
    plt.imshow(img2)
    # Creating object of Sharpness class 
    
def predict(path):
     
    image= cv2.imread(path) #read image
    image=cv2.resize(image,fixed_size) #resize image
    fv_hu_moments=fd_hu_moments(image)
    fv_haralick=fd_haralick(image)
    fv_histogram=fd_histogram(image)
    global_feature=np.hstack([fv_histogram,fv_haralick,fv_hu_moments])
    global_feature=np.reshape(global_feature,(1, -1))
    x = pd.DataFrame(global_feature)
    X=x.iloc[:,0:15]
    model=joblib.load('dnn.pk1')
    preds = model.predict_classes(np.reshape(X,(1, -1)), verbose=0)
    print(preds) 
    if preds[0]==0:
        print('Diseased')
        disp_conf='THE IMAGE IS DISEASED'
    else:
        print('Not Diseased')
        disp_conf='THE IMAGE IS NON DISEASED'
    # showing resultant image 
    label = ttk.Label(root, justify=tk.CENTER, text=disp_conf, font=("Verdana", 11,"bold"))
    label.pack(side="top", pady=3, padx=30)
    
    pathlabel = ttk.Label(root, font=("Verdana", 11, "bold"))
    pathlabel.pack(side="top", pady=3, padx=30)

     
def show_exit():
    exit()
# ============================================================================================

root = tk.Tk()
tk.Tk.wm_title(root, "Psorasis Project ")
label = ttk.Label(root, text="Welcome to the Psoaris project GUI", font=("Verdana", 22, "bold"))
label.pack(side="top", pady=30, padx=50)

desc = '''This GUI allows you to load image and display super resolution of image and  then do analysis .
to identify diseased or not.'''
label = ttk.Label(root, justify=tk.CENTER, text=desc, font=("Verdana", 11))
label.pack(side="top", pady=30, padx=30)

label = ttk.Label(root, justify=tk.CENTER,
                  text="Click the browse button below to select the image file", font=("Verdana", 11))
label.pack(side="top", pady=5, padx=30)


button1 = ttk.Button(root, text="BROWSE", command=lambda: browsefunc())
button1.pack()



#pathlabel = ttk.Label(root, font=("Verdana", 11, "bold"))
#pathlabel.pack(side="top", pady=3, padx=30)

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=1, padx=30)

button1 = ttk.Button(root, text="SHOW ORIGINAL IMAGE", command=lambda: show_lr(global_filename))
button1.pack()
#%%
label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)

button2 = ttk.Button(root, text="Image Resolution", command=lambda: show_sr(global_filename, scale=2))
button2.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)

button3 = ttk.Button(root, text=" Analysis ", command=lambda: predict(global_filename))
button3.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)


button3 = ttk.Button(root, text="QUIT", command=lambda:show_exit(0))
button3.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=5, padx=30)

if __name__ == "__main__":
    root.mainloop()