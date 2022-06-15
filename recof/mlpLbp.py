from turtle import width
import keras as ker
import cv2                  
import matplotlib.pyplot as plt     
import random    
import os  #manipulation des dossiers et fichiers
import numpy as np # la manipulation des tableaux
from PIL import Image                   
from keras.models import Sequential # le modele qui sera utilisé
from keras.layers.core import Dense
from keras.utils import np_utils     # NumPy related tools

from keras.layers.core import Dense, Dropout, Activation # type de couche qui serons utiliser dans ce model
import pandas as pda

from getImageLbp import get_pixel, lbp_calculated_pixel

def construireLBP(imgOrig,dim):
    height=dim; width=dim
    img_lbp = np.zeros((height, width),
                np.uint8)
    imgOrig = np.array(imgOrig, "uint8")
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(imgOrig, i, j)
    return img_lbp        

def getfiles():
    Data=[]; labels=[]
    Base_Dir="C:/Users/PC/Desktop/projectpy/apprentissage"
    Base_Dir=input("Donner le repertoire des images: \n")
    print("base directory: ", Base_Dir)
    idx=0; pers=[] ; dim=100 
    for root,dirs,files in os.walk(Base_Dir):
        print("root: ",root)
        #print("fichiers = ", files)
        print("directory:" ,dirs)
        pers.append(dirs)
        for file in files:
           #print("liste de fichiers: ",files)
           print("file = ",file)
           if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"):
               print("file image N°=  ",file)
               paths=os.path.join(root,file)
               print("nom fichir complet :   ",paths)
               print("chemin label: ",os.path.basename(root))
               imgf= Image.open(paths)
               print(paths)
               pil_image = Image.open(paths).convert("L")  
               res_img = pil_image.resize((dim,dim))
               res_img2 = construireLBP(res_img,dim)

               im=res_img.copy()
               vect_img= np.reshape(res_img,(dim*dim))
               print("taille image  : ", vect_img.shape)
               image_array = np.array(vect_img, "uint8")
               label = idx
               labels.append(label)
               Data.append(image_array)
        idx=idx+1
    data=np.array(Data)
    Labels=np.array(labels)
    print("la taille de data ",data.shape)           
    return data,Labels,pers

dim=100                        
print("chemin des images d'apprentissage : ")
path="C:/Users/PC/Desktop/app+testbasekemla"                                                                                                                                                  
datatrain,labeltrain,perso=getfiles()                                                           
print("TRAIN labels : \n ",labeltrain)
print("taille des labels : ");print(labeltrain.shape)
print("mes images en memoires :",datatrain)
print("chemin des images de Tests : ")                  
datatest,labeltest,perso=getfiles()                                                           
print("Test   labels: \n ",labeltest)
print("taille des labels : "); print(labeltest.shape)
print("datatrain shape", datatrain.shape);print("labeltrain shape", labeltrain.shape)
print("datatest shape", datatest.shape);print("labeltest shape", labeltest.shape)

datatrain = datatrain.astype('float32')  
datatest = datatest.astype('float32')
datatrain /= 255.0    
datatest /= 255.0;  print("Training matrix shape", datatrain.shape)
print("Testing matrix shape", datatest.shape)
drop=0.1
nb_classes = 44
labeltrain=labeltrain-1
labeltest=labeltest-1
print("labeltrain ",labeltrain[0:5])

labeltrain = np_utils.to_categorical(labeltrain, nb_classes)
labeltest = np_utils.to_categorical(labeltest, nb_classes)
print("labeltrain ",labeltrain[0:5])
model = Sequential()
nbNeuron1=512 
nbNeuron2=512 
nbNeuron3=128 
nbNeuron4=128 
model.add(Dense(nbNeuron1, input_shape=(dim*dim,))) 
model.add(Activation('relu'))
model.add(Dropout(drop))
model.add(Dense(nbNeuron2)); model.add(Activation('relu'));  model.add(Dropout(drop))
model.add(Dense(nbNeuron3)); model.add(Activation('relu')); model.add(Dropout(drop))
model.add(Dense(nbNeuron4));model.add(Activation('relu')); model.add(Dropout(drop))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()
print('  Let"s use the Adam optimizer for learning')
from keras import losses 
from keras import optimizers 
from keras import metrics 
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(datatrain, labeltrain, batch_size=10, epochs=25, verbose=1, validation_data=(datatest,labeltest))
score = model.evaluate(datatest, labeltest)
print('Test score:', score[0]); print('Test accuracy:', score[1])
model.save('C:/Users/PC/Desktop/projectpy/modelsaveLBP0.h5')

print(score)



