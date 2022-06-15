   # Ce programme permet de reconnaitre en direct le visage d'une personne deja apprise
import cv2
fichModel="C:/Users/PC/Desktop/projectpy/modelsave77crop.h5"
fichModel2="C:/Users/PC/Desktop/projectpy/modelsaveMLPtry.h5"
path0="C:/Users/PC/Desktop/app+testbasekemla/"
fichModel3="C:/Users/PC/Desktop/projectpy/modelsaveLBP0.h5"

pathModel= [fichModel2,fichModel, fichModel3]
import keras as ker                 
from keras.models import Sequential # le modele qui sera utilisé

modelMlp = ker.models.load_model(pathModel[0])
modelCrop = ker.models.load_model(pathModel[1])
modelLbp = ker.models.load_model(pathModel[2])

personne= ['abdou', 'achref', 'adel', 'ahcen', 'akram', 'anas', 'anfel', 'aymen','basem', 'brahim', 'cerine', 'chaima', 
 'chouiab', 'doria', 'fatma','fethii', 'hana','hendouzi', 'hocine', 'imen', 'ines', 'ines boudebouz', 'islem',
 'jamila', 'lamine', 'liwae', 'marouaa', 'meriam', 'miraa', 'nadine', 'nour', 'omar', 'oussama', 'rachid', 'ramy', 
 'ramy kz', 'rania', 'raouf', 'salah', 'soraya', 'toufik', 'worod', 'yasmine', 'zina']
face_cascade= cv2.CascadeClassifier('C:/Users/PC/Desktop/projectpy/haarcascade_frontalface_default.xml')
import numpy as np
import os
from PIL import Image
dim=100 ;img_item = "my_image.png"
cap= cv2.VideoCapture(0)
ret,img = cap.read()

if (ret==0):
    print("Erreur la camera ne demarre pas..")
cv2.imshow('img', img)
print(" \n\n ------------------- RESULTAT ----------------------\n\n ")
dim2=70
while(True):
      ret,img = cap.read()
      if (ret==0):
          print("Erreur la camera ne demarre pas..")
          break
      cv2.imshow('img', img)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)   
      for (x,y,w,h) in faces:
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = img[y:y+h, x:x+w]
# lancer la prediction de nouvelle images,le resultat est  le nom de la personne  
         cv2.imwrite(img_item,roi_gray)
         img2 = Image.open(img_item).convert("L")  #
         res_img = img2.resize((dim,dim))
         X_new=res_img.copy()
         vec_img= np.reshape(res_img,[1,dim*dim])
         vec_img.astype('float32') ;  
         reslbp= modelLbp.predict(vec_img)


         res1 = modelMlp.predict(vec_img)
         
         # dimension=70
         res_img2 = img2.resize((dim2,dim2))
         X_new2=res_img2.copy()
         vec_img2= np.reshape(res_img2,[1,dim2*dim2])
         vec_img2.astype('float32') ;  
         rescrop= modelCrop.predict(vec_img2)

         
         print("result MLP  =\n ",res1)
         print("result MLP crop  =\n ",rescrop)
         print("result MLP LBP =\n ",reslbp)
         print(" ===============  ")

         pos1 = np.argmax(res1[0,:])
         pos2 = np.argmax(rescrop[0,:])
         pos3 = np.argmax(reslbp[0,:])
         pos=0 # valeur de depart
         print("les positions des visages ", pos1, "  " , pos2,"   ",pos3)
         if (pos1==pos2) and (pos1==pos3):
            print("Vainqueur = TOUS   TOUS OK  ")
            pos= np.argmax(res1[0,:])
         elif (pos1==pos2):      # lbp different
            pos= np.argmax(res1[0,:])
            print("MLP ET CROP OKAY ")
         elif (pos1==pos3):  # rescrop est different
            pos= np.argmax(res1[0,:])
         elif (pos2==pos3):  # resmlp est different
            pos= np.argmax(rescrop[0,:])
            print("CROP ET LBP OKAY")

         else:
            print("Desaccord entre les different mlp ")
            pos= np.argmax(res1[0,:])
         print("personne Visage  id : ",pos)
         idperson= personne[pos]
         print('liste a reconnaitre : \n',idperson)
         font = cv2.FONT_HERSHEY_SIMPLEX;  color =(255,255,255)
         stroke = 2                 
         print("name is: ",idperson)
         cv2.putText(img, str(idperson), (x,y), font, 1.5, color, stroke, cv2.LINE_AA)  # afficher le nom de la personne 
         img_item = "my_image.png"
         cv2.imwrite(img_item,roi_gray)
         color= (255, 0, 0);  stroke= 2
      cv2.imshow('img',img); k = (cv2.waitKey(30) & 0xff)
      if (k == 27):
            break      
cap.release()
cv2.destroyAllWindows()