import cv2
import numpy as np
from matplotlib import pyplot as plt
  
def get_pixel(img, center, x, y):
    
    new_value = 0
    try:
     #si la valeur du pixel central et superieure ou egal a son voisin on met 1 sinon on met 0
        if img[x,y] >= center:
            new_value = 1
            
    except:
        pass
    
    return new_value
# La fonction LBP
def lbp_calculated_pixel(img, x, y):

    center = img[x,y]

    val_bin = []
    
    val_bin.append(get_pixel(img, center, x-1, y-1)) # voisin a gauche en haut

    val_bin.append(get_pixel(img, center, x-1, y)) # voisin a gauche

    val_bin.append(get_pixel(img, center, x-1, y + 1)) # voisin a gauche en bas

    val_bin.append(get_pixel(img, center, x, y + 1)) # voisin en bas 

    val_bin.append(get_pixel(img, center, x + 1, y + 1)) #voisin a droite en bas

    val_bin.append(get_pixel(img, center, x + 1, y)) # voisin a droite 

    val_bin.append(get_pixel(img, center, x + 1, y-1)) # voisin a droite en haut

    val_bin.append(get_pixel(img, center, x, y-1)) # voisin en haut
    
    # transformation de valeur binaire en decimal en utilisant la puissance
    degrepoly = [1, 2, 4, 8, 16, 32, 64, 128]
    vallbp = 0
    
    for i in range(len(val_bin)):
        vallbp += val_bin[i] * degrepoly[i]
        
    return vallbp