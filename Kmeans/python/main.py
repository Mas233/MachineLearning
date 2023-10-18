from KMeans import KMeans
from imgHandler import *

if __name__=='__main__':
    img_path='./crane.jpg'
    clusters=3
    iterations=50
    delta=0.0001

    img_array=imgToArray(img_path)
    labels=KMeans(img_array,clusters,iterations,delta)
    arrayToImg(img_array,labels)