from KMeans import *
from imgHandler import *


if __name__=='__main__':
    img_path='./crane.jpg'
    clusters=3
    iterations=15
    delta=0.0001

    width,height,img_array=imgToArray(img_path)
    labels=KMeans(img_array,clusters,iterations,delta)

    img_array=img_array.reshape((height,width,-1))
    labels=np.array(labels).reshape((height,width,-1))
    arrayToImg(img_array,labels,'.')