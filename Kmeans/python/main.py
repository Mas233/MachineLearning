from KMeans import *
from imgHandler import *

def _KMeans(img_path='crane.jpg',clusters=4,iterations=30,delta=0.0001,single_output=True,output_path='.',output_file_name='single'):
    width, height, img_array = img_to_array(img_path)
    labels = KMeans(img_array, clusters, iterations, delta)

    img_array = img_array.reshape((height, width, -1))
    labels = np.array(labels).reshape((height, width, -1))
    if single_output:
        array_to_single_img(img_array, labels, output_path,output_file_name)
    else:
        array_to_imgs(img_array,labels,output_path,output_file_name)

if __name__=='__main__':
    for i in range(6,7):
        _KMeans(img_path='ramen.jpg',clusters=i,iterations=15+(i-3)*3,output_file_name=f'ramen_c{i}')
        _KMeans(img_path='crane.jpg', clusters=i, iterations=15 + (i - 3) * 3, output_file_name=f'crane_c{i}')
