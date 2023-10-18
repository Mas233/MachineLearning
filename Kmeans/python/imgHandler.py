import PIL.Image as pilImage
import numpy as np
def imgToArray(path):
    file=open(path,"rb")
    img=pilImage.open(file)
    img_width,img_height=img.size
    imgData=[]
    for y in range(img_height):
        for x in range(img_width):
            r,g,b=img.getpixel((x,y))
            r/=255.0
            g/=255.0
            b/=255.0
            imgData.append([r,g,b])
    file.close()
    return np.array(imgData)

def arrayToImg(data,mask,path):
    img_height,img_width,_=data.shape
    imgs=[pilImage.new('RGB',(img_width,img_height)) for _ in range(np.argmax(mask,axis=0)+1)]
    for y in range(img_height):
        for x in range(img_width):
            pixel=data[y,x]*255.0
            group_index=mask[y,x]
            imgs[group_index].putpixel((x,y),(int(pixel[0]),int(pixel[1]),int(pixel[2])))

    for i, img in enumerate(imgs):
        img.show()
        # img.save(path+f'group_{i}.png')