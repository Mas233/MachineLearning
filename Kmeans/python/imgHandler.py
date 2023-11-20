import PIL.Image as pilImage
import numpy as np
_COLORS=[
    '#9933FF','#FF3300','#000066','#6699FF','#66CC00','#00CCAA'
]

def img_to_array(path):
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
    return img_width,img_height,np.array(imgData)


def array_to_imgs(data,mask,path,name):
    img_height=data.shape[0]
    img_width=data.shape[1]
    imgs=[pilImage.new('RGB',(img_width,img_height)) for _ in range(int(np.amax(mask))+1)]
    for y in range(img_height):
        for x in range(img_width):
            pixel=data[y,x]*255.0
            group_index=mask[y,x]
            imgs[group_index[0]].putpixel((x,y),(int(pixel[0]),int(pixel[1]),int(pixel[2])))

    for i, img in enumerate(imgs):
        img.show()
        img.save(path+f'/{name}_section{i+1}.png')


def array_to_single_img(data,mask,path,name):
    img_height = data.shape[0]
    img_width = data.shape[1]
    img=pilImage.new('RGB',(img_width,img_height))
    for y in range(img_height):
        for x in range(img_width):
            group_index=mask[y,x]
            img.putpixel((x,y), int(_COLORS[group_index[0]][1:],16))
    img.show()
    img.save(path+f'/{name}_output.png')
