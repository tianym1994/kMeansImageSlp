from sklearn import  preprocessing
import  PIL.Image as image

import  numpy as np
from  sklearn.cluster import  KMeans
import matplotlib.image as mping
from skimage import color
#加载图像，并对数据进行规范化
def load_data(filePath):
#读文件
    f=open(filePath,'rb')
    data=[]
    img=image.open(f)
    width,height=img.size
    for x in range(width):
        for y in range(height):
            #得到点（x,y）的三个通道值
            c1,c2,c3=img.getpixel((x,y))
            data.append( [c1,c2,c3])
    f.close()
    #采用min-max规范化
    mm=preprocessing.MinMaxScaler()
    data=mm.fit_transform(data)
    return np.mat(data),width,height



#加载图像，得到规范化的结果IMG，以及图像尺寸
img,width,height=load_data('./weixin.jpg')

#用kmeans对图像进行2聚类
kmeans=KMeans(n_clusters=2)
kmeans.fit(img)
label=kmeans.predict(img)
#将图像聚类结果，转化为图像尺寸的矩阵
label=label.reshape([width,height])
#创建个新的图像pic_mark,用来保存图像聚类的结果，并设置不同的灰度
pic_mark=image.new('L',(width,height))
for x in range(width):
    for y in range(height):
        #根据类别设置图像灰度，类别0灰度值为255，类别1灰度值为127
        pic_mark.putpixel((x,y),int(256/(label[x][y]+1))-1)
pic_mark.save('weixin_mark.jpg','JPEG')

#分割成16个部分


def load_data_c(filePath):
#读文件
    f=open(filePath,'rb')
    data=[]
    img=image.open(f)
    width,height=img.size
    for x in range(width):
        for y in range(height):
            #得到点（x,y）的三个通道值
            c1,c2,c3=img.getpixel((x,y))
            data.append([(c1 + 1) / 256.0, (c2 + 1) / 256, (c3 + 1) / 256])
    f.close()
    #采用min-max规范化
    return np.mat(data),width,height



# 加载图像，并对数据进行规范化
def load_data(filePath):
    # 读文件
    f = open(filePath,'rb')
    data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到点(x,y)的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            data.append([c1, c2, c3])
    f.close()
    # 采用Min-Max规范化
    mm = preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data), width, height

# 加载图像，得到规范化的结果img，以及图像尺寸
img, width, height = load_data('./weixin.jpg')




from  skimage import  color


img,width,height=load_data_c('./weixin.jpg')
#使用kemans对图像进行16聚类
kemans=KMeans(n_clusters=16)
kemans.fit(img)
label=kmeans.fit_predict(img)
#将图像聚类结果，转化成图像尺寸的矩阵
label=label.reshape([width,height])
#将聚类标识转化为不同颜色的矩阵
# label_color=(color.label2rgb(label)*255).astype(np.uint8)
# label_color=label_color.transpose(1,0,2)
# images=image.fromarray((label_color))
# images.save('weixin_mark_color.jpg')
#创建个新的图像img，用来保存图像聚类压缩后的结果
img=image.new('RGB',(width,height))
for x in range(width):
    for y in range(height):
        c1=kemans.cluster_centers_[label[x,y],0]
        c2 = kemans.cluster_centers_[label[x, y], 1]
        c3 = kemans.cluster_centers_[label[x, y], 2]
        img.putpixel((x,y),(int(c1*256)-1, int(c2*256)-1, int(c3*256)-1))
img.save('weixin_new2.jpg')

