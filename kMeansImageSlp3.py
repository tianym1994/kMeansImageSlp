import  PIL.Image as image
import  numpy as np
from  sklearn.cluster import  KMeans
# 使用K-means对图像进行聚类，并显示聚类压缩后的图像
#分割成16个部分
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
            data.append([(c1 + 1) / 256.0, (c2 + 1) / 256, (c3 + 1) / 256])
    f.close()
    #采用min-max规范化
    return np.mat(data),width,height

# 加载图像，得到规范化的结果img，以及图像尺寸
img,width,height=load_data('./weixin.jpg')
#使用kemans对图像进行16聚类
kemans=KMeans(n_clusters=16)
kemans.fit(img)
label=kemans.fit_predict(img)
#将图像聚类结果，转化成图像尺寸的矩阵
label=label.reshape([width,height])
img=image.new('RGB',(width,height))
for x in range(width):
    for y in range(height):
        c1=kemans.cluster_centers_[label[x,y],0]
        c2 = kemans.cluster_centers_[label[x, y], 1]
        c3 = kemans.cluster_centers_[label[x, y], 2]
        img.putpixel((x,y),(int(c1*256)-1, int(c2*256)-1, int(c3*256)-1))
img.save('weixin_new2.jpg')

