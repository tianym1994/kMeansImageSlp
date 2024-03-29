from sklearn import  preprocessing
import  PIL.Image as image
import  numpy as np
from  sklearn.cluster import  KMeans
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
