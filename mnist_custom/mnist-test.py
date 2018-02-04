import sys
sys.path.insert(0,'/home/dana/caffe/python')
import caffe
from caffe import layers as L, params as P
import cv2
import numpy as np


image=cv2.imread('./pics/wds1.png')
image1=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)

image=cv2.imread('./pics/wds2.png')
image2=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)

image=cv2.imread('./pics/wds3.png')
image3=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)


image1=image1.transpose(2,0,1)/255.0
image2=image2.transpose(2,0,1)
image3=image3.transpose(2,0,1)/255.0


model_def ='./deploy.prototxt'
model_weights='./pretrained.caffemodel'
net=caffe.Net(model_def,model_weights,caffe.TEST)

net.blobs['data'].data[0,:,:,:]=image2
output=net.forward()

print(output)
output_prob=net.blobs['prob'].data[0]
print(output_prob)
cv2.imshow('image',image2.transpose(1,2,0))
cv2.waitKey(0)
