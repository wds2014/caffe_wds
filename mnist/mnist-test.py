import sys
sys.path.insert(0,'/home/dana/caffe/python')
import caffe
from caffe import layers as L, params as P
import cv2
import numpy as np
# solver = caffe.SGDSolver('./solver.prototxt')
# # for k,v in solver.net.blobs.items():
# #     print(k,v.data)
# # for k,v in solver.net.params.items():
# #     print(k,v[0].data,v[1].data)
# solver.net.forward()
# solver.test_nets[0].forward()
# solver.step(2)
image=cv2.imread('./pics/wds1.png')
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image1=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)

image=cv2.imread('./pics/wds2.png')
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image2=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)

image=cv2.imread('./pics/wds3.png')
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image3=cv2.resize(image,(28,28),interpolation=cv2.INTER_CUBIC)

#images=np.array([image,image1,image2]).reshape(3,1,28,28)
#print(images.shape)


model_def ='./lenet_deploy.prototxt'
model_weights='./pretrained.caffemodel'
net=caffe.Net(model_def,model_weights,caffe.TEST)

net.blobs['data'].data[0,0,:,:]=image3
#net.blobs['data'].data[1,0,:,:]=image2
#net.blobs['data'].data[2,0,:,:]=image3
#net.blobs['data'].data[0,0]=np.zeros((28,28))
output=net.forward()

print(output)
output_prob=net.blobs['prob'].data[0]
print(output_prob)
cv2.imshow('image',image3)
cv2.waitKey(0)
