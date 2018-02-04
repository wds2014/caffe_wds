import sys
sys.path.insert(0,'/home/dana/caffe/python')
import caffe
from caffe import layers as L, params as P
import cv2
import numpy as np
import random

def get_label(label,total_num):
    y=np.zeros((total_num))
    for i in range(total_num):
        if i == int(label):
            y[i]=1
    return y

def read_images(img_txt):
    X=[]
    Y=[]
    with open(img_txt) as f:
        for i, lines in enumerate(f.readlines()):
            img_path=lines.split()
            image=cv2.imread(img_path[0]).transpose(2,0,1)
            #label=get_label(img_path[1],10)
            label=int(img_path[1])
            X.append(image)
            Y.append(label)
    X=np.array(X,dtype=np.float32)/255
    Y=np.array(Y,dtype=np.float32)
    return X,Y

#def get_train_batch(batch_size,train_data,train_labels):
def get_sample(size):
    a=np.zeros((size))
    for i in range(size):
        a[i]=i
    a=np.array(a,dtype=np.int64)
    return a

if __name__ == '__main__':
    size_num=200000
    train_data=np.load('./mnist/train_data.npy')
    train_labels=np.load('./mnist/train_labels.npy')
    test_data=np.load('./mnist/test_data.npy')
    test_labels=np.load('./mnist/test_labels.npy')
    train_sample=get_sample(60000)
    test_sample=get_sample(10000)
    print("done")
    solver=caffe.SGDSolver('./solver.prototxt')
    net=solver.net
    net_test=solver.test_nets[0]
    for i in range(size_num):
        sample_train=np.array(random.sample(train_sample,64))
        net.blobs['data'].data[...]=train_data[sample_train,:,:,:]
        net.blobs['label'].data[...]=train_labels[sample_train]
        solver.step(1)
        if i %500 ==0:
            sample_test=np.array(random.sample(test_sample,100))
            print(sample_train)
            print(sample_test)
            net_test.blobs['data'].data[...]=test_data[sample_test,:,:,:]
            net_test.blobs['label'].data[...]=test_labels[sample_test]
            net_test.forward()
            print("custom acc:",net_test.blobs['acc'].data)























