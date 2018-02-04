import sys
sys.path.insert(0,'/home/dana/caffe/python')
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import glob
import numpy as np
import sys
import os.path

base_root='/home/dana/caffe_wds/mnist_pic'
train_data=os.path.join(base_root,'mnist/train/train.txt')
test_data=os.path.join(base_root,'mnist/test/test.txt')
train_net=os.path.join(base_root,'train.prototxt')
test_net=os.path.join(base_root,'test.prototxt')
deploy_net=os.path.join(base_root,'deploy.prototxt')
solver_path=os.path.join(base_root,'solver.prototxt')
snap_path=os.path.join(base_root,'snap/')


def creat_train_net(train_data, batch_size):
    n=caffe.NetSpec()
    n.data,n.label=L.ImageData(source=train_data,batch_size=batch_size,ntop=2,
                   transform_param=dict(scale=1./255))
    n.conv1=L.Convolution(n.data,kernel_size=5,stride=1,num_output=20,
             weight_filler=dict(type='xavier'))
    n.pool1=L.Pooling(n.conv1,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.conv2=L.Convolution(n.pool1,kernel_size=5,stride=1,num_output=50,
             weight_filler=dict(type='xavier'))
    n.pool2=L.Pooling(n.conv2,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.fc1=L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.relu1=L.ReLU(n.fc1,in_place=True)
    n.fc2=L.InnerProduct(n.fc1,num_output=10,weight_filler=dict(type='xavier'))
    n.loss=L.SoftmaxWithLoss(n.fc2,n.label)
    return n.to_proto()

def creat_test_net(test_data, batch_size):
    n=caffe.NetSpec()
    n.data,n.label=L.ImageData(source=test_data,batch_size=batch_size,ntop=2,
                   transform_param=dict(scale=1./255))
    n.conv1=L.Convolution(n.data,kernel_size=5,stride=1,num_output=20,
             weight_filler=dict(type='xavier'))
    n.pool1=L.Pooling(n.conv1,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.conv2=L.Convolution(n.pool1,kernel_size=5,stride=1,num_output=50,
             weight_filler=dict(type='xavier'))
    n.pool2=L.Pooling(n.conv2,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.fc1=L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.relu1=L.ReLU(n.fc1,in_place=True)
    n.fc2=L.InnerProduct(n.fc1,num_output=10,weight_filler=dict(type='xavier'))
    n.acc=L.Accuracy(n.fc2,n.label)
    return n.to_proto()

def creat_deploy_net():
    n=caffe.NetSpec()
    n.conv1=L.Convolution(bottom='data',kernel_size=5,stride=1,num_output=20,
             weight_filler=dict(type='xavier'))
    n.pool1=L.Pooling(n.conv1,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.conv2=L.Convolution(n.pool1,kernel_size=5,stride=1,num_output=50,
             weight_filler=dict(type='xavier'))
    n.pool2=L.Pooling(n.conv2,kernel_size=2,stride=2,pool=P.Pooling.MAX)
    n.fc1=L.InnerProduct(n.pool2,num_output=500,weight_filler=dict(type='xavier'))
    n.relu1=L.ReLU(n.fc1,in_place=True)
    n.fc2=L.InnerProduct(n.fc1,num_output=10,weight_filler=dict(type='xavier'))
    n.prob=L.Softmax(n.fc2)
    return n.to_proto()

def write_deploy(deploy_path):
    with open(deploy_path,'w') as f:
        f.write('name:"mnist_net"\n')
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        f.write(str(creat_deploy_net()))

def create_solver(train_net,test_net,snap_path):
    s=caffe_pb2.SolverParameter()
    s.random_seed=0xCAFFE
    s.train_net=train_net
    s.test_net.append(test_net)
    s.test_interval=500
    s.test_iter.append(100)
    s.max_iter=20000
    s.type="SGD"
    s.base_lr=0.01
    s.momentum=0.9
    s.weight_decay=5e-4
    s.lr_policy='inv'
    s.gamma=0.0001
    s.power=0.75
    s.display=200
    s.snapshot=1000
    s.snapshot_prefix=snap_path
    s.solver_mode=caffe_pb2.SolverParameter.CPU
    return s

with open(train_net,'w') as f:
    f.write(str(creat_train_net(train_data,64)))
with open(test_net,'w') as f:
    f.write(str(creat_test_net(test_data,100)))
write_deploy(deploy_net)
with open(solver_path,'w') as f:
    f.write(str(create_solver(train_net,test_net,snap_path)))