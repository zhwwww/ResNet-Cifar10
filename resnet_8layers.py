import mxnet as mx


def residual_unit(data, num_filter, stride, dim_match, name):
    conv0=mx.symbol.Convolution(data=data,num_filter=num_filter//4,kernel=(1,1),stride=stride,pad=(0,0),
                                no_bias=True,name=name+'_conv0')
    bn0=mx.symbol.BatchNorm(data=conv0,fix_gamma=True,eps=2e-5,momentum=0.9,name=name+'_bn0')
    act0 = mx.symbol.Activation(data=bn0, act_type='relu', name=name+'_relu0')
    conv1=mx.symbol.Convolution(data=act0,num_filter=num_filter//4,kernel=(3,3),stride=(1,1),pad=(1,1),
                                no_bias=True,name=name+'_conv1')
    bn1=mx.symbol.BatchNorm(data=conv1,fix_gamma=True,eps=2e-5,momentum=0.9,name=name+'_bn1')
    act1 = mx.symbol.Activation(data=bn1, act_type='relu', name=name+'_relu1')
    conv2=mx.symbol.Convolution(data=act1,num_filter=num_filter,kernel=(1,1),stride=(1,1),pad=(0,0),
                                no_bias=True,name=name+'_conv2')
    bn2=mx.symbol.BatchNorm(data=conv2,fix_gamma=True,eps=2e-5,momentum=0.9,name=name+'_bn2')
    # option B
    if dim_match:
        shortcut = data
    else: 
        shortcut = mx.symbol.Convolution(data=data,num_filter=num_filter,kernel=(1,1),stride=stride,pad=(0,0),
                                         no_bias=True,name=name+'_shortcut0')
    out = shortcut+bn2
    return mx.symbol.Activation(data=out,act_type='relu',name=name+'relu2')

# define a 56 layers of resnet
def resnet():
    data = mx.symbol.Variable(name='data')
    data = mx.symbol.identity(data=data,name='id')
    # BN
    data = mx.symbol.BatchNorm(data=data,fix_gamma=True,eps=2e-5,momentum=0.9,name='bn_data')
    # use for cifar10
    body = mx.symbol.Convolution(data=data,num_filter=16,kernel=(3,3),stride=(1,1),pad=(1,1),
                                 no_bias=True,name='conv0')
    for i in range(9):
        body = residual_unit(data=body,num_filter=16,stride=(1,1),dim_match=True,name='unit0_{}'.format(i))
    for i in range(9):
        body = residual_unit(data=body,num_filter=32,stride = (2,2) if i==0 else (1,1) ,dim_match= False if i==0 else True,name='unit1_{}'.format(i))
    for i in range(9):
       body = residual_unit(data=body,num_filter=64,stride = (2,2) if i==0 else (1,1) ,dim_match= False if i==0 else True,name='unit2_{}'.format(i))

    pool = mx.symbol.Pooling(data=body, global_pool=True, kernel=(7, 7), stride=(1,1), pad=(0,0), pool_type='avg', name='pool0')
    flat = mx.symbol.Flatten(data=pool)
    fc = mx.symbol.FullyConnected(data=flat,num_hidden=10,name='fc0')
    return mx.symbol.SoftmaxOutput(data=fc,name='softmax')


def get_symbol():
    return resnet()

#def get_symbol(num_classes=10):
#    data = mx.symbol.Variable('data')
#    data = mx.sym.Flatten(data=data)
#    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
#    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
#    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
#    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
#    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
#    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
#    return mlp