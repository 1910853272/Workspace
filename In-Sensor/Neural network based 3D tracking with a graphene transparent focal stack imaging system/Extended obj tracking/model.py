import torch.nn as nn
class VGG(nn.Module):

    def __init__(self, features, num_out=3,num_featurein=64,num_hiddenunit=4096, init_weights=True,dropout = True):
        #num_out is the # of VGG end output neurons
        #num_featurein is the # of features into the classifer, dependes on the cfg choice.
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) #to pool the feature map s.t. the output is 7 by 7, in original vgg, the input is 224 x 224, and before adpative pool, the feature map is already 7 by 7 so this does nothing.For larger image, this will pool the image s.t. the output is 7 by 7.
        if dropout:
            self.classifier = nn.Sequential(
                nn.Linear(num_featurein *7 * 7, num_hiddenunit),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hiddenunit, num_hiddenunit),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hiddenunit, num_out),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_featurein *7 * 7, num_hiddenunit),
                nn.ReLU(True),
                nn.Linear(num_hiddenunit, num_hiddenunit),
                nn.ReLU(True),
                nn.Linear(num_hiddenunit, num_out),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
class VGG_DDFF(nn.Module):
#VGG network with structure based on DDFF paper, i.e., features are calculated from each FS independently and then recombine
    def __init__(self, features, num_out=3,num_featurein=64,num_hiddenunit=100, init_weights=True,dropout = True):
        #num_out is the # of VGG end output neurons
        #num_featurein is the # of features into the classifer, dependes on the cfg choice.
        super(VGG_DDFF, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) #to pool the feature map s.t. the output is 7 by 7, in original vgg, the input is 224 x 224, and before adpative pool, the feature map is already 7 by 7 so this does nothing.For larger image, this will pool the image s.t. the output is 7 by 7.
        if dropout:
            self.classifier = nn.Sequential(
                nn.Linear(num_featurein *7 * 7, num_hiddenunit),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hiddenunit, num_hiddenunit),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(num_hiddenunit, num_out),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_featurein *7 * 7, num_hiddenunit),
                nn.ReLU(True),
                nn.Linear(num_hiddenunit, num_hiddenunit),
                nn.ReLU(True),
                nn.Linear(num_hiddenunit, num_out),
            )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        B,nF,H,W=x.shape
        x=x.view(B*nF,1,H,W)# reshape nF dimension into Batch dimension, following DDFF
        x = self.features(x)
        x=x.view(B,nF,x.shape[2],x.shape[3]) #reform back to nF dimension
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=2,batch_norm=False):
    #in_channels=2 is default for two plane FS 
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


#config A are for vgg11, config D are vgg16
cfg = {
    'A_8': [8, 'M', 16, 'M', 32, 32, 'M', 64, 64, 'M', 64, 64, 'M'],
    'A_4':[4, 'M', 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M'],
    'A_2':[2, 'M', 4, 'M', 8, 8, 'M', 16, 16, 'M', 16, 16, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D_4': [4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M'],
     'A_4_DDFF':[4, 'M', 8, 'M', 16, 16, 'M', 32, 32, 'M', 32, 32, 'M',1],
    'D_4_DDFF':[4, 4, 'M', 8, 8, 'M', 16, 16, 16, 'M', 32, 32, 32, 'M', 32, 32, 32, 'M',1], #convert multiple feature map to one low resolution sharpness measure map per focal plane 
    
}


def vgg(config_key='A_8',**kwargs):
    """VGG N-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    model = VGG(make_layers(cfg[config_key]), num_featurein=cfg[config_key][-2], **kwargs)  #cfg[config_key][-2] to get the  output feature number. 
    return model


def vgg_bn(config_key='A_8', **kwargs):
    """VGG N-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = VGG(make_layers(cfg[config_key], batch_norm=True), num_featurein=cfg[config_key][-2], **kwargs)
    return model

def vgg_DDFF(nF=2,config_key='D_4_DDFF', **kwargs):
    
    model = VGG_DDFF(make_layers(cfg[config_key],in_channels=1),num_featurein=nF, **kwargs)
    return model


def vgg_DDFF_bn(nF=2,config_key='D_4_DDFF', **kwargs):
    
    model = VGG_DDFF(make_layers(cfg[config_key],in_channels=1, batch_norm=True),num_featurein=nF, **kwargs)
    return model