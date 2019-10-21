import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class CNN(BaseNetwork):
    def __init__(self, num_conv=3, c_in=1, channel=[96,48,1], kernel_size=[9,1,5], stride=[1,1,1], padding=[2,2,2]):
        layers = []
        for i in range(num_conv):
            layers += [
                nn.Conv2d(c_in if i == 0 else channel[i-1], channel[i], kernel_size[i], stride[i], padding[i], bias=True),
                nn.ReLU(inplace=True)
            ]

        self.feature = nn.Sequential(*layers)

    
    def forward(self, x):
        fmap = self.feature(x)
        return fmap
        

if __name__ == "__main__":
    cnn_t = CNN()
    x = torch.randn(10,1,32,32)
    y = cnn_t(x)

    import ipdb; ipdb.set_trace()

