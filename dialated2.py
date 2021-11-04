import torch
import torch.nn as nn


class DilatedCNN(nn.Module):
  
    def __init__(self,load_weights=False):
        super(DilatedCNN,self).__init__()

       def __init__(self,load_weights=False):
        super(DilatedCNN,self).__init__()

        self.convlayers = nn.Sequential(
            nn.Conv2d(3,16,9, stride = 2, padding = 0, dilation=2),
            nn.ReLU(),
            nn.Conv2d(16,32,7, stride = 2, padding= 0, dilation = 2),
            nn.ReLU(),
            nn.Conv2d(32,16,5, stride = 2, padding = 0, dilation=2),
            nn.ReLU(),
        )
        
        if not load_weights:
            self._initialize_weights()

    def forward(self,img_tensor):
        x=self.convlayers(img_tensor)
        #x = x.view(-1,2304)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)