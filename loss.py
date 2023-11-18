import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(VGGLoss, self).__init__()
        # self.vgg = VGG19().to(device)
        index = 31
        vgg_model = torchvision.models.vgg16(pretrained=True).to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.features.children())[:index])
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # self.criterion = nn.MSELoss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(self.vgg_preprocess(x)), self.vgg(self.vgg_preprocess(y))
        # loss = 0
        # for i in range(len(x_vgg)):
        #     loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        loss = torch.mean((self.instancenorm(x_vgg) - self.instancenorm(y_vgg)) ** 2)
        return loss
    
    def vgg_preprocess(self, batch): 
        tensor_type = type(batch.data) 
        (r, g, b) = torch.chunk(batch, 3, dim=1) 
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR 
        batch = batch * 255  # * 0.5  [-1, 1] -> [0, 255] 
        mean = tensor_type(batch.data.size()).cuda() 
        mean[:, 0, :, :] = 103.939 
        mean[:, 1, :, :] = 116.779 
        mean[:, 2, :, :] = 123.680 
        batch = batch.sub(Variable(mean))  # subtract mean 
        return batch 
    
