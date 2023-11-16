import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from opacus import PrivacyEngine, GradSampleModule


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # out = self.linear(F.softmax(x))
        out = self.linear(x)
        return out


class GradientModel(nn.Module):
    """
    Generate gradient matrix
    """
    def __init__(self, weights, criterion, img_specs, args):
        """
        Arguments:

        """
        super(GradientModel, self).__init__()
        self.weights = weights
        self.img_specs = img_specs
        self.model = LogisticRegression(img_specs['width'] * img_specs['height'], img_specs['num_classes']).to(args.device)


        self.args = args
        self.criterion = criterion

    def forward(self, x, y):
        # Now we need to compute gradient similarity
        # Using an explicit for loop, not sure how to avoid it
        

                # compute grad w.r.t. syn data a.k.a. x
        pred_x = self.model(x)
        loss_x = self.criterion(pred_x, y)
        grad_x = torch.autograd.grad(loss_x, self.model.parameters(),create_graph=True)

        #grad_x_flatten = torch.cat([grad.view(-1) for grad in grad_x])
        #all_grad_x.append(grad_x_flatten)

        #for p in model_i.parameters():
        #    p.requires_grad = False


        #all_grad_x = torch.vstack(all_grad_x).to(self.args.device)

        return list(grad_x)
    
    def load_weights(self,index): 
        offset = 0
        for p in self.model.parameters():
            #numel = p.data.numel()
            p.data = self.weights[index][offset]
            offset += 1