import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from copy import deepcopy

# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

class GradientWeightRecorder:
    def __init__(self):
        self.gradients = []
        self.weights = []
        #self.sample_gradient =[]

    def record(self, model, netparam):

        gw_real = list((p.grad.detach().clone() for p in netparam))
        weight_real = list((p.grad.detach().clone() for p in netparam))
        #grad = torch.cat([p.grad.data.view(-1) for p in model.parameters()]).cpu()
        #TODO:should I change the gradient to flat?
        #weight = torch.cat([p.data.view(-1) for p in model.parameters()]).cpu()
        # weight = deepcopy(model._module.state_dict())
        #weight_samplegrad=deepcopy(model.linear.weight.grad_sample.detach().clone())
        #bias_samplegrad=deepcopy(model.linear.bias.grad_sample.detach().clone())
        self.gradients.append(gw_real)
        self.weights.append(netparam)
        #self.sample_gradient.append([weight_samplegrad,bias_samplegrad])

    def save_grad_sample(self, model, path,steps,netparam,batch):
        weight_grad_sample = model.linear.weight.grad_sample
        torch.save(weight_grad_sample, os.path.join(path, f'exp3_batch8_weight_grad_sample_no_dp_{steps}.pth'))

        sum_grad_real = list((p.grad.detach().clone() for p in netparam))
        torch.save(sum_grad_real, os.path.join(path, f'exp3_batch8_sum_grad_dp_{steps}.pth'))

        torch.save(netparam, os.path.join(path, f'exp3_batch8_sum_weight_dp_{steps}.pth'))

        torch.save(batch, os.path.join(path, f'exp3_batch8_batch_dp_{steps}.pth'))

        # Save the grad_sample of bias
        bias_grad_sample = model.linear.bias.grad_sample
        torch.save(bias_grad_sample, os.path.join(path, f'exp3_batch8_bias_grad_sample_no_dp_{steps}.pth'))
        

class SynImageDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data = data_tensor
        self.labels = label_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return self.data[idx], self.labels[idx]

def get_data_loader(args):
    if args.dataset == 'mnist':
        img_specs = {
            'num_classes': 10,
            'num_channel': 1,
            'width': 28,
            'height': 28
        }
        data_dir = os.path.join(args.data_dir, 'mnist')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
            ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)

        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    else:
        raise NotImplementedError('Not matching dataset found')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=8)
    test_loader = DataLoader(test_dataset,
                              batch_size=args.eval_batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=8)

    return train_loader, test_loader, img_specs