import os
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)


def save_checkpoint(net, epoch, args):
    """
    Save the checkpoint for an opacus processed model
    """
    state = {
        'net': net._module.state_dict(),
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
    }

    torch.save(state, os.path.join(args.result_dir, f'dp_trained_model.pth'))

def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)


def make_the_loss_within_10(value):
    scales = ["ones", "tens", "hundreds", "thousands", "millions", "billions", "trillions"]

    scale_index = 0

    while value >= 100 and scale_index < len(scales) - 1:
        value /= 100
        scale_index += 1

    return value


def normalize_syn_image(image_syn):
    shape = image_syn.shape
    image_syn = image_syn.view(image_syn.shape[0], -1)
    min_value = torch.min(image_syn, dim=1, keepdim=True).values
    max_value = torch.max(image_syn, dim=1, keepdim=True).values

    image_syn = (image_syn - min_value) / (max_value - min_value)
    return image_syn.view(shape)


def imshow_single(images, spc,batch_index,result_dir):
    # Convert PyTorch tensor to NumPy array
    image_array = images.numpy()

    # Create a 10x10 grid for displaying 100 images
    fig, axes = plt.subplots(10, spc, figsize=(10, 10))

    for i in range(10):
            image_index = i 
            image = image_array[image_index, 0]  # Assuming grayscale images
            axes[i].imshow(image, cmap='gray')
            axes[i].axis('off')

    plt.tight_layout()  # Ensure proper spacing
    plt.show()
    directory=os.path.join(result_dir, 'syn_image_{}_batch_class{}.png'.format(batch_index,spc))
    fig.savefig(directory)



def imshow(images, spc):
    # Convert PyTorch tensor to NumPy array
    image_array = images.numpy()

    # Create a 10x10 grid for displaying 100 images
    fig, axes = plt.subplots(10, spc, figsize=(10, 10))

    for i in range(10):
        for j in range(spc):
            image_index = i * 10 + j
            image = image_array[image_index, 0]  # Assuming grayscale images
            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].axis('off')

    plt.tight_layout()  # Ensure proper spacing
    plt.show()

def distance_wb(gwr, gws):
    #from Chen 2022
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)


def match_loss(gw_syn, gw_real, args):
    #From Chen 2022
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'gm':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s' % dis_metric)

    return dis

if __name__ == '__main__':
    stored_syn_img = torch.load('./results/eps2bs50epoch10/syn_image.pt')
    syn_img = stored_syn_img['syn_image']
    imshow(syn_img, 10)