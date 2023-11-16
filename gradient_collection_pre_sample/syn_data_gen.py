"""
    Generate synthetic data with noisy gradient and weights
    1. Run `train_and_collect.py` to store gradients and weights
    2. Run this file to generate syn data
"""
import logging
import os
import argparse
import copy
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.fx import symbolic_trace
from torch.fx.passes import graph_drawer
from functorch.compile import compiled_function, draw_graph
from torchviz import make_dot
from torchmetrics.image import TotalVariation

from models import LogisticRegression, SampleConvNet, GradientModel
from data_utils import get_data_loader, GradientWeightRecorder, SynImageDataset
from utils import count_parameters, set_seed, AverageMeter, imshow, normalize_syn_image,match_loss,imshow_single


from opacus import PrivacyEngine, GradSampleModule
logger = logging.getLogger('__main__')


def read_options():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--sess', default='test', type=str, help='Session name. Note: Make sure this is consistent with the one in `train_and_collect.py`')
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Which downstream task.")
    parser.add_argument("--dataset", choices=["mnist"],
                        default="mnist",
                        help="Which downstream task.")

    parser.add_argument("--train_batch_size", default=500, type=int,
                        help="Total batch size for training. Note: we don't use this args here.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")

    parser.add_argument('--lr_img', type=float, default=0.5, help='learning rate for updating synthetic data')

    parser.add_argument("--num_epoch", default=2000, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # Syn data related
    parser.add_argument("--spc", type=int, default=1,
                        help="Number of examples per class. Note that we also use this number as batch size")
    parser.add_argument("--load_syndata", action='store_true', help='Load synthetic data and continue to optimize.')

    # File related
    parser.add_argument("--data_dir", type=str, default='../data',
                        help="Folder to store datasets.")
    parser.add_argument("--dis_metric", type=str, default='cos',
                        help="the gradient distant match")
    args = parser.parse_args()
    return args


def setup_model(args, mnist_spec, test_loader):
    # Prepare model
    model = LogisticRegression(mnist_spec['width'] * mnist_spec['height'], mnist_spec['num_classes'])
    # freeze all the params
    for param in model.parameters():
        param.requires_grad = False
    model = model.to(args.device)
    num_params = count_parameters(model)
    print('total number of parameters: ', num_params)
    logger.info(f"total number of parameters: {num_params}")

    # Load dp_trained_model
    trained_model_path = os.path.join(args.result_dir, f'dp_trained_model.pth')
    print(f'load dp_trained_model from: {trained_model_path}')
    checkpoint = torch.load(trained_model_path)

    model.load_state_dict(checkpoint['net'])
    torch.set_rng_state(checkpoint['rng_state'])

    #accuracy = valid(args, model, test_loader, nn.CrossEntropyLoss())
    #print('accuracy from dp trained model: ', accuracy)
    #logger.info(f'accuracy from dp trained model: {accuracy}')

    return model


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def valid(args, model, test_loader, loss_fct):
    # Validation!
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            output = model(x)
            logits = model(x)[0]

            eval_loss = loss_fct(output, y)
            eval_losses.update(eval_loss.item())

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)

        if len(all_preds) == 0:
            all_preds.append(preds)
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds, axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)
    logger.info(f"Accuracy: {accuracy}")
    print(f"\n===> Valid Loss: {eval_losses.avg}, Valid Accuracy: {accuracy}")

    return accuracy

def train(args, mnist_specs, model,batch_index):
    # Load stored gradients and weights
    gradients_and_weights = torch.load(os.path.join(args.result_dir, 'gradients_weights.pth'))
    gradients = gradients_and_weights.gradients
    #gradients = torch.vstack(gradients).to(args.device)     # n x p

    #weights = [weight.to(args.device) for weight in gradients_and_weights.weights]
    weights = gradients_and_weights.weights

    criterion_dp_train = nn.CrossEntropyLoss().to(args.device)

    gradient_model = GradientModel(weights, criterion_dp_train, mnist_specs, args)
    gradient_model.load_weights(batch_index)
    ### Initialize the synthetic data from random noise
    num_classes = mnist_specs['num_classes']
    img_width = mnist_specs['width']
    img_height = mnist_specs['height']
    num_channel = mnist_specs['num_channel']



##TODO: starting change: single batch syn data/ gradient match/ method of attack list( DLG/IDLG/representation attack/method of cheng)
    image_syn = torch.randn(size=(num_classes * args.spc, num_channel, img_width, img_height), dtype=torch.float, requires_grad=True, device=args.device)
    if args.load_syndata:
        stored_syn_img = torch.load(os.path.join(args.result_dir,'syn_image.pt'))
        image_syn.data = stored_syn_img['syn_image'].to(args.device)
        imshow(image_syn.data.cpu(), args.spc)

    label_syn = torch.tensor([i // args.spc for i in range(num_classes * args.spc)], dtype=torch.long, device=args.device)  # [0,0,0, 1,1,1, ..., 9,9,9]
    label_syn = F.one_hot(label_syn) + torch.ones(size=(num_classes * args.spc, num_classes), device=args.device) / 100000

    #syn_dataset = torch.utils.data.TensorDataset(image_syn, label_syn)
    #syn_dataloader = DataLoader(syn_dataset, shuffle=False, batch_size=args.spc)

    #total_iters = args.num_epoch * (len(syn_dataloader.dataset) // syn_dataloader.batch_size)

    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
    optimizer_img.zero_grad()

    ### Setup a bunch of criterion
    criterion = nn.CrossEntropyLoss().to(args.device)
    # we empirically found this cos loss may not work
    criterion_grad_similarity_cos = nn.CosineEmbeddingLoss()
   # similarity_target = torch.ones(size=(gradients.shape[0],)).to(args.device)  # meaning we want them to be similar
    criterion_grad_similarity_cos2=torch.nn.CosineSimilarity(dim=0)

    criterion_grad_similarity_norm = nn.MSELoss().to(args.device)
    criterion_tv = TotalVariation().to(args.device)



    ### Training Need to only get one  
    global_step = 1
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    for epoch in range(args.num_epoch):

        pred = model(image_syn)
        loss_class = criterion(pred, label_syn)
        loss_tv = criterion_tv(image_syn)

        all_grad_x = gradient_model(image_syn, label_syn)
        # make_dot(all_grad_x, params=dict(gradient_model.named_parameters())).render('gm', format='png')

        loss_similarity = match_loss(all_grad_x, gradients[batch_index],args)

        ## Now we can take one update step
        alpha_t = np.cos(np.pi / 2.0 * global_step / (epoch+1))

        # TODO: find a better way to balance these three types of loss
        #total_loss = alpha_t * loss_class + \
        #                (1 - alpha_t) * loss_similarity + \
        #                0.0001 * loss_tv

        total_loss =loss_similarity
        total_loss.backward()
        losses.update(total_loss.item())
        optimizer_img.step()

        optimizer_img.zero_grad()
        global_step += 1


        logger.info(f'loss: {losses.val}')

        if epoch % 1000 == 0:
            torch.save({'syn_image': copy.deepcopy(image_syn.detach().cpu()),
                            'current_loss': losses.val,
                            'current_epoch': epoch},
                       os.path.join(args.result_dir, 'syn_image_{}_batch.pt'.format(batch_index)))
            imshow_single(image_syn.detach().cpu(), args.spc,batch_index,args.result_dir)
## End of editing 

def main():
    args = read_options()

    # Config Log file
    result_dir = os.path.join('./results', args.sess)
    if not os.path.exists(result_dir):
        raise FileExistsError("Sess not found! Run train_and_collect.py first. And make sure sess is consistent")
    args.result_dir = result_dir

    log_filename = os.path.join(result_dir, 'syn_data.log')
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logging.basicConfig(filename=log_filename,
                        encoding='utf8',
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        # level=logging.DEBUG if args.local_rank in [-1, 0] else logging.WARN,
                        force=True)  # This is MAGIC!
    logger.setLevel(logging.INFO)

    logger.info("Training parameters %s", args)

    # Set seed
    set_seed(args)

    # Get mnist data loader
    train_loader, test_loader, img_specs = get_data_loader(args)

    # Config Model
    model = setup_model(args, img_specs, test_loader)

    # Train
    train(args, img_specs, model,8000)




if __name__ == '__main__':
    main()