import logging
import os
import datetime
from datetime import timedelta
import argparse
import random
from tqdm import tqdm

import sys

sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from utils import set_seed, count_parameters, AverageMeter, restore_param, save_checkpoint, imshow
from data_utils import get_data_loader, SynImageDataset
from models import SampleConvNet, LogisticRegression


from opacus import PrivacyEngine, GradSampleModule
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

logger = logging.getLogger('__main__')


def read_options():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--sess', default='eps2bs50epoch10', type=str, help='session name')
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Which downstream task.")
    parser.add_argument("--dataset", choices=["mnist"],
                        default="mnist",
                        help="Which downstream task.")

    parser.add_argument("--train_batch_size", default=20, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=10., type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epoch", default=20, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    # File related
    parser.add_argument("--data_dir", type=str, default='../data',
                        help="Folder to store log files.")

    # Syn data related
    parser.add_argument("--spc", type=int, default=10,
                        help="Number of example per class")


    args = parser.parse_args()
    return args


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def setup_model(args, img_specs, test_loader):
    # Prepare model
    model = LogisticRegression(img_specs['width'] * img_specs['height'], img_specs['num_classes'])
    model = model.to(args.device)
    num_params = count_parameters(model)
    print('total number of parameters: ', num_params)
    logger.info(f"total number of parameters: {num_params}")

    # Load dp_trained_model
    trained_model_path = os.path.join(args.result_dir, f'dp_trained_model.pth')
    print(f'load dp_trained_model from: {trained_model_path}')
    checkpoint = torch.load(trained_model_path)

    model = GradSampleModule(model)
    model.load_state_dict(checkpoint['net'])
    torch.set_rng_state(checkpoint['rng_state'])

    accuracy = valid(args, model, test_loader, nn.CrossEntropyLoss())
    print('accuracy from dp trained model: ', accuracy)
    logger.info(f'accuracy from dp trained model: {accuracy}')

    return model


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

def continue_train(args, model, img_specs, test_loader):
    num_classes = img_specs['num_classes']
    img_width = img_specs['width']
    img_height = img_specs['height']
    num_channel = img_specs['num_channel']

    ### Load synthetic image
    stored_syn_img = torch.load(os.path.join(args.result_dir, 'syn_image.pt'))
    image_syn = stored_syn_img['syn_image']
    imshow(image_syn, args.spc)

    ### Create train_loader with synthetic image
    label_syn = torch.tensor([i // args.spc for i in range(num_classes * args.spc)], dtype=torch.long)
    syn_dataset = torch.utils.data.TensorDataset(image_syn, label_syn)
    train_loader = DataLoader(syn_dataset, shuffle=False, batch_size=args.spc)

    # First, Run inference
    accuracy_on_syn = valid(args, model, train_loader, nn.CrossEntropyLoss())
    print(f"\n===> Accuracy on Synthetic Image: {accuracy_on_syn}")
    logger.info(f"Accuracy on Synthetic Image: {accuracy_on_syn}")

    n_epoch = args.num_epoch
    t_total = n_epoch * (len(train_loader.dataset) // train_loader.batch_size)
    best_acc, epoch = 0, 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    ### Train!
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step = 0

    while True:
        model.train()

        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            losses.update(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )

        accuracy = valid(args, model, test_loader, criterion)

        if best_acc < accuracy:
            best_acc = accuracy

        scheduler.step()
        model.train()
        losses.reset()

        epoch += 1
        if epoch % args.num_epoch == 0:
            break

    print(f"\n===> Best Accuracy: {best_acc}")
    logger.info(f"Best Accuracy: {best_acc}")


def main():
    args = read_options()

    # Config Log file
    result_dir = os.path.join('./results', args.sess)
    if not os.path.exists(result_dir):
        raise FileExistsError("Sess not found! Run train_and_collect.py first. And make sure sess is consistent")
    if not os.path.exists(os.path.join(result_dir, 'syn_image.pt')):
        raise FileExistsError("Synthetic image not found! Run syn_data_gen.py first. And make sure sess is consistent")
    args.result_dir = result_dir

    log_filename = os.path.join(result_dir, 'continue_train.log')
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

    train_loader, test_loader, img_specs = get_data_loader(args)

    # Config Model
    model = setup_model(args, img_specs, test_loader)

    # Continue Train
    continue_train(args, model, img_specs, test_loader)


if __name__ == '__main__':
    main()
