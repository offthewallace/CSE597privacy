import logging
import os
import shutil
import datetime
from datetime import timedelta
import argparse
import random
from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from utils import set_seed, count_parameters, AverageMeter, restore_param, save_checkpoint, imshow
from data_utils import get_data_loader, SynImageDataset, GradientWeightRecorder
from models import SampleConvNet, LogisticRegression

import json

from opacus import PrivacyEngine, GradSampleModule
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager

logger = logging.getLogger('__main__')


def read_options():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--sess', default='test', type=str, help='session name')
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="Which downstream task.")
    parser.add_argument("--dataset", choices=["mnist"],
                        default="mnist",
                        help="Which downstream task.")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=0.005, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_epoch", default=5, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    # Privacy params
    parser.add_argument("--max_grad_norm", type=float, default=3.,
                        help="The maximum L2 norm of per-sample gradients before they are aggregated by the averaging "
                             "step.")
    parser.add_argument("--epsilon", type=float, default=2.,
                        help="Privacy budget.")
    parser.add_argument("--delta", type=float, default=1e-5,
                        help="Generally, it should be set to be less than the inverse of the size of the training "
                             "dataset.")

    # File related
    parser.add_argument("--data_dir", type=str, default='../data',
                        help="Folder to store datasets.")

    args = parser.parse_args()
    return args


# Define a class to store the gradients and weights



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def setup_model(args, img_specs):
    # Prepare model
    model = LogisticRegression(img_specs['width'] * img_specs['height'], img_specs['num_classes'])
    model = model.to(args.device)
    num_params = count_parameters(model)

    print('total number of parameters: ', num_params)
    logger.info(f"total number of parameters: {num_params}")

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


def train(args, model, train_loader, test_loader):
    n_epoch = args.num_epoch
    t_total = n_epoch * (len(train_loader.dataset) // args.train_batch_size)
    best_acc, epoch = 0, 0

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Privacy setting
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.num_epoch,
        target_epsilon=args.epsilon,
        target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        poisson_sampling=False
    )

    # Create a GradientWeightRecorder instance
    recorder = GradientWeightRecorder()

    # Train!
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
            net_parameters = list(model.parameters())
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()

            losses.update(loss.item())

            optimizer.step()
            # Record gradients and weights
            #save the whole model
            # model.linear.bias.grad/grad_sample
            # model.linear.weights.grad/grad_sample
            recorder.record(model, net_parameters)

            if global_step % 7500==0:
                recorder.save_grad_sample(model, args.result_dir,global_step,net_parameters,batch)

            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )
            


        accuracy = valid(args, model, test_loader, criterion)
        epsilon = privacy_engine.get_epsilon(args.delta) 
        print(f"(ε = {epsilon:.2f}, δ = {args.delta})")
        logger.info(f"(ε = {epsilon:.2f}, δ = {args.delta})")
        if best_acc < accuracy:
            best_acc = accuracy
            save_checkpoint(model, epoch, args)
        scheduler.step()
        model.train()
        losses.reset()

        epoch += 1
        if epoch % args.num_epoch == 0:
            break

    torch.save(recorder, os.path.join(args.result_dir, 'gradients_weights.pth'))
    print(f"\n===> Best Accuracy: {best_acc}")
    logger.info(f"Best Accuracy: {best_acc}")


def main():
    args = read_options()
    # Config result file
    result_dir = os.path.join('./results', args.sess)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)       # just rerun, it takes <1min
    os.makedirs(result_dir)

    args.result_dir = result_dir
    # logger = get_logger('main', log_filename=log_path + '/' + str(dt) + 'log.log')
    log_filename = os.path.join(result_dir, 'train_and_collect.log')
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

    # Prepare dataset
    train_loader, test_loader, img_specs = get_data_loader(args)

    # Model & Tokenizer Setup
    model = setup_model(args, img_specs)

    train(args, model, train_loader, test_loader)


if __name__ == '__main__':
    main()
