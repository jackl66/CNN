from args import get_parser, args_val
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os, random, math
from models import fc_model
from dataset import get_loader
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------


def main(args, args2):
    print("train on ", device)
    # train set loader
    data_loader, dataset = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, drop_last=False, args=args)
    # validation set loader
    val_loader, val_set = get_loader(args2.data_dir, batch_size=args2.batch_size, shuffle=False,
                                     num_workers=args2.num_workers, drop_last=False, args=args2)
    data_size = dataset.get_data_size()
    num_classes = dataset.get_num_classes()

    # original image shape 450*675
    # image size after args.crop_size * args.crop_size * 3
    instance_size = dataset.get_instance_size()
    print("instance size is ", instance_size)
    # Build the model
    model = fc_model(input_size=instance_size, num_classes=num_classes, dropout=args.dropout)

    # create optimizer
    params = list(model.parameters())
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args.learning_rate)
    print(" parameters number {}".format(param_count))

    # Using CrossEntropy cost function softmax is included
    label_crit = nn.CrossEntropyLoss()
    # load_checkpoint(model, optimizer, 'data/68best_model.pth')
    model = model.to(device)

    print("model created & starting training ...\n\n")
    # Training script
    best = 0
    a_t = []
    l_t = []
    a_v = []
    l_v = []
    early_stopping = []
    for epoch in range(args.num_epochs):

        total_correct_preds = 0.0
        total = 1e-10
        loss = 0.0

        # saving model
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        # step loop for training
        model.train()
        for step, (image_input, class_idxs) in enumerate(data_loader):
            # move all data loaded from dataloader to gpu
            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)

            # feed-forward data in the model
            output = model(image_input)  # 32 * 150528 --> 32 * 11

            # compute losses
            state_loss = label_crit(output, class_idxs)  # --> 32 * 1
            print(state_loss,99,class_idxs)
            print(output.size())
            exit()
            # aggregate loss for logging
            loss += state_loss.item()

            # back-propagate the loss in the model & optimize
            model.zero_grad()
            # optimizer.zero_grad()
            state_loss.backward()
            optimizer.step()

            # accuracy computation
            _, pred_idx = torch.max(output, dim=1)
            total_correct_preds += torch.sum(pred_idx == class_idxs).item()
            total += output.size(0)

        # epoch accuracy & loss
        accuracy = round(total_correct_preds / total, 4)
        loss = round(loss / total, 4)
        a_t.append(accuracy)
        l_t.append(loss)
        print('epoch {}: accuracy: {}, loss: {}'.format(epoch, accuracy, loss), end="")

        # run on validation set
        model.eval()
        loss = 0.0
        total = 0
        total_correct_preds = 0

        with torch.no_grad():
            for step, (image_input, class_idxs) in enumerate(val_loader):
                # move all data loaded from dataloader to gpu
                class_idxs = class_idxs.to(device)
                image_input = image_input.to(device)

                # feed-forward data in the model
                output = model(image_input)  # 32 * 150528 --> 32 * 11

                # compute losses
                state_loss = label_crit(output, class_idxs)  # --> 32 * 1
                print(state_loss.size())
                # aggregate loss for logging
                loss += state_loss.item()

                # accuracy computation
                _, pred_idx = torch.max(output, dim=1)
                total_correct_preds += torch.sum(pred_idx == class_idxs).item()
                total += output.size(0)

            accuracy = round(total_correct_preds / total, 4)
            loss = round(loss / total, 4)

            print('\nvalidation {}: accuracy: {}, loss: {}\n'.format(epoch, accuracy, loss), end="")
            # save best model
            a_v.append(accuracy)
            l_v.append(loss)
            is_best = accuracy > best
            if is_best:
                best = accuracy
                save_checkpoint(checkpoint)
        # early stopping check
        early_stopping.append(accuracy)
        ind = early_stopping.index(best)
        print(ind, "   ", best, "\n")
        if len(early_stopping) - ind > args.patience:
            print('patience has ran out, only wait for {} epochs'.format(args.patience))
            break
    # plot results
    x = np.arange(len(a_t))
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.plot(x, a_t, 'r', label='training accuracy')
    plt.plot(x, a_v, 'g', label='validation accuracy')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(x, l_t, 'r', label='training loss')
    plt.plot(x, l_v, 'g', label='validation loss')
    plt.legend()

    print("\ndone")


def save_checkpoint(state,file_name='best_model.pth'):
    torch.save(state,file_name)
    print("saving succeed\n")


def load_checkpoint(model,optimizer,path):
    checkpoint = torch.load(path, map_location= torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if __name__ == '__main__':
    args = get_parser()
    args2 = args_val()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)


    main(args,args2)
