import torch.nn as nn
from models import fc_model
from dataset import get_loader
from args import args_test, args_val
from train import load_checkpoint
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    args2 = args_val()
    val_loader, val_set = get_loader(args2.data_dir, batch_size=args2.batch_size, shuffle=True,
                                         num_workers=args2.num_workers, drop_last=False, args=args2)

    data_size = val_set.get_data_size()
    num_classes = val_set.get_num_classes()

    # original image shape 450*675
    # image size after args.crop_size * args.crop_size * 3
    instance_size = val_set.get_instance_size()
    print("instance size is ", instance_size)

    # Build the model
    model = fc_model(input_size=instance_size, num_classes=num_classes, dropout=args2.dropout)

    # create optimizer
    params = list(model.parameters())
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, lr=args2.learning_rate)
    print("parameters number {}".format(param_count))

    # load the pretrained model
    load_checkpoint(model, optimizer, 'data/best_model_L.pth')
    model = model.to(device)
    model.eval()
    print("model loaded & starting testing ...\n\n")

    total_correct_preds = 0.0
    total = 1e-10
    # run on validation set
    with torch.no_grad():
        for step, (image_input, class_idxs) in enumerate(val_loader):
            # move all data loaded from dataloader to gpu
            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)

            # feed-forward data in the model
            output = model(image_input)  # 32 * 150528 --> 32 * 11

            # accuracy computation
            _, pred_idx = torch.max(output, dim=1)
            total_correct_preds += torch.sum(pred_idx == class_idxs).item()
            total += output.size(0)

        accuracy = round(total_correct_preds / total, 4)

        print('\naccuracy: {}\n'.format(accuracy), end="")


if __name__ == '__main__':
    main()