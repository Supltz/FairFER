from utils import *
import numpy as np
from resnets import *
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset import *
from EfficientNets import *
from swin_transformer import *
from vit_pytorch import ViT
from collections import Counter





parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--description', required=True, default="test", type=str)
args = parser.parse_args()

def train(epoch, trainloader):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs,target[:,0])
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
        _, prediction = torch.max(outputs.data, 1)

    # Print metrics for each task
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader)}')

    return running_loss / len(trainloader)

def val(epoch, valloader):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    all_targets, all_predictions = [], []
    with torch.no_grad():  # Disable gradient computation
        for batch_idx, (data, target) in enumerate(tqdm(valloader)):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target[:,0])
            running_loss = running_loss + loss.item()
            _, prediction = torch.max(outputs.data, 1)

            # Directly append tensors to lists
            all_predictions.append(prediction.cpu())
            all_targets.append(target[:,0].cpu())


        # Concatenate tensors and convert to numpy arrays
        all_targets = torch.cat(all_targets).numpy()
        all_predictions = torch.cat(all_predictions).numpy()

        # Calculate metrics for each task
        acc, acc_per_class = calculate_acc(all_targets, all_predictions)

    # Print metrics for each task
    print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(valloader)}')
    print(f'Average Accuracy: {acc * 100:.2f}%, Per-Class Accuracy: {acc_per_class * 100}')

    return running_loss / len(valloader), acc, acc_per_class


def main():

    writer = SummaryWriter('runs/{}'.format(args.description))
    best_val_acc = 0
    epochs_no_improve = 0
    early_stop_epochs = 5

    for epoch in range(num_epochs):
        train_loss = train(epoch, trainloader)
        # Log average metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)

         # Validation phase
        val_loss, val_acc, val_acc_per_class = val(epoch, valloader)

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        for cls in range(len(val_acc_per_class)):
            writer.add_scalar(f'Accuracy/AClass_{cls}/Val', val_acc_per_class[cls], epoch)

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), './checkpoints/best_model_{}.pth'.format(args.description))

        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == early_stop_epochs:
            print("Early stopping triggered")
            break
    writer.close()  # Close the TensorBoard writer

if __name__=="__main__":

    data_path = '/vol/lian/datasets/basic/Image/aligned/'
    train_list, val_list = RAF_spliting(data_path, 'emotion')

    batchsize = 8
    lr = 0.0003
    momentum = 0.9
    weight_decay = 5e-4


    # Balanced dataset and dataloader
    trainset = RAFDataset(train_list, data_path, mode='train')
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

    valset = RAFDataset(val_list, data_path, mode='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)


    # Training loop
    num_epochs = 200

    model = resnet34(pretrained=True,num_classes=7)
    # model = ViT(image_size=224, patch_size=16, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1,
    #     emb_dropout=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    main()