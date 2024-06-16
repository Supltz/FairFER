from utils import *
import numpy as np
from FT_R34 import ResNet34_FT, EntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from dataset import *




parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--description', required=True, default="test", type=str)
parser.add_argument('--attr', required=True, default="gender", type=str)

args = parser.parse_args()

def train(epoch, trainloader):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        if args.attr == 'age':
            loss = criterion(output, target[:, 1])
        elif args.attr == 'gender':
            loss = criterion(output, target[:, 0])
        elif args.attr == 'race':
            loss = criterion(output, target[:, 2])

        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()

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
            output = model(data)

            if args.attr == 'age':
                loss = criterion(output, target[:, 1])
            elif args.attr == 'gender':
                loss = criterion(output, target[:, 0])
            elif args.attr == 'race':
                loss = criterion(output, target[:, 2])

            running_loss += loss.item()

            _, prediction = torch.max(output.data, 1)

            # Directly append tensors to lists
            all_predictions.append(prediction.cpu())

            if args.attr == 'age':
                all_targets.append(target[:, 1].cpu())
            elif args.attr == 'gender':
                all_targets.append(target[:, 0].cpu())
            elif args.attr == 'race':
                all_targets.append(target[:, 2].cpu())


    # Concatenate tensors and convert to numpy arrays
    all_targets = torch.cat(all_targets).numpy()
    all_predictions = torch.cat(all_predictions).numpy()

    # Calculate metrics for each task
    acc, acc_per_class = calculate_acc(all_targets, all_predictions)

    # Print metrics for each task
    print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(valloader)}')
    if args.attr == 'age':
        print(f'Age - Overall Accuracy: {acc * 100:.2f}%, Per-Class Accuracy: {acc_per_class * 100}')
    elif args.attr == 'gender':
        print(f'Gender - Overall Accuracy: {acc * 100:.2f}%, Per-Class Accuracy: {acc_per_class * 100}')
    elif args.attr == 'race':
        print(f'Race - Overall Accuracy: {acc * 100:.2f}%, Per-Class Accuracy: {acc_per_class * 100}')

    return running_loss / len(valloader), acc, acc_per_class


def main():

    writer = SummaryWriter('runs/{}'.format(args.description))
    best_val_accuracy = 0
    epochs_no_improve = 0
    early_stop_epochs = 5

    for epoch in range(num_epochs):
        train_loss = train(epoch, trainloader)
        # Log average metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)

        val_loss, val_acc, val_acc_per_class, = val(epoch, valloader)

        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/Val', val_loss, epoch)

        if args.attr == 'age':
            writer.add_scalar('Accuracy/Age/Val', val_acc, epoch)
            for cls in range(len(val_acc_per_class)):
                writer.add_scalar(f'Accuracy/Age_Class_{cls}/Val', val_acc_per_class[cls], epoch)
        elif args.attr == 'gender':
            writer.add_scalar('Accuracy/Gender/Val', val_acc, epoch)
            for cls in range(len(val_acc_per_class)):
                writer.add_scalar(f'Accuracy/Gender_Class_{cls}/Val', val_acc_per_class[cls], epoch)
        elif args.attr == 'race':
            writer.add_scalar('Accuracy/Race/Val', val_acc, epoch)
            for cls in range(len(val_acc_per_class)):
                writer.add_scalar(f'Accuracy/Race_Class_{cls}/Val', val_acc_per_class[cls], epoch)


        avg_val_accuracy = val_acc

        # Check for improvement
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), './checkpoints/best_model_{}_{}.pth'.format(args.attr, args.description))

        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == early_stop_epochs:
            print("Early stopping triggered")
            break
    writer.close()  # Close the TensorBoard writer

if __name__=="__main__":

    data_path = '/vol/lian/datasets/utkcropped/'
    trainIDs, valIDs = UTK_spliting(data_path)

    if args.attr == 'age':
        num_classes = 5
        batchsize = 8
        lr = 0.0001
        momentum = 0.9
        weight_decay = 5e-4
    elif args.attr == 'gender':
        num_classes = 2
        batchsize = 8
        lr = 0.0001
        momentum = 0.9
        weight_decay = 5e-4
    elif args.attr == 'race':
        num_classes = 4
        batchsize = 8
        lr = 0.0001
        momentum = 0.9
        weight_decay = 5e-4


    # Balanced dataset and dataloader
    trainset = UTKDataset(trainIDs, data_path, mode='train')
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

    valset = UTKDataset(valIDs, data_path, mode='val')
    valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)


    # Training loop
    num_epochs = 200

    model = ResNet34_FT(pretrained=True, num_classes=num_classes, path = '/vol/lian/FairFace_code/FairFace/res34_fair_align_multi_4_20190809.pt', mode='prediction')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    main()