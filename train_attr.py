from utils import *
import numpy as np
from models.resnets import *
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter # type: ignore
import argparse
from dataset import *
from models.EfficientNets import *
from models.swin_transformer import *
from vit_pytorch import ViT




parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--dataset', required=True, default="Fair", type=str)
parser.add_argument('--model', required=True, default="UTK", type=str)
parser.add_argument('--batchsize', required=True, default=8, type=int)
parser.add_argument('--lr', required=True, default=2e-4, type=float)
parser.add_argument('--momentum', required=True, default=0.9, type=float)
parser.add_argument('--wd', required=True, default=5e-4, type=float)
parser.add_argument('--early_stop', required=True, default=5, type=int)
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

    writer = SummaryWriter('runs/{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.attr, args.batchsize, args.lr, args.momentum, args.wd))
    best_val_accuracy = 0
    epochs_no_improve = 0
    early_stop_epochs = args.early_stop

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
            # Define the save path with directory
            save_dir = './checkpoints'
            save_path = os.path.join(save_dir, 'best_model_{}_{}_{}_{}_{}_{}_{}.pth'.format(
                args.dataset, args.model, args.attr, args.batchsize, args.lr, args.momentum, args.wd
            ))

            # Check if the directory exists, if not, create it
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f'Best model saved to {save_path}')

        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == early_stop_epochs:
            print("Early stopping triggered")
            break
    writer.close()  # Close the TensorBoard writer

if __name__=="__main__":

    if args.dataset == "Fair":

        data_path = '/vol/lian/datasets/Fairface/'
        trainIDs, valIDs = Fairface_spliting(data_path)
        trainset = FairfaceDataset(trainIDs, data_path, mode='train')
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

        valset = FairfaceDataset(valIDs, data_path, mode='val')
        valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
        num_classes_dict = {
            'age': 9,
            'gender': 2,
            'race': 7
        }


    else:

        data_path = '/vol/lian/datasets/utkcropped/'
        trainIDs, valIDs = UTK_spliting(data_path)
        trainset = UTKDataset(trainIDs, data_path, mode='train')
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
        valset = UTKDataset(valIDs, data_path, mode='val')
        valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
        num_classes_dict = {
            'age': 5,
            'gender': 2,
            'race': 4
        }


    models = {"R18":resnet18(pretrained=True, num_classes = num_classes_dict[args.attr]), "R34":resnet34(pretrained=True, num_classes = num_classes_dict[args.attr]),
                "R50":resnet50(pretrained=True, num_classes = num_classes_dict[args.attr]), "R101":resnet101(pretrained=True, num_classes = num_classes_dict[args.attr]),
                "Eb0":efficientnet_b0(pretrained=True, num_classes = num_classes_dict[args.attr]), "Eb2":efficientnet_b2(pretrained=True, num_classes = num_classes_dict[args.attr]), 
                "ViT":ViT(image_size=224, patch_size=16, num_classes=num_classes_dict[args.attr], dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1),
                "Swin_B":resnet18(pretrained=True, num_classes = num_classes_dict[args.attr])}


    # Training loop
    num_epochs = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)

    #optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    main()