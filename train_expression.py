from utils import * # type: ignore
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
parser.add_argument('--dataset', required=True, default="RAF", type=str)
parser.add_argument('--model', required=True, default="R18", type=str)
parser.add_argument('--batchsize', required=True, default=8, type=int)
parser.add_argument('--lr', required=True, default=2e-4, type=float)
parser.add_argument('--momentum', required=True, default=0.9, type=float)
parser.add_argument('--wd', required=True, default=5e-4, type=float)
parser.add_argument('--early_stop', required=True, default=5, type=int)
args = parser.parse_args()

def train(epoch, trainloader):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(tqdm(trainloader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs,target)
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

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss = running_loss + loss.item()
            _, prediction = torch.max(outputs.data, 1)

            # Directly append tensors to lists
            all_predictions.append(prediction.cpu())
            all_targets.append(target.cpu())

        # Concatenate tensors and convert to numpy arrays
        all_targets = torch.cat(all_targets).numpy()
        all_predictions = torch.cat(all_predictions).numpy()

        # Calculate metrics for each task
        acc, acc_per_class = calculate_acc(all_targets, all_predictions)

    # Print metrics for each task
    print(f'Validation - Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(valloader)}')
    print(f'Average Accuracy: {acc * 100:.2f}%')

     # Print per-class accuracy with emotion labels
    for idx, accuracy in enumerate(acc_per_class):
        emotion = expression_dict[idx]  # Assume all indices will be covered
        print(f'Class {idx} ({emotion}): {accuracy * 100:.2f}%')

    return running_loss / len(valloader), acc, acc_per_class


def main():

    writer = SummaryWriter('runs/{}_{}_{}_{}_{}_{}'.format(args.dataset, args.model, args.batchsize, args.lr, args.momentum, args.wd))
    best_val_acc = 0
    epochs_no_improve = 0
    early_stop_epochs = args.early_stop

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
            # Define the save path with directory
            save_dir = './checkpoints'
            save_path = os.path.join(save_dir, 'best_model_{}_{}_{}_{}_{}_{}.pth'.format(
                args.dataset, args.model, args.batchsize, args.lr, args.momentum, args.wd
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

    if args.dataset == "RAF":

        data_path = '/vol/lian/datasets/basic/Image/aligned/'
        train_list, val_list = RAF_spliting(data_path)
        trainset = RAFDataset(train_list, data_path, mode='train')
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
        valset = RAFDataset(val_list, data_path, mode='val')
        valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
        expression_dict = {0: "Surprise", 1: "Fear", 2: "Disgust", 3: "Happiness", 4: "Sadness", 5: "Anger", 6: "Neutral"}

    else:

        data_path = '/vol/lian/datasets/AffectNet8/'
        train_dict, val_dict = Aff_spliting(data_path)
        trainset = AffDataset(train_dict, mode='train')
        trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
        valset = AffDataset(val_dict, mode='val')
        valloader = DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
        expression_dict = {0: "Neutral", 1: "Happiness", 2: "Sadness", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger"}


    models = {"R18":resnet18(pretrained=True, num_classes = 7), "R34":resnet34(pretrained=True, num_classes = 7),
                "R50":resnet50(pretrained=True, num_classes = 7), "R101":resnet101(pretrained=True, num_classes = 7),
                "Eb0":efficientnet_b0(pretrained=True, num_classes = 7), "Eb2":efficientnet_b2(pretrained=True, num_classes = 7), 
                "ViT":ViT(image_size=224, patch_size=16, num_classes=7, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1),
                "Swin_B":resnet18(pretrained=True, num_classes = 7)}

    # Training loop
    num_epochs = 200

    # Calculate the number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(trainable_params)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    criterion = nn.CrossEntropyLoss()
    main()