import argparse
import json
import os
from utils import *
from dataset import *
from models.EfficientNets import *
from models.swin_transformer import *
from vit_pytorch import ViT
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from models.resnets import *

# Define the argument parser
parser = argparse.ArgumentParser(description='Compile embeddings for various datasets using pretrained models.')
parser.add_argument('--dataset', required=True, choices=['RAF', 'AffectNet', 'Fairface', 'UTK'], help='Dataset to process.')
parser.add_argument('--model_attr', required=True, choices=['age', 'gender', 'race', 'emotion'], help='Model attribute to evaluate.')
parser.add_argument('--model', default='R34', type=str)

args = parser.parse_args()

# Function to load the dataset
def load_dataset(dataset_name, mode):
    if dataset_name == 'RAF':
        data_path = '/vol/lian/datasets/basic/Image/aligned/'
        _, val_list = RAF_spliting(data_path)
        dataset = RAFDataset(val_list, data_path, mode)
    elif dataset_name == 'AffectNet':
        data_path = '/vol/lian/datasets/AffectNet8/'
        _, val_dict = Aff_spliting(data_path)
        dataset = AffDataset(val_dict, mode)
    elif dataset_name == 'Fairface':
        data_path = '/vol/lian/datasets/Fairface/'
        _, valIDs = Fairface_spliting(data_path)
        val_tuples = [(filename, 'val') for filename in valIDs]
        dataset = FairfaceDataset(val_tuples, data_path, mode)
    elif dataset_name == 'UTK':
        data_path = '/vol/lian/datasets/utkcropped/'
        _, valIDs = UTK_spliting(data_path)
        dataset = UTKDataset(valIDs, data_path, mode)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset

# Function to load the model
def load_model(dataset, attribute, model_name):
    if attribute == 'emotion':
        num_classes = 7
    elif attribute == 'gender':
        num_classes = 2
    elif attribute == 'age':
        if dataset == "Fairface":
            num_classes = 9
        else:
            num_classes = 5
    elif attribute == 'race':
        if dataset == 'Fairface':
            num_classes = 7
        else:
            num_classes = 4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    models = {"R18":resnet18(pretrained=False, num_classes = num_classes), "R34":resnet34(pretrained=False, num_classes = num_classes),
              "R50":resnet50(pretrained=False, num_classes = num_classes), "R101":resnet101(pretrained=False, num_classes = num_classes),
              "Eb0":efficientnet_b0(pretrained=False, num_classes = num_classes), "Eb2":efficientnet_b2(pretrained=False, num_classes = num_classes),
              "ViT":ViT(image_size=224, patch_size=16, num_classes=num_classes, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1),
              "Swin_B":resnet18(pretrained=False, num_classes = num_classes)}
    model = models[model_name] 
    model = nn.DataParallel(model, device_ids=[0, 1, 2])
    model.to(device)
    
     

    directory = './checkpoints'
    
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter files that contain both dataset and attribute in their names
    matching_files = [file for file in files if dataset in file and attribute in file and model_name in file and file.endswith('.pth')]
    
    if not matching_files:
        raise FileNotFoundError(f"No .pth file found containing both '{dataset}' and '{attribute}' in the filename.")
    
    
    # Full path to the model file
    model_path = os.path.join(directory, matching_files[0])

    model_dict = model.state_dict()
    pretrained = torch.load(model_path)
    pretrained = {k.replace('module.', ''): v for k, v in pretrained.items()}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    return model

# Function to compile data for Fairface dataset
def compile_embedd(attribute, data_loader, model):
    model.eval()
    device = "cuda:0"

    data = []

    for images, labels, paths in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        if attribute == "emotion":
            labels = labels.items()
        elif attribute == "gender":
            labels = labels[:,0].items()
        elif attribute == "age":
            labels = labels[:,1].items()
        elif attribute == "race":
            labels = labels[:,2].items()

        emb, _ = model(images)

        data_entry = {
            "image_path": paths[0],
            "{}_embeddings".format(attribute): emb[0].cpu().detach().numpy().tolist(),
            "{}_label".format(attribute): labels
        }
        data.append(data_entry)

    return data

# Main script execution
if __name__ == "__main__":
    # Load the selected dataset
    dataset = load_dataset(args.dataset, 'embedding')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    # Load the models for the specified attributes
    model = load_model(args.dataset, args.model_attr, args.model)

    compiled_data = compile_embedd(args.attribute, dataloader, args.model)
    # Save the compiled data to a JSON file
    with open('./saved_embeddings/{}_{}_{}'.format(args.dataset,args.attribute,args.model), 'w') as f:
        json.dump(compiled_data, f)