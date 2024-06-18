import argparse
import json
import os
from utils import *
from dataset import *
from FT_R34 import *
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import DataLoader

# Define the argument parser
parser = argparse.ArgumentParser(description='Compile embeddings and predictions for various datasets using pretrained models.')
parser.add_argument('--dataset', required=True, choices=['RAF', 'AffectNet', 'ExpW', 'Fairface', 'UTK'], help='Dataset to process.')
parser.add_argument('--model_attr', required=True, choices=['age', 'gender', 'race'], help='Model attribute to evaluate.')
parser.add_argument('--embedding_mode', required=True, type=str, choices=['embedding'], help='Mode for the dataset processing.')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for DataLoader.')
parser.add_argument('--device', default='cuda:0', type=str, help='Device to run the models on.')
parser.add_argument('--output_file', default='output.json', type=str, help='Output JSON file to save the results.')

args = parser.parse_args()

# Function to load the dataset
def load_dataset(dataset_name, mode):
    if dataset_name == 'RAF':
        data_path = '/vol/lian/datasets/basic/Image/aligned/'
        train_list, val_list = RAF_spliting(data_path, 'emotion')
        data_list = train_list + val_list
        dataset = RAFDataset(data_list, data_path, mode)
    elif dataset_name == 'AffectNet':
        data_path = '/vol/lian/datasets/AffectNet8/'
        train_dict, val_dict = Aff_spliting(data_path)
        data_dict = {**train_dict, **val_dict}
        dataset = AffDataset(data_dict, mode)
    elif dataset_name == 'ExpW':
        data_path = '/vol/lian/datasets/ExpW_new/'
        train_dict, val_dict = Exp_spliting(data_path)
        data_dict = {**train_dict, **val_dict}
        dataset = ExpDataset(data_dict, mode)
    elif dataset_name == 'Fairface':
        data_path = '/vol/lian/datasets/Fairface/'
        trainIDs, valIDs = Fairface_spliting(data_path)
        data_tuples = [(filename, 'train') for filename in trainIDs] + [(filename, 'val') for filename in valIDs]
        dataset = FairfaceDataset(data_tuples, data_path, mode)
    elif dataset_name == 'UTK':
        data_path = '/vol/lian/datasets/utkcropped/'
        trainIDs, valIDs = UTK_spliting(data_path)
        data_list = trainIDs + valIDs
        dataset = UTKDataset(data_list, data_path, mode)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset

# Function to load the model
def load_model(attribute, device):
    model_paths = {
        'age': '/vol/lian/Fairness/checkpoints/best_model_age_Fair_age_0.0002.pth',
        'gender': '/vol/lian/Fairness/checkpoints/best_model_gender_gender_classifier.pth',
        'race': '/vol/lian/Fairness/checkpoints/best_model_race_Fair_race.pth'
    }

    num_classes_dict = {
        'age': 9,
        'gender': 2,
        'race': 7
    }

    model = ResNet34_FT(pretrained=True, num_classes=num_classes_dict[attribute], path=model_paths[attribute], mode='embedding')
    model.to(device)
    return model

# Function to compile data for Fairface dataset
def compile_fairface(data_loader, age_model, gender_model, race_model, device):
    age_model.eval()
    gender_model.eval()
    race_model.eval()

    fairface_data = []
    age_label_distribution = Counter()
    gender_label_distribution = Counter()
    race_label_distribution = Counter()

    for images, labels, paths in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        age_emb, age_output = age_model(images)
        gender_emb, gender_output = gender_model(images)
        race_emb, race_output = race_model(images)

        _, age_prediction = torch.max(age_output.data, 1)
        _, gender_prediction = torch.max(gender_output.data, 1)
        _, race_prediction = torch.max(race_output.data, 1)

        data_entry = {
            "image_path": paths[0],
            "age_embeddings": age_emb[0].cpu().detach().numpy().tolist(),
            "age_prediction": age_prediction.item(),
            "age_label": labels[:, 1].item(),
            "gender_embeddings": gender_emb[0].cpu().detach().numpy().tolist(),
            "gender_prediction": gender_prediction.item(),
            "gender_label": labels[:, 0].item(),
            "race_embeddings": race_emb[0].cpu().detach().numpy().tolist(),
            "race_prediction": race_prediction.item(),
            "race_label": labels[:, 2].item()
        }
        fairface_data.append(data_entry)
        age_label_distribution.update([labels[:, 1].item()])
        gender_label_distribution.update([labels[:, 0].item()])
        race_label_distribution.update([labels[:, 2].item()])

    print("Distribution of age labels for correctly classified instances:")
    for label, count in age_label_distribution.items():
        print(f"Label {label}: {count}")
    print("Distribution of gender labels for correctly classified instances:")
    for label, count in gender_label_distribution.items():
        print(f"Label {label}: {count}")
    print("Distribution of race labels for correctly classified instances:")
    for label, count in race_label_distribution.items():
        print(f"Label {label}: {count}")

    return fairface_data

# Main script execution
if __name__ == "__main__":
    # Load the selected dataset
    dataset = load_dataset(args.dataset, args.embedding_mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Load the models for the specified attributes
    model = load_model(args.model_attr, args.device)

    # Compile data based on the dataset selected
    if args.dataset == 'Fairface':
        # Load additional models for multi-attribute datasets like Fairface
        age_model = load_model('age', args.device)
        gender_model = load_model('gender', args.device)
        race_model = load_model('race', args.device)

        compiled_data = compile_fairface(dataloader, age_model, gender_model, race_model, args.device)
    else:
        raise ValueError(f"Compilation for dataset {args.dataset} is not implemented.")

    # Save the compiled data to a JSON file
    with open(args.output_file, 'w') as f:
        json.dump(compiled_data, f)

    print(f"Data saved to {args.output_file}")