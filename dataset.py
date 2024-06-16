from PIL import Image
from torchvision import transforms
import torch
from utils import RAF_GetLabel, FER_GetLabel, UTK_GetLabel

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class CeleADataset(Dataset):
    def __init__(self, data, labels, path):
        self.data = data
        self.labels = labels
        self.path = path
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(240),
            # transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(),
            # transforms.Normalize((0.5355, 0.4249, 0.3801), (0.2832, 0.2578, 0.2548)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample_data = self.path + self.data[idx]
        sample_data = f"/vol/lian/datasets/CelebA_aligned/processed/img_align_celeba_aligned/frame_det_00_{self.data[idx].split('/')[-1].split('.')[0]}.bmp"
        image = Image.open(sample_data)
        image = self.transform_train(image)
        labels = torch.vstack((torch.tensor(self.labels[idx][0]), torch.tensor(self.labels[idx][1])))
        return image,labels


class RAFDataset(Dataset):
    def __init__(self, data, path, mode):
        self.data = data
        self.path = path
        self.mode = mode
        self.transform_train = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomCrop(224),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data = self.path + self.data[idx]
        image = Image.open(sample_data)
        if self.mode == 'train':
            image = self.transform_train(image)
            labels = RAF_GetLabel(self.data[idx])
            labels = torch.tensor((labels["emotion"], labels["gender"], labels["age"], labels["race"]))
            return image, labels
        elif self.mode == 'val':
            image = self.transform_val(image)
            labels = RAF_GetLabel(self.data[idx])
            labels = torch.tensor((labels["emotion"], labels["gender"], labels["age"], labels["race"]))
            return image,labels
        else:
            image = self.transform_val(image)
            labels = RAF_GetLabel(self.data[idx])
            labels = torch.tensor((labels["emotion"], labels["gender"], labels["age"], labels["race"]))
            return image, labels, sample_data

class AffDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self.image_list = list(self.data.keys())
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path)
        if self.mode == 'train':
            image = self.transform_train(image)
            label = self.data[image_path]
            return image, label
        elif self.mode == 'val':
            image = self.transform_val(image)
            label = self.data[image_path]
            return image,label
        else:
            image = self.transform_val(image)
            label = self.data[image_path]
            return image, label, image_path


class FairfaceLabelGetter:
    def __init__(self, path):
        # Load and cache the CSV data at initialization
        self.data = pd.read_csv(path)
        # No need to set 'index' as the index column here since we will lookup by number

    def get_label(self, item):
        # Extract the numeric part from the item string
        item_number = int(item.split('.')[0]) - 1

        # Use the number to access the row
        if item_number in self.data.index:
            row = self.data.loc[item_number]
            labels = {
                'age': row['age'],
                'gender': row['gender'],
                'race': row['race']
            }
        else:
            error_message = f"Error: Item number '{item_number}' not found in the dataset."
            raise ValueError(error_message)
        return labels

class FairfaceDataset(Dataset):
    def __init__(self, data, path, mode):
        self.data = data
        self.path = path
        self.mode = mode

        # Initialize the label getter with the path to the labels CSV file
        if self.mode != "embedding":
            self.label_getter = FairfaceLabelGetter(f"{self.path}fairface_label_{self.mode}.csv")

        # Define transforms for training and validation modes
        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Construct file path based on mode
        if self.mode != "embedding":
            file_name = self.data[idx]
            file_path = f"{self.path}{self.mode}/{file_name}"
            image = Image.open(file_path)

            # Apply the appropriate transformation based on mode
            image = self.transforms[self.mode](image)

            # Fetch labels for the current image using the optimized label getter
            # The item is the file name extracted from the data list
            labels = self.label_getter.get_label(file_name)
            labels_tensor = torch.tensor((labels["gender"], labels["age"], labels["race"]))

            return image, labels_tensor
        else:
            file_name = self.data[idx]
            file_path = f"{self.path}{file_name[1]}/{file_name[0]}"
            image = Image.open(file_path)

            # Apply the appropriate transformation based on mode
            image = self.transforms["val"](image)

            # Fetch labels for the current image using the optimized label getter
            # The item is the file name extracted from the data list
            label_getter = FairfaceLabelGetter(f"{self.path}fairface_label_{file_name[1]}.csv")
            labels = label_getter.get_label(file_name[0])
            labels_tensor = torch.tensor((labels["gender"], labels["age"], labels["race"]))

            return image, labels_tensor, file_path

class FERDataset(Dataset):
    def __init__(self, data, path, mode):
        self.data = data
        self.path = path
        self.mode = mode
        self.transform_train = transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.RandomCrop(224),
            transforms.RandomRotation((-15, 15)),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample_data = self.path + "FER2013Train/" + self.data[idx]
            image = Image.open(sample_data)
            image = self.transform_train(image)
            labels = FER_GetLabel(self.data[idx])
            return image, torch.tensor(labels)
        elif self.mode == 'val':
            sample_data = self.path + "FER2013Valid/" + self.data[idx]
            image = Image.open(sample_data)
            image = self.transform_val(image)
            labels = FER_GetLabel(self.data[idx])
            return image,torch.tensor(labels)
        elif self.mode == 'test':
            sample_data = self.path + "FER2013Test/" + self.data[idx]
            image = Image.open(sample_data)
            image = self.transform_val(image)
            labels = FER_GetLabel(self.data[idx])
            return image,torch.tensor(labels)
        else:

            # to be implemented
            sample_data = self.path + "FER2013Train/" + self.data[idx]
            image = Image.open(sample_data)
            image = self.transform_val(image)
            labels = RAF_GetLabel(self.data[idx])
            labels = torch.tensor((labels["emotion"], labels["gender"], labels["age"], labels["race"]))
            return image, labels, sample_data


class UTKDataset(Dataset):
    def __init__(self, data, path, mode):
        self.data = data
        self.path = path
        self.mode = mode

        # Define transforms for training and validation modes
        self.transforms = {
            'train': transforms.Compose([
                transforms.Resize((240, 240)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Construct file path based on mode
        if self.mode != 'embedding':
            sample_data = self.path + self.data[idx]
            image = Image.open(sample_data)

            # Apply the appropriate transformation based on mode
            image = self.transforms[self.mode](image)

            labels = UTK_GetLabel(self.data[idx])

            return image, labels
        else:
            sample_data = self.path + self.data[idx]
            image = Image.open(sample_data)

            # Apply the appropriate transformation based on mode
            image = self.transforms["val"](image)

            labels = UTK_GetLabel(self.data[idx])

            return image, labels, sample_data



class ExpDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self.image_list = list(self.data.keys())
        self.transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path)
        if self.mode == 'train':
            image = self.transform_train(image)
            label = self.data[image_path]
            return image, label
        elif self.mode == 'val':
            image = self.transform_val(image)
            label = self.data[image_path]
            return image,label
        else:
            image = self.transform_val(image)
            label = self.data[image_path]
            return image, label, image_path