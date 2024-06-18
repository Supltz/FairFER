import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import os
import random
import torch
import os




def calculate_acc(true_labels, predicted_labels):
    unique_labels = np.unique(true_labels)
    classification_type = "binary" if len(unique_labels) == 2 else "multi-class"

    # Calculate overall accuracy
    overall_acc = accuracy_score(true_labels, predicted_labels)

    if classification_type == "binary":
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        acc_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate for class 0
        acc_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate for class 1
        return overall_acc, np.array([acc_class_0, acc_class_1])
    else:
        num_classes = len(unique_labels)
        per_class_acc = np.zeros(num_classes)

        for cls in range(num_classes):
            cls_mask = (true_labels == cls)
            cls_acc = np.sum(predicted_labels[cls_mask] == true_labels[cls_mask]) / np.sum(cls_mask)
            per_class_acc[cls] = cls_acc

        return overall_acc, per_class_acc


def RAF_spliting(folder_path):
    # Lists to store file names
    train_files = []
    test_files = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if "train" in filename:
            train_files.append(filename)
        elif "test" in filename:
            test_files.append(filename)

    return train_files, test_files


def Fairface_spliting(folder_path):
    # Lists to store file names
    train_files = []
    val_files = []

    train_path = os.path.join(folder_path, 'train')
    val_path = os.path.join(folder_path, 'val')

    # Iterate over all files in the folder
    for filename in os.listdir(train_path):
        train_files.append(filename)

    for filename in os.listdir(val_path):
        val_files.append(filename)
    return train_files, val_files

def RAF_GetLabel(item):

    identifier = "_".join(item.split("_")[:2])
    emo_path = "/vol/lian/datasets/basic/EmoLabel/list_patition_label.txt"
    attr_path = "/vol/lian/datasets/basic/Annotation/manual/{}_manu_attri.txt".format(identifier)
    with open(emo_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            img,label = line.split(" ")
            if identifier in img:
                emo = int(label) - 1
                break

    with open(attr_path,'r') as f:
        lines = f.readlines()
        lines = lines[5:]
        gender = int(lines[0].strip())
        race = int(lines[1].strip())
        age = int(lines[2].strip())

    # Create a dictionary with the labels
    labels = {
        'emotion': emo,
        'gender': gender,
        'race': race,
        'age': age
    }

    # if labels["emotion"] == 3:
    #     print(item)

    return labels



def Aff_spliting(dir):
    train_dict = {}
    test_dict = {}
    train_path = os.path.join(dir, 'train')
    test_path = os.path.join(dir, 'test')

    #train set

    for label in range(7):
        label_path = os.path.join(train_path, str(label))
        if os.path.exists(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                train_dict[img_path] = label
    #test set

    for label in range(7):
        label_path = os.path.join(test_path, str(label))
        if os.path.exists(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                test_dict[img_path] = label


    return train_dict,test_dict



def UTK_spliting(path):
    random.seed(3097)
    all_images = []
    for filename in os.listdir(path):
        # Validate and parse the filename format
        parts = filename.split('_')
        if len(parts) >= 4 and parts[0].isdigit() and parts[1] in ['0', '1'] and parts[2] in ['0', '1', '2', '3'] and parts[2] != '4':
            age = int(parts[0])
            if 0 <= age <= 116:
                # If the filename fits the criteria, add it to the list
                all_images.append(filename)
    random.shuffle(all_images)

    # Calculate the split index for a 4:1 ratio
    split_index = int(len(all_images) * 0.8)  # 80% for training, 20% for validation

    # Split the list
    train_images = all_images[:split_index]
    validation_images = all_images[split_index:]

    return train_images, validation_images


def UTK_GetLabel(item):
    # Parse the filename
    parts = item.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])

    if age >= 0 and age <= 3:
        age_label = 0
    elif age >= 4 and age <= 19:
        age_label = 1
    elif age >= 20 and age <= 39:
        age_label = 2
    elif age >= 40 and age <= 69:
        age_label = 3
    elif age >= 70:
        age_label = 4
    else:
        raise ValueError("Age is out of the expected range")

    # Convert labels to tensor
    labels_tensor = torch.tensor((gender, age_label, race))

    return labels_tensor