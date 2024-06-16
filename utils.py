import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from collections import Counter

import re



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



def calculate_f1(true_labels, predicted_labels):
    unique_labels = np.unique(true_labels)
    classification_type = "binary" if len(unique_labels) == 2 else "multi-class"

    if classification_type == "binary":
        f1 = f1_score(true_labels, predicted_labels, average='binary')
        return f1, np.array([f1])  # Same value for both per-class and average
    else:
        if len(np.unique(true_labels)) < len(unique_labels) or len(np.unique(predicted_labels)) < len(unique_labels):
            print(
                "Warning: F1 score per class may not be meaningful if some classes are missing in true labels or predictions.")
        f1_per_class = f1_score(true_labels, predicted_labels, average=None)
        f1_avg = np.mean(f1_per_class)
        return f1_avg, f1_per_class


def random_sample_from_categories(sampled_dict, ratios):
    sampled_data = {}
    for category, df in sampled_dict.items():
        n_samples = int(ratios[category])  # Get the number of samples for this category
        sampled_data[category] = df.sample(n_samples, random_state=42)  # Random sampling
    # Concatenate the sampled data from all categories into a single DataFrame
    concatenated_df = pd.concat(sampled_data.values(), ignore_index=True)
    return concatenated_df

def read_txt_to_df(filepath):
    # Open the text file
    with open(filepath, 'r') as file:
        # Read lines from the file
        lines = file.readlines()

    # Get the headings from the second line, split by space
    headings = lines[1].strip().split()

    # Find the index of each heading in the headings list
    smiling_index = headings.index('Smiling')
    male_index = headings.index('Male')
    young_index = headings.index('Young')

    # Create a dictionary to hold the data
    data_dict = {
        "ImageIndex": [],
        "Smiling": [],
        "Male": [],
        "Young": []
    }
    # Process the content lines (third line onwards)
    for line in lines[2:]:
        # Split the line by space to get the values
        values = line.strip().split()
        # Append the values to the appropriate lists in the dictionary
        data_dict["ImageIndex"].append(values[0])
        data_dict["Smiling"].append(int(values[smiling_index + 1]))  # +1 to account for the ImageIndex column
        data_dict["Male"].append(int(values[male_index + 1]))  # +1 to account for the ImageIndex column
        data_dict["Young"].append(int(values[young_index + 1]))  # +1 to account for the ImageIndex column
    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)
    return df

def random_sample_per_category(df):
    # Create an empty DataFrame to hold the samples
    sampled_dict = {}

    # Categories
    categories = {
        'Male_1_Smiling_1': (1, 1),
        'Male_1_Smiling_-1': (1, -1),
        'Male_-1_Smiling_1': (-1, 1),
        'Male_-1_Smiling_-1': (-1, -1)
    }

    for category, (male_val, smiling_val) in categories.items():
        # Filter the DataFrame for each category
        category_df = df[(df['Male'] == male_val) & (df['Smiling'] == smiling_val)]
        # Store the sampled DataFrame in the sampled_dict dictionary
        sampled_dict[category] = category_df

    return sampled_dict


def CelebA_spliting(split_path):
    # Initialize empty lists to store image IDs for each set
    training_set_ids = []
    validation_set_ids = []
    test_set_ids = []

    # Open the file and read its contents
    with open(split_path, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into image ID and set indicator
            image_id, set_indicator = line.strip().split()
            # Assign the image ID to the appropriate set based on the set indicator
            if set_indicator == '0':
                training_set_ids.append(image_id)
            elif set_indicator == '1':
                validation_set_ids.append(image_id)
            elif set_indicator == '2':
                test_set_ids.append(image_id)
    return training_set_ids, validation_set_ids, test_set_ids



def form_loaders(dataset, trainIDs = None, train = True, valIDs = None, df = None):

    if dataset == "CelebA":
        n_samples = 68140

        train_df = df[df['ImageIndex'].isin(trainIDs)]

        # Get the sampled DataFrame
        sampled_df = random_sample_per_category(train_df)

        ratios = {
            "balanced": {category: 0.25 * n_samples for category in sampled_df.keys()},
            "gender_imb": {category: 0.1 * n_samples if "Male_1" in category else 0.4 * n_samples for category in
                           sampled_df.keys()},
            "class_imb": {category: 0.4 * n_samples if "Smiling_1" in category else 0.1 * n_samples for category in
                          sampled_df.keys()},
            "imbalanced": {category: 0.7 * n_samples if "Male_-1_Smiling_1" in category else 0.1 * n_samples for category in
                           sampled_df.keys()},
        }

        return ratios, sampled_df

    elif dataset == "RAF-DB":
        image_label_pairs = {}

        print("Process the training files...\n")

        if train:
            for file in tqdm(trainIDs):
                labels = RAF_GetLabel(file)
                image_label_pairs[file] = labels
            # Filter out entries with gender label 2
            image_label_pairs = {k: v for k, v in image_label_pairs.items() if v['gender'] !=2}

            return RAF_Sampling(image_label_pairs)
        else:
            for file in tqdm(valIDs):
                labels = RAF_GetLabel(file)
                image_label_pairs[file] = labels
            # Filter out entries with gender label 2
            image_label_pairs = {k: v for k, v in image_label_pairs.items() if v['gender'] !=2}

            return RAF_Sampling(image_label_pairs)

def RAF_Sampling(image_label_pairs):
    # Initialize a dictionary to count gender distribution in each emotion category
    emotion_gender_distribution = {i: {'male': 0, 'female': 0} for i in range(7)}

    # Count the distribution
    for labels in image_label_pairs.values():
        emotion = labels['emotion']
        gender = labels['gender']
        if gender == 0:
            gender = 'male'
        elif gender == 1:
            gender = 'female'
        emotion_gender_distribution[emotion][gender] += 1

    emotion_samples = {}
    for emotion_idx in range(7):
        male_count = emotion_gender_distribution[emotion_idx]['male']
        female_count = emotion_gender_distribution[emotion_idx]['female']

        # Calculate N for both scenarios
        N_balanced = 2 * min(male_count, female_count)
        N_ratio = 5 * min(male_count, female_count // 4)
        N = min(N_balanced, N_ratio)

        while N % 10 != 0:
            N -= 1

            # Adjust N if it exceeds available images
        while (4 * N) // 5 > female_count or N // 5 > male_count:
            N -= 10

        male_images = [img for img, labels in image_label_pairs.items() if
                       labels['emotion'] == emotion_idx and labels['gender'] == 0]
        female_images = [img for img, labels in image_label_pairs.items() if
                         labels['emotion'] == emotion_idx and labels['gender'] == 1]

        # Scenario 1: Balanced male and female
        balanced_sample = random.sample(male_images, N // 2) + random.sample(female_images, N // 2)

        # Scenario 2: Female-to-male ratio 4:1
        female_sample = random.sample(female_images, (4 * N) // 5)
        male_sample = random.sample(male_images, N // 5)
        ratio_sample = female_sample + male_sample

        # Get image samples
        emotion_samples[emotion_idx] = {'balanced': balanced_sample, 'gender_imb': ratio_sample}
    all_balanced_samples = []
    all_gender_imb_samples = []

    for emotion_idx, samples in emotion_samples.items():
        all_balanced_samples.extend(samples['balanced'])
        all_gender_imb_samples.extend(samples['gender_imb'])

    return all_balanced_samples, all_gender_imb_samples


def calculate_metrics(targets, predictions):
    cm = confusion_matrix(targets, predictions)

    # Check if confusion matrix is 1x1
    if cm.size == 1:
        print(
            "Warning: True Positive Rate (TPR) and False Positive Rate (FPR) cannot be calculated with a 1x1 confusion matrix.")
        return None, None
    else:
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Protect against division by zero
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Protect against division by zero
        return tpr, fpr

def RAF_spliting(folder_path, mode):
    # Lists to store file names
    train_files = []
    test_files = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if "train" in filename:
            train_files.append(filename)
        elif "test" in filename:
            test_files.append(filename)

    if mode != 'emotion':
        train_files = [f for f in tqdm(train_files) if RAF_GetLabel(f)["gender"] <= 1]
        test_files = [f for f in tqdm(test_files) if RAF_GetLabel(f)["gender"] <= 1]

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




def FER_spliting(folder_path):
    # Lists to store file names
    train_files = []
    val_files = []
    test_files = []

    train_path = os.path.join(folder_path, 'FER2013Train')
    val_path = os.path.join(folder_path, 'FER2013Valid')
    test_path = os.path.join(folder_path, 'FER2013Test')

    # Iterate over all files in the folder
    for filename in os.listdir(train_path):
        train_files.append(filename)

    for filename in os.listdir(val_path):
        val_files.append(filename)

    for filename in os.listdir(test_path):
        test_files.append(filename)

    return train_files, val_files, test_files


def FER_GetLabel(item):
    path = '/vol/lian/datasets/FER+/fer2013new.csv'
    # Make sure the CSV is read without setting an index column so we can access rows by integer location
    data = pd.read_csv(path, header=None)

    # Regular expression to match the numerical part of the filename
    pattern = r"fer(\d+)\.png"
    match = re.search(pattern, item)
    if match:
        index = int(match.group(1))
    else:
        raise ValueError(f"Error: The filename '{item}' does not match the expected format.")

    # Extract emotions scores from the row
    emotions = data.iloc[index + 1, 2:9].astype(float)  # Ensure conversion to float for numerical operations

    # Find the index of the largest number in the emotions list
    label = emotions.idxmax() - 2

    return label

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

def Exp_spliting(path):
    random.seed(3097)
    all_images = {}
    for label in range(7):
        label_path = os.path.join(path, str(label))
        if os.path.exists(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                all_images[img_path] = label

    # Convert dictionary keys to a list and shuffle it
    image_keys = list(all_images.keys())

    random.shuffle(image_keys)

    # Calculate the split index for a ratio
    split_index = int(len(image_keys) * 0.8)  # 80% for training, 20% for validation

    # Split the keys list
    train_keys = image_keys[:split_index]
    validation_keys = image_keys[split_index:]

    # Create dictionaries for training and validation sets
    train_images = {key: all_images[key] for key in train_keys}
    validation_images = {key: all_images[key] for key in validation_keys}

    return train_images, validation_images

if __name__=="__main__":
    folder_path = "/vol/lian/datasets/basic/Image/aligned/"
    mode = "m"
    train,test = RAF_spliting(folder_path,mode)
    all = train + test
    labels = []

    for i in tqdm(all):
        labels.append(RAF_GetLabel(i))

    # Initialize dictionaries for counting
    emotion_age_counts = {}
    emotion_race_counts = {}

    # Count combinations
    for item in tqdm(labels):
        emotion = item['emotion']
        age = item['age']
        race = item['race']

        # Count emotion-age combinations
        emotion_age_key = (emotion, age)
        emotion_age_counts[emotion_age_key] = emotion_age_counts.get(emotion_age_key, 0) + 1

        # Count emotion-race combinations
        emotion_race_key = (emotion, race)
        emotion_race_counts[emotion_race_key] = emotion_race_counts.get(emotion_race_key, 0) + 1

    print(labels)