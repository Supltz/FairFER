import json
import random
import argparse
from collections import defaultdict
from itertools import combinations
from tqdm import tqdm
import torch
import torch.nn.functional as F

def load_data(file_path):
    with open(file_path, 'r') as file:
        return [dp for dp in json.load(file)]

def cosine_similarity_tensors(x, y):
    return torch.mm(F.normalize(x, p=2, dim=1), F.normalize(y, p=2, dim=1).t())

def group_embeddings_by_labels(data, embeddings_key, label_key):
    grouped = defaultdict(list)
    for entry in data:
        embeddings = torch.tensor(entry[embeddings_key], dtype=torch.float).to(device)  # Move to GPU
        grouped[entry[label_key]].append(embeddings)
    return {label: torch.stack(embeds) for label, embeds in grouped.items()}


def calculate_scores_and_differential(emotion_groups, attribute_groups, device='cuda'):
    emotion_labels = sorted(emotion_groups.keys())
    attribute_labels = sorted(attribute_groups.keys())

    num_emotions = len(emotion_labels)
    num_attributes = len(attribute_labels)
    cos_sim_matrix = torch.zeros((num_emotions, num_attributes), device=device)

    for i, emotion_label in enumerate(emotion_labels):
        for j, attribute_label in enumerate(attribute_labels):
            emotion_embeddings = emotion_groups[emotion_label].to(device)
            attribute_embeddings = attribute_groups[attribute_label].to(device)
            cos_sim_matrix[i, j] = cosine_similarity_tensors(emotion_embeddings, attribute_embeddings).mean(dim=1).sum()

    diff_scores = cos_sim_matrix.unsqueeze(2) - cos_sim_matrix.unsqueeze(1)
    upper_tri_mask = torch.triu(torch.ones(num_attributes, num_attributes, device=device), diagonal=1).bool()
    differential_scores_matrix = diff_scores[:, upper_tri_mask]

    pair_labels = [f"{attribute_labels[i]}-{attribute_labels[j]}" for i, j in combinations(range(num_attributes), 2)]

    return differential_scores_matrix.cpu(), cos_sim_matrix.cpu(), emotion_labels, pair_labels


def permute_attribute_labels(attr_groups, num_permutations=10000):
    # Extract all unique labels across all attribute groups
    all_labels = []
    for label, embeddings in attr_groups.items():
        all_labels.extend([label] * len(embeddings))

    unique_permutations = set()
    permutations = []

    with tqdm(total=num_permutations, desc="Generating unique attribute label permutations") as pbar:
        while len(unique_permutations) < num_permutations:
            # Shuffle the labels to generate a new permutation
            shuffled_labels = tuple(random.sample(all_labels, len(all_labels)))

            # Check if the permutation is unique
            if shuffled_labels not in unique_permutations:
                unique_permutations.add(shuffled_labels)

                # Append the unique permutation to the list
                permutations.append(shuffled_labels)
                pbar.update(1)

    return permutations


def apply_permuted_labels_to_dataset(permuted_labels, attr_data):
    # Initialize a structure to hold the newly permuted groups
    permuted_attr_groups = defaultdict(list)
    # Assume permuted_labels is a list of labels that has been shuffled
    # and is the same length as the number of data points in rafdb_data

    # Iterate through the dataset and assign each data point a new label from permuted_labels
    for dp, new_label in zip(attr_data, permuted_labels):
        # Fetch the embedding
        embedding = torch.tensor(dp['{}_embeddings'.format(args.attribute)], dtype=torch.float).to(device)  # Move tensor to GPU
        # Append the embedding to the new group based on the permuted label
        permuted_attr_groups[new_label].append(embedding)

    # Convert lists to tensors for each group
    for label in permuted_attr_groups:
        permuted_attr_groups[label] = torch.stack(permuted_attr_groups[label]) # type: ignore

    return permuted_attr_groups
def calculate_p_values(original_diff_scores, permuted_diff_scores_list):
    """
    Calculate p-values for each emotion and each pair of attributes, considering the direction of differences.

    Args:
    - original_diff_scores (Tensor): The original differential scores, with shape [num_emotions, num_pairs].
    - permuted_diff_scores_list (list): A list of tensors of permuted differential scores.

    Returns:
    - p_values_matrix (Tensor): A matrix of p-values with shape [num_emotions, num_pairs].
    """
    # Stack the permuted scores to form a tensor with shape [num_permutations, num_emotions, num_pairs]
    permuted_scores_tensor = torch.stack(permuted_diff_scores_list)

    # Calculate the absolute differences from the original to the permuted scores
    # This step is crucial if considering the magnitude of change rather than the direct comparison
    original_abs_diff = original_diff_scores.abs()
    permuted_abs_diff = permuted_scores_tensor.abs()

    # Count how many permuted absolute differences are greater or equal to the original absolute differences
    greater_or_equal_counts = (permuted_abs_diff > original_abs_diff.unsqueeze(0)).sum(dim=0).float()

    # Calculate p-values
    p_values_matrix = greater_or_equal_counts / permuted_scores_tensor.size(0)
    return p_values_matrix



def calculate_effect_size(differential_scores_matrix, cosine_sim_matrix, pairs , emotion_groups):
    N, P = differential_scores_matrix.shape

    # Initialize the effect sizes matrix
    effect_sizes = torch.zeros_like(differential_scores_matrix)

    if P == 1:

        for i in range(N):
            for j in range(P):
                A_i = int(pairs[j].split('-')[0])
                A_j = int(pairs[j].split('-')[1])
                effect_sizes[i, j] = (cosine_sim_matrix[i, A_i] - cosine_sim_matrix[i, A_j]) / (
                            len(emotion_groups[i]))


    else:
        for i in range(N):
            for j in range(P):
                A_i = int(pairs[j].split('-')[0])
                A_j = int(pairs[j].split('-')[1])
                effect_sizes[i,j] = (cosine_sim_matrix[i,A_i] - cosine_sim_matrix[i,A_j]) / (len(emotion_groups[i]) * differential_scores_matrix[i].std(dim=0, unbiased=True, keepdim=False))

    return effect_sizes


def print_differential_scores(matrix, row_labels, pair_labels, title):
    print(f"{title} Differential Association Scores:")
    print("Rows (Emotions):", row_labels)
    print("Columns (Group Pairs):", pair_labels)
    for row in matrix.cpu().numpy():
        print(row)
    print("\n")

def main(emo_dataset, attr_dataset, attribute, model):
    emo_data = load_data('./saved_embeddings/{}_emotion_{}.json'.format(emo_dataset,model))
    attr_data =  load_data('./saved_embeddings/{}_{}_{}.json'.format(attr_dataset,attribute, model))
    emotion_groups = group_embeddings_by_labels(emo_data, 'emotion_embeddings', 'emotion_label')
    attr_groups = group_embeddings_by_labels(attr_data, f'{attribute}_embeddings', f'{attribute}_label')

    original_diff_scores, cosine_sim_matrix, emotions, pairs = calculate_scores_and_differential(emotion_groups, attr_groups)
    print(f"Original Differential Scores for {attribute.capitalize()}:")
    print_differential_scores(original_diff_scores, emotions, pairs, attribute.capitalize())

    permuted_attr_groups_list = permute_attribute_labels(attr_groups, 10000)
    permuted_diff_scores_list = []



    # # Loop through each set of permuted labels
    for perm_labels in tqdm(permuted_attr_groups_list, desc="Processing permutations"):
        # Apply permuted labels to generate new emotion group mappings
        permuted_attr_groups = apply_permuted_labels_to_dataset(perm_labels, attr_data)

        # Recalculate association and differential scores using permuted emotion labels
        permuted_diff_scores, _, _, _ = calculate_scores_and_differential(emotion_groups, permuted_attr_groups)

        # Store the permuted differential scores for later analysis
        permuted_diff_scores_list.append(permuted_diff_scores)
    # Calculate p-values
    p_values = calculate_p_values(original_diff_scores, permuted_diff_scores_list)

    # Calculate the effect sizes for the differential scores
    d_effect_sizes = calculate_effect_size(original_diff_scores, cosine_sim_matrix, pairs, emotion_groups)

    # Open a text file to write the output
    with open(f"./effect_size/{attribute}_{emo_dataset}_{attr_dataset}_{args.model}_differential_scores_and_effects.txt", "w") as file:
        print("P-Values for Differential Scores:")
        file.write("P-Values for Differential Scores:\n")
        for i, emotion in enumerate(emotions):
            message = f"\nEmotion {emotion}:"
            print(message)
            file.write(message + "\n")
            for j, pair in enumerate(pairs):
                message = f"  {pair}: {p_values[i, j].item()}"
                print(message)
                file.write(message + "\n")

        print("\nEffect Sizes (d) for each attribute pair comparison:")
        file.write("\nEffect Sizes (d) for each attribute pair comparison:\n")
        for i, emotion in enumerate(emotions):
            message = f"\nEmotion {emotion}:"
            print(message)
            file.write(message + "\n")
            for j, pair_label in enumerate(pairs):
                message = f"  {pair_label}: {d_effect_sizes[i, j].item()}"
                print(message)
                file.write(message + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Differential Association Scores")
    parser.add_argument('--attribute', choices=['age', 'gender', 'race', 'emotion'], required=True)
    parser.add_argument('--emo_dataset', choices=['RAF', 'AffectNet'], required=True)
    parser.add_argument('--attr_dataset', choices=['Fairface', 'UTK'], required=True)
    parser.add_argument('--model', required=True, type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(args.emo_dataset,args.attr_dataset, args.attritbue, args.model)
