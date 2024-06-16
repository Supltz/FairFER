import json
from utils import *
from dataset import *
from FT_R34 import *
from tqdm import tqdm
from collections import Counter
#
#
# Load datasets
# aff_path = '/vol/lian/datasets/AffectNet8/'
# train_dict, val_dict = Aff_spliting(aff_path)
#
# aff_dict = {**train_dict, **val_dict}
#
# aff_set = AffDataset(aff_dict, mode='embedding')
# affloader = DataLoader(aff_set, batch_size=1, shuffle=True, num_workers=2)

# Exp_path = '/vol/lian/datasets/ExpW_new/'
# train_dict, val_dict = Exp_spliting(Exp_path)
#
# Exp_dict = {**train_dict, **val_dict}
#
# Exp_set = ExpDataset(Exp_dict, mode='embedding')
# Exploader = DataLoader(Exp_set, batch_size=1, shuffle=True, num_workers=2)

# Load datasets
# raf_path = '/vol/lian/datasets/basic/Image/aligned/'
# train_list, val_list = RAF_spliting(raf_path,'emotion')
#
# raf_list = train_list + val_list
#
# raf_set = RAFDataset(raf_list, raf_path, 'embedding')
# rafloader = DataLoader(raf_set, batch_size=1, shuffle=True, num_workers=2)

# Load datasets
fair_path = '/vol/lian/datasets/Fairface/'
trainIDs, valIDs = Fairface_spliting(fair_path)

# Step 1: Create tuples for each image filename with an associated label
train_tuples = [(filename, 'train') for filename in trainIDs]
val_tuples = [(filename, 'val') for filename in valIDs]

# Step 2: Combine the tuples into a single list
fair_list = train_tuples + val_tuples


fair_set = FairfaceDataset(fair_list, fair_path, 'embedding')
fairloader = DataLoader(fair_set, batch_size=1, shuffle=True, num_workers=2)

# utk_path = '/vol/lian/datasets/utkcropped/'
# trainIDs, valIDs = UTK_spliting(utk_path)
#
# utk_list = trainIDs + valIDs
#
# utk_set = UTKDataset(utk_list, utk_path, 'embedding')
# utkloader = DataLoader(utk_set, batch_size=1, shuffle=True, num_workers=2)
#
#
device = 'cuda:0'

# age_model = ResNet34_FT(pretrained=True,num_classes=5,path='/vol/lian/Fairness/checkpoints/best_model_age_age_5e3.pth',mode='embedding')
# gender_model = ResNet34_FT(pretrained=True,num_classes=2,path='/vol/lian/Fairness/checkpoints/best_model_gender_gender_classifier.pth',mode='embedding')
# race_model = ResNet34_FT(pretrained=True,num_classes=3,path='/vol/lian/Fairness/checkpoints/best_model_race_race_classifier.pth',mode='embedding')
#raf_emotion_model = ResNet34_FT(pretrained=True,num_classes=7,path='/vol/lian/Fairness/checkpoints/best_model_RAF_R34_0.0003.pth',mode='embedding')
# aff_emotion_model = ResNet34_FT(pretrained=True,num_classes=7,path='/vol/lian/Fairness/checkpoints/best_model_Aff_R34.pth',mode='embedding')
#exp_emotion_model = ResNet34_FT(pretrained=True,num_classes=7,path='/vol/lian/Fairness/checkpoints/best_model_Exp_R34.pth',mode='embedding')

age_model = ResNet34_FT(pretrained=True,num_classes=9,path='/vol/lian/Fairness/checkpoints/best_model_age_Fair_age_0.0002.pth',mode='embedding')
gender_model = ResNet34_FT(pretrained=True,num_classes=2,path='/vol/lian/Fairness/checkpoints/best_model_gender_gender_classifier.pth',mode='embedding')
race_model = ResNet34_FT(pretrained=True,num_classes=7,path='/vol/lian/Fairness/checkpoints/best_model_race_Fair_race.pth',mode='embedding')
# age_model = ResNet34_FT(pretrained=True,num_classes=5,path='/vol/lian/Fairness/checkpoints/best_model_age_UTK_age.pth',mode='embedding')
# gender_model = ResNet34_FT(pretrained=True,num_classes=2,path='/vol/lian/Fairness/checkpoints/best_model_gender_UTK_gender.pth',mode='embedding')
# race_model = ResNet34_FT(pretrained=True,num_classes=4,path='/vol/lian/Fairness/checkpoints/best_model_race_UTK_race.pth',mode='embedding')
# raf_emotion_model = ResNet34_FT(pretrained=True,num_classes=7,path='/vol/lian/Fairness/checkpoints/best_model_RAF_R34_SGD.pth',mode='embedding')
# aff_emotion_model = ResNet34_FT(pretrained=True,num_classes=7,path='/vol/lian/Fairness/checkpoints/best_model_Aff_R34.pth',mode='embedding')

age_model.to(device)
gender_model.to(device)
race_model.to(device)
#raf_emotion_model.to(device)
#aff_emotion_model.to(device)
#exp_emotion_model.to(device)
#
# # Function to compile data for RAF-DB
# def compile_rafdb(data_loader, emo_model, device):
#     emo_model.eval()
#     rafdb_data = []
#     emotion_label_distribution = Counter()
#
#     for images, labels, paths in tqdm(data_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         emo_emb, emo_output = emo_model(images)
#         _, emo_prediction = torch.max(emo_output.data, 1)
#
#         # Check if the prediction is correct
#
#         if emo_prediction == labels[:,0]:
#             data_entry = {
#                 "image_path": paths[0].split('/vol/lian/datasets/', 1)[-1],
#                 "emotion_embeddings": emo_emb[0].cpu().detach().numpy().tolist(),
#                 "emotion_label": labels[:, 0].item()
#             }
#
#             rafdb_data.append(data_entry)
#             # Update the distribution counter
#             emotion_label_distribution.update([labels[:, 0].item()])
#
#     # Print out the distribution of emotion labels for correctly classified instances
#     print("Distribution of emotion labels for correctly classified instances:")
#     for label, count in emotion_label_distribution.items():
#         print(f"Label {label}: {count}")
#
#     return rafdb_data
#
# Function to compile data for AffectNet
# def compile_affectnet(data_loader, emo_model, device):
#     emo_model.eval()
#     affectnet_data = []
#     emotion_label_distribution = Counter()
#     for images, labels, paths in tqdm(data_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         emo_emb, emo_output = emo_model(images)
#         _, emo_prediction = torch.max(emo_output.data, 1)
#
#         data_entry = {
#             "image_path": paths[0],
#             "emotion_embeddings": emo_emb[0].cpu().detach().numpy().tolist(),
#             "emotion_prediction": emo_prediction.item(),
#             "emotion_label": labels.item()
#         }
#         affectnet_data.append(data_entry)
#         emotion_label_distribution.update([labels.item()])
#
#     print("Distribution of emotion labels for correctly classified instances:")
#     for label, count in emotion_label_distribution.items():
#         print(f"Label {label}: {count}")
#     return affectnet_data

# def compile_exp(data_loader, emo_model, device):
#     emo_model.eval()
#     exp_data = []
#     emotion_label_distribution = Counter()
#     for images, labels, paths in tqdm(data_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         emo_emb, emo_output = emo_model(images)
#         _, emo_prediction = torch.max(emo_output.data, 1)
#
#         if emo_prediction == labels:
#             data_entry = {
#                 "image_path": paths[0],
#                 "emotion_embeddings": emo_emb[0].cpu().detach().numpy().tolist(),
#                 "emotion_prediction": emo_prediction.item(),
#                 "emotion_label": labels.item()
#             }
#             exp_data.append(data_entry)
#             emotion_label_distribution.update([labels.item()])
#
#     print("Distribution of emotion labels for correctly classified instances:")
#     for label, count in emotion_label_distribution.items():
#         print(f"Label {label}: {count}")
#     return exp_data

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

# def compile_utk(data_loader, age_model, gender_model, race_model, device):
#     age_model.eval()
#     gender_model.eval()
#     race_model.eval()
#
#     utk_data = []
#     age_label_distribution = Counter()
#     gender_label_distribution = Counter()
#     race_label_distribution = Counter()
#     for images, labels, paths in tqdm(data_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         age_emb, age_output = age_model(images)
#         gender_emb, gender_output = gender_model(images)
#         race_emb, race_output = race_model(images)
#         _, age_prediction = torch.max(age_output.data, 1)
#         _, gender_prediction = torch.max(gender_output.data, 1)
#         _, race_prediction = torch.max(race_output.data, 1)
#
#         data_entry = {
#             "image_path": paths[0],
#             "age_embeddings": age_emb[0].cpu().detach().numpy().tolist(),
#             "age_prediction": age_prediction.item(),
#             "age_label": labels[:, 1].item(),
#             "gender_embeddings": gender_emb[0].cpu().detach().numpy().tolist(),
#             "gender_prediction": gender_prediction.item(),
#             "gender_label": labels[:, 0].item(),
#             "race_embeddings": race_emb[0].cpu().detach().numpy().tolist(),
#             "race_prediction": race_prediction.item(),
#             "race_label": labels[:, 2].item()
#         }
#         utk_data.append(data_entry)
#         age_label_distribution.update([labels[:, 1].item()])
#         gender_label_distribution.update([labels[:, 0].item()])
#         race_label_distribution.update([labels[:, 2].item()])
#
#     print("Distribution of age labels for correctly classified instances:")
#     for label, count in age_label_distribution.items():
#         print(f"Label {label}: {count}")
#     print("Distribution of gender labels for correctly classified instances:")
#     for label, count in gender_label_distribution.items():
#         print(f"Label {label}: {count}")
#     print("Distribution of race labels for correctly classified instances:")
#     for label, count in race_label_distribution.items():
#         print(f"Label {label}: {count}")
#     return utk_data

# Compile data for RAF-DB
#rafdb_data = compile_rafdb(rafloader, raf_emotion_model, device)
# Compile data for AffectNet
#exp_data = compile_exp(Exploader, exp_emotion_model, device)
# aff_data = compile_affectnet(affloader, aff_emotion_model, device)
fairface_data = compile_fairface(fairloader, age_model, gender_model, race_model, device)
#utk_data = compile_utk(utkloader, age_model, gender_model, race_model, device)


#Save to JSON file for RAF-DB
# with open('rafdb_data.json', 'w') as f:
#     json.dump(rafdb_data, f)

# # Save to JSON file for AffectNet
# with open('affectnet_data_all.json', 'w') as f:
#     json.dump(affectnet_data, f)

with open('fairface_data.json', 'w') as f:
    json.dump(fairface_data, f)

# with open('utk_data.json', 'w') as f:
#     json.dump(utk_data, f)