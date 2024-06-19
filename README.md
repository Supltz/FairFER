# iEAT-MG

## Training the encoders

- Step 1: Prepare the datasets and change the "data_path" in the files.
- Step 2: download the pre-trained model on ImageNet and train facial expression encodes by (adjusting the hyperparameters to each model and dataset accordingly):
```txt
  python train_expression.py --dataset RAF --model R34 --batchsize 8 --lr 2e-4 --momentum 0.9 --wd 5e-4 --early_stop 5
```

and train attribute encoders by (adjusting the hyperparameters to each model, attribute and dataset accordingly):
```txt
  python train_expression.py --dataset UTK --model R34 --batchsize 8 --lr 2e-4 --momentum 0.9 --wd 5e-4 --early_stop 5 --attr age
```

those encoders will be saved at ***./checkpoint***.
- Step 3: Load those encoders and form both facial expression and attribute embeddings by (adjusting each model, attribute and dataset accordingly):
```txt
  python form_embeddings.py --dataset RAF --model R34 --model_attr emotion
  python form_embeddings.py --dataset Fairface --model R34 --model_attr age
```
those embeddings (.json files) will be saved at ***./saved_embeddings***.
- Step 4: run iEAT_MG with those .json files by:
```txt
  python iEAT_MG.py --attribute age --model R34 --emo_dataset RAF --attr_dataset UTK
```
the d-values (.txt files) will be saved at ***./effect_size***.
- Step 5: Get the graphs and NAS values by:
```txt
  python plot_graph.py --emotion Happiness --model R34 --emo_dataset RAF --attr_dataset UTK --attribute age
```

