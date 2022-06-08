import os
from collections import OrderedDict
from pathlib import Path
import json

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import concatenate_datasets, load_dataset, load_metric, load_from_disk
from omegaconf import OmegaConf
from PIL import Image
from pkg_resources import packaging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             top_k_accuracy_score)
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
import optuna
import clip

# os.chdir('/content/drive/MyDrive/project/buzzni')

cfg = OmegaConf.load('clip_config.yaml')
catalog = cfg.catalog

THRESHOLD = cfg.THRESHOLD
K = cfg.K
BATCH_SIZE = cfg.BATCH_SIZE
EVAL_BATCH_SIZE = cfg.EVAL_BATCH_SIZE
PREP_BATCH_SIZE = cfg.PREP_BATCH_SIZE
HUMAN_IN_THE_LOOP = cfg.HUMAN_IN_THE_LOOP

DEBUG = cfg.DEBUG
UPDATE_SIZE = cfg.UPDATE_SIZE
WARMUP_EPOCH = cfg.WARMUP_EPOCH
MODEL_NAME = cfg.MODEL_NAME

# clip.available_models()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(MODEL_NAME,device=device,jit=False) #Must set jit=False for training
model.eval()
checkpoint = torch.load(catalog.model_weight)
model.load_state_dict(checkpoint['model_state_dict'])
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")

id2txt = (
    pd.read_csv(catalog.catalog, index_col=['index'])
    ['en_final']
    .astype(str)
    .str.strip()
    .to_dict()
)

id2txt = {index : f"This is a photo of a {txt}" for index,txt in id2txt.items()}
text_tokens = clip.tokenize(id2txt.values()).to(device)
class_ids = list(range(50))

test_ds_path = Path(catalog.test_ds_path)
test_ds = load_from_disk(test_ds_path)
   
def compute_metrics(labels, probs):
    # labels = np.array(dataset['label'])
    num_samples = len(labels)
    preds = probs.argmax(axis=-1)
    certain = probs.max(axis=-1) > THRESHOLD

    num_certain = certain.sum()
    num_uncertain = (~certain).sum()

    # certain & correct
    TP_at_1 = (labels[certain] ==  preds[certain]).sum()
    # certain & false
    FP_at_1 = (labels[certain] !=  preds[certain]).sum()

    # uncertain & correct at top_k
    TP_at_k = top_k_accuracy_score(
        labels[~certain],
        probs[~certain],
        k = K, normalize=False, labels=class_ids) 

    # ucertain & false
    FP_at_k = num_uncertain - TP_at_k

    # certain & false is worse than uncertain and false(not in top5), but how much? both got wrong.
    FP = FP_at_1  + FP_at_k

    # few percentage of FP_at_k could be rectified. so add to FP_at_k
    # up to UPDATE_SIZE samples. fp_at_k and tp_at_k not differ except latency. 
    # hyper parameter optimization step(with multi objective) penalize latency there.
    TP_at_k = FP_at_k * np.min([1, UPDATE_SIZE / num_uncertain]) * 0.1  
    # penalize by K
    TP =  TP_at_1 + TP_at_k / K
    buzzni = TP / (TP + FP)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {
        'buzzni': buzzni,
        'acc_at_1': acc,
        'acc_at_k': top_k_accuracy_score(labels, probs, k = K, normalize=True, labels=class_ids), 
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

@torch.no_grad()
def eval_epoch(dataset, model, human_in_the_loop=False):
    model.eval()
    batch_size = EVAL_BATCH_SIZE
    labels = np.array(dataset['label'])

    steps, residue = np.divmod(len(dataset), batch_size)
    if residue > 0: steps += 1

    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    shape = dataset.shape[0], 50
    probs = np.zeros(shape, dtype=np.float16)

    for step in tqdm.trange(steps): 
        image_input = torch.tensor(np.stack(dataset[step*batch_size:(step+1)*batch_size]['image'])).to(device)
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs[step*batch_size:(step+1)*batch_size] = text_probs.cpu().numpy().astype(np.float16)
        
    metrics = compute_metrics(labels, probs)

    if not human_in_the_loop:
        return metrics

test_metrics = eval_epoch(test_ds, model)
test_metrics = {f'test_{k}':v for k, v in test_metrics.items()}
with open(catalog.final_score, 'w') as f:
    json.dump(test_metrics, f)





