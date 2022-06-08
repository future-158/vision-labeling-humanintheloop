#pip install ftfy regex tqdm datasets transformers omegaconf git+https://github.com/openai/CLIP.git optuna
import os
from collections import OrderedDict
from pathlib import Path

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

cfg = OmegaConf.load('clip_config.yaml')
catalog = cfg.catalog

USE_PUBLIC_DATASET = cfg.USE_PUBLIC_DATASET
THRESHOLD = cfg.THRESHOLD
K = cfg.K
EPOCHS = cfg.EPOCHS
BATCH_SIZE = cfg.BATCH_SIZE
EVAL_BATCH_SIZE = cfg.EVAL_BATCH_SIZE
PREP_BATCH_SIZE = cfg.PREP_BATCH_SIZE
HUMAN_IN_THE_LOOP = cfg.HUMAN_IN_THE_LOOP
UPDATE_SIZE = cfg.UPDATE_SIZE
WARMUP_EPOCH = cfg.WARMUP_EPOCH
MODEL_NAME = cfg.MODEL_NAME

clip.available_models()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(MODEL_NAME,device=device,jit=False) #Must set jit=False for training
model.eval()
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


if Path(catalog.dataset_path).exists():
    ds = load_from_disk(catalog.dataset_path)

else:
    ds = load_dataset("imagefolder", data_dir=catalog.image_folder, split='train')
    test_ds = load_dataset("imagefolder", data_dir=catalog.image_folder, split='test')

    # true labels
    LABELS = ds.features['label'].names

    new_column = [id2txt[index] for index in ds['label']]
    ds = ds.add_column("txt", new_column)

    new_column = [id2txt[index] for index in test_ds['label']]
    test_ds = test_ds.add_column("txt", new_column)

    # set evaluation set as 10% train_dir
    ds = ds.shuffle(0).train_test_split(test_size=0.2)
    ds['eval'] = ds['test'] # rename to eval

    # after manually labeling, assign 20% of unlabel folder to test_set
    # use rest of data as unlabel dataset.

    # test_ds = test_ds.train_test_split(shuffle=True, seed=0, test_size=0.2)
    # ds['test'] = test_ds['test'] # 20%

    # empty dataset. iterative add datas.
    ds['test'] = test_ds.filter(lambda x: False)
    ds['unlabel'] = test_ds

    def tarnsform(example):
        example['image'] = preprocess(example['image'].resize((224,224)).convert('RGB'))
        return example

    def transforms(examples):
        # examples["pixel_values"] = [jitter(image.convert("RGB")) for image in examples["image"]]
        examples['image'] =  [preprocess(image.resize((224,224)).convert('RGB')) for image in examples['image']]    
        return examples

    ds = ds.map(transforms, batched=True, batch_size=PREP_BATCH_SIZE)
    ds.save_to_disk(catalog.dataset_path)

# ds.set_transform(transforms)
if cfg.DEBUG:
    ds['train'] = ds['train'].select(range(100))
    ds['eval'] = ds['eval'].select(range(100))
    ds['unlabel'] = ds['unlabel'].select(range(100))
    ds['test'] = ds['test'].select(range(100))   


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
    TP_at_k = top_k_accuracy_score(labels[~certain], probs[~certain], k = K, normalize=False, labels=class_ids) 

    # ucertain & false
    FP_at_k = num_uncertain - TP_at_k

    # certain & false is worse than uncertain and false(not in top5), but how much? both got wrong.
    FP = FP_at_1  + FP_at_k

    # few percentage of FP_at_k could be rectified. so add to FP_at_k
    # up to UPDATE_SIZE samples. fp_at_k and tp_at_k not differ except latency. 
    # hyper parameter optimization step(with multi objective) penalize latency there.
    TP_at_k += np.max([FP_at_k, UPDATE_SIZE])
    
    # penalize by K
    TP =  TP_at_1 + TP_at_k / K
    buzzni = TP / (TP + FP)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    return {
        'buzzni': buzzni,
        'acc_at_1': acc,
        'acc_at_k': top_k_accuracy_score(labels[~certain], probs[~certain], k = K, normalize=True, labels=class_ids), 
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

@torch.no_grad()
def eval_epoch(dataset, model, human_in_the_loop=False):
    model.eval()
    batch_size = EVAL_BATCH_SIZE
    steps, residue = np.divmod(len(dataset), batch_size)
    if residue > 0: steps += 1
    text_probs_numpy_li = []        

    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for step in tqdm.trange(steps): 
        image_input = torch.tensor(np.stack(dataset[step*batch_size:(step+1)*batch_size]['image'])).to(device)
        image_features = model.encode_image(image_input).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
        text_probs_numpy_li.append(
            text_probs.cpu().numpy()
        )

    text_probs_numpy = np.concatenate(text_probs_numpy_li)
    labels = np.array(dataset[:]['label']) # ? why :    
    metrics = compute_metrics(labels, text_probs_numpy)

    if not human_in_the_loop:
        return metrics

    max_proba = text_probs_numpy.max(axis=-1)
    preds = text_probs_numpy.argmax(axis=-1)
    certain =  max_proba > THRESHOLD

    # select 100 case of most unsure case.
    to_inspect = np.argsort(max_proba)[:UPDATE_SIZE]

    # for every epoch. add 10% of update_size(manual labelled) to test_set.
    test_size = int(UPDATE_SIZE / 10)
    
    np.random.seed(0)
    to_update = np.random.choice(to_inspect,test_size, replace=False)

    to_inspect = np.setdiff1d(to_inspect, to_update)    
    certain_index = np.flatnonzero(certain)
    
    # 하위 100개 중 THRESHOLD 이상인 것들은 제거. 확실한 케이스에 대한 manual inspection 필요를 줄여줌.
    to_inspect = np.setdiff1d(to_inspect, certain_index)
    to_inspect = np.concatenate([to_inspect, to_update])
    
    # calculate latency
    bad_inspect_time = 1
    worse_inspect_time = 2
    worst_inspect_time = 10

    # uncertain but still correct. human just enther next
    uncertain_correct_at_1 = (labels[to_inspect] ==  preds[to_inspect]).sum()
    
    # uncertain & incorrect. but top k suggestion is correct. human need to click 1/5
    uncertain_correct_at_k = top_k_accuracy_score(labels[to_inspect], text_probs_numpy[to_inspect], k = K, normalize=False, labels=class_ids) 
    uncertain_incorrect_at_k = len(to_inspect) - uncertain_correct_at_k

    uncertain_correct_at_k -= uncertain_correct_at_1

    bad_inspect_taken = bad_inspect_time * uncertain_correct_at_1
    worse_inspect_taken = worse_inspect_time * uncertain_correct_at_k
    worst_inspect_taken = worst_inspect_time * uncertain_incorrect_at_k

    latency = bad_inspect_taken + worse_inspect_taken + worst_inspect_taken
    # uncertain & incorrect. and top k suggestion not contains correct label. human need to manually type correct label.
    
    # uncertain 데이터의 경우 역설적으로  human이 확인하므로 certain해 져서 train_dataset에 permanently 추가됨
    permanent_appendix_dataset = dataset.select(to_inspect)
    
    # certain의 경우 중간에 human 검수 과정이 들어가지 않으므로 매 epoch마다 다시 계산해서 넣어줌
    temporary_appendix_dataset = dataset.select(certain_index)
    
    # update label to top_1 prediction values
    if temporary_appendix_dataset.shape[0] > 0:
        temporary_appendix_dataset = (
            temporary_appendix_dataset
            .remove_columns("label")
            .add_column('label', list(preds[certain]))
        )

    # remove permanent_appendix_dataset from dataset
    whole_index = np.arange(text_probs_numpy.shape[0])
    keep_index = np.setdiff1d(whole_index, to_inspect)

    test_update_dataset = dataset.select(to_update)

    # drop manualy labelled data from unlabel dataset
    updated_dataset = dataset.select(keep_index)     
    return permanent_appendix_dataset, temporary_appendix_dataset, updated_dataset, test_update_dataset, latency
            
    # _, yhat = text_probs.cpu().topk(1, dim=-1)
    # cm = confusion_matrix(ds['label'], yhat.flatten())
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot()
    # clustering = AgglomerativeClustering(n_clusters=20)
    # clustering.fit(normalize(cm, norm='l2', axis=1))
    # idx2group = {idx:group  for idx, group in enumerate(clustering.labels_)}
    # group2idx = {v:k for k,v in idx2group.items()}

class ImageCaptionDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # images = preprocess(self.ds[idx]['image'])
        images = torch.tensor(self.ds[idx]['image'])
        caption = clip.tokenize(self.ds[idx]['txt'])[0]
        return images,caption

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# torch.dot(loss, torch.from_numpy(sample_weight).float()) / len(y_pred)
# model.train()
# clip.model.convert_weights(model) # convert to fp16
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params from paper

def train_epoch(dl, model):
    model.train()
    for batch in tqdm.tqdm(dl):
        optimizer.zero_grad()
        images,texts = batch #list_images is list of image in numpy array(np.uint8)    
        logits_per_image, logits_per_text = model(images.to(device), texts.to(device))
        # ground_truth = torch.arange(BATCH_SIZE).to(device)
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()

        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model) # back to fp16


def objective(trial):
    history = []
    total_latency = 0

    # due to limited resources, fix lr to 5e-5.
    # you need to change this code to optimizer hyper paremeters
    lr = trial.suggest_categorical("lr", [5e-5]) 
    # lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9,0.98),
        eps=1e-6,
        weight_decay=0.2
        ) #Params from paper

    train_dataset = ImageCaptionDataset(ds['train'])
    # sample_weights = torch.ones(len(train_dataset), dtype=torch.float32)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(42))
        ) #Define your own dataloader


    # save zeroshot result
    eval_metrics = eval_epoch(ds['eval'], model)
    # test_metrics = eval_epoch(ds['test'], model)

    history.append({
        'epoch': -1,
        'train_size': len(train_dataset), # variadic 
        'eval_size': len(ds['eval']), # fixed
        'unlabel_size': len(ds['unlabel']), # fixed
        'test_size': len(ds['test']), # fixed
        **{f'eval_{k}':v for k, v in eval_metrics.items()},
        # **{f'test_{k}':v for k, v in test_metrics.items()},
        'latency': 0,
        'total_latency': 0
    })        

    for epoch in range(EPOCHS):
        _ = train_epoch(train_dataloader, model)

        if HUMAN_IN_THE_LOOP and ds['unlabel'].shape[0] > 0:
            # append pseudo labeled dataset with human in the loop step(mimic)
            permanent_appendix_dataset, temporary_appendix_dataset, updated_dataset, test_update_dataset, latency = eval_epoch(ds['unlabel'], model, human_in_the_loop=True)
            total_latency += latency
            # if len(updated): ds['test']['train'] = updated

            ds['test'] = concatenate_datasets([ds['test'], test_update_dataset])

            ds['unlabel'] = updated_dataset

            if len(permanent_appendix_dataset):
                permanent_appendix_dataset = ImageCaptionDataset(permanent_appendix_dataset)            
                train_dataset = torch.utils.data.ConcatDataset([train_dataset, permanent_appendix_dataset])

            temporary_appendix_dataset = ImageCaptionDataset(temporary_appendix_dataset)
            temporary_train_dataset = torch.utils.data.ConcatDataset([train_dataset, temporary_appendix_dataset])
            train_dataloader = DataLoader(
                temporary_train_dataset,
                batch_size = BATCH_SIZE,
                # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(42))
                ) #Define your own dataloader
                
                # sanity check.
            _ = next(iter(train_dataloader))
        # save untrained result
        eval_metrics = eval_epoch(ds['eval'], model)
        test_metrics = eval_epoch(ds['test'], model)

        history.append({
            'epoch': epoch,
            'train_size': len(temporary_train_dataset), # variadic 
            'eval_size': len(ds['eval']), # fixed
            'unlabel_size': len(ds['unlabel']), # fixed
            'test_size': len(ds['test']), # fixed
            **{f'eval_{k}':v for k, v in eval_metrics.items()},
            **{f'test_{k}':v for k, v in test_metrics.items()},
            'latency': latency,
            'total_latency': total_latency,
        })        

        history_dataframe = pd.DataFrame(history)
        print(history_dataframe)
    dest = Path(catalog.trial_result) 
    dest = dest.parent / f'{dest.stem}_{trial.number}{dest.suffix}'
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    history_dataframe.to_csv(dest, index=False)
    best_score = history_dataframe['eval_buzzni'].max()
    return total_latency, float(best_score)

# Path(cfg.optuna.storage).parent.mkdir(parents=True, exist_ok=True)
# if not cfg.optuna.load_if_exists:
#     if Path(cfg.optuna.storage).exists():
#     Path(cfg.optuna.storage).unlink()

study = optuna.create_study(
    study_name = cfg.optuna.study_name,
    # storage = cfg.optuna.storage,
    load_if_exists = cfg.optuna.load_if_exists,
    directions=["minimize", "maximize"])
# study = optuna.create_study(directions=["maximize"])


study.optimize(objective, n_trials=cfg.optuna.n_trials, timeout=30000)
study.trials_dataframe().to_csv(catalog.trials_dataframe)
print("Number of finished trials: ", len(study.trials))
# optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])





