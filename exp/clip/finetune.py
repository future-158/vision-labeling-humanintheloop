#pip install ftfy regex tqdm datasets transformers omegaconf
#pip install git+https://github.com/openai/CLIP.git
import os
from collections import OrderedDict
import clip
from pathlib import Path
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import torch
import tqdm
from datasets import load_dataset, load_metric, concatenate_datasets
from PIL import Image
from pkg_resources import packaging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support,
                             top_k_accuracy_score)
from torch import nn, optim
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100
from omegaconf import OmegaConf


wd = '/content/drive/MyDrive/project/buzzni'
os.chdir(wd)

cfg = OmegaConf.load('clip_config.yaml')
THRESHOLD = cfg.THRESHOLD
K = cfg.K
EPOCHS = cfg.EPOCHS
BATCH_SIZE = cfg.BATCH_SIZE
HUMAN_IN_THE_LOOP = cfg.HUMAN_IN_THE_LOOP

WARMUP_EPOCH = 1
UPDATE_SIZE = 100
DENOMINATOR = K

clip.available_models()
# model, preprocess = clip.load("ViT-B/32")
# model.cuda().eval()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
model.eval()


# input_resolution = model.visual.input_resolution
# context_length = model.context_length
# vocab_size = model.vocab_size

test_ds = load_dataset("imagefolder", data_dir="./data", split='test')
test_ds = test_ds.add_column('uid', list(range(len(test_ds))))

ds = load_dataset("imagefolder", data_dir="./data", split='train')
ds = ds.add_column('uid', list(range(len(ds))))

"""
rename_columns
save_to_disk
select
shape
shard
shuffle
sort
to_pandas
train_test_split
""" 

id2txt = (
    pd.read_csv('data/catalog.csv', index_col=['index'])
    ['en_final']
    .astype(str)
    .str.strip()
    .to_dict()
)


LABELS = [int(x) for x in np.sort(np.unique(ds['label']))]
id2txt = {index : f"This is a photo of a {txt}" for index,txt in id2txt.items()}
text_tokens = clip.tokenize(id2txt.values()).cuda()

new_column = [id2txt[index] for index in ds['label']]
ds = ds.add_column("txt", new_column)

new_column = [id2txt[index] for index in test_ds['label']]
test_ds = test_ds.add_column("txt", new_column)


ds = ds.shuffle(0).train_test_split(test_size=0.2)

# split test_ds to train / test
# test_ds[test]: real test set
# test_ds[train]: unlabeled set. for each epoch, human in the loop train / test

# every epoch only 100 uncertain case studied. because time limit. this is also hyper parameter.

def compute_metrics(dataset, probs):
    labels = np.array(dataset['label'])
    num_samples = len(labels)
    preds = probs.argmax(axis=-1)
    certain = probs.max(axis=-1) > THRESHOLD
    TP_at_1 = (labels[certain] ==  preds[certain]).sum()
    FP_at_1 = (labels[certain] !=  preds[certain]).sum()
    TP_at_k = top_k_accuracy_score(labels[~certain], probs[~certain], k = K, normalize=False, labels=LABELS) 
    TP =  TP_at_1 + TP_at_k / K

    # label
    # append, drop, 
    # use additional label, additional label
    # false positive penalty?
    # every epoch train set/ valid set
    # split by index
    # append 
    # drop    
    num_certain = certain.sum()
    num_uncertain = (~certain).sum()
    FP_at_k = num_uncertain - TP_at_k
    FP = FP_at_1 * K + FP_at_k
    buzzni = TP / (TP + FP)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'acc_at_k': top_k_accuracy_score(labels[~certain], probs[~certain], k = K, normalize=True, labels=LABELS), 
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'buzzni': buzzni
    }


def eval_epoch(dataset, model, after_transform=False, human_in_the_loop=False):
    model.eval()
    batch_size = 256
    steps, residue = np.divmod(len(dataset), batch_size)
    if residue > 0: steps += 1

    image_features = []
    for step in tqdm.trange(steps):
        if after_transform:
            image_input = torch.stack(dataset[step*batch_size:(step+1)*batch_size]['image']).cuda()

        else:
            images = [
                preprocess(image.resize((224,224)).convert('RGB'))
                for image
                in dataset[step*batch_size:(step+1)*batch_size]['image']]
            image_input = torch.tensor(np.stack(images)).cuda()            
        with torch.no_grad():
            _image_features = model.encode_image(image_input).float()
            _image_features /= _image_features.norm(dim=-1, keepdim=True)
            image_features.append(_image_features)
    image_features = torch.cat(image_features)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
        
        text_props_numpy = text_probs.cpu().numpy()
        metrics = compute_metrics(dataset, text_props_numpy)

        if human_in_the_loop:
            certain = text_props_numpy.max(axis=-1) > THRESHOLD
            preds = text_props_numpy.argmax(axis=-1)
            # to_filter = np.concatenate(
            #     [
            #         np.array(dataset['uid'])[certain],
            #         np.array(dataset['uid'])[~certain][:100]
            #     ]
            # )
            # appendix_dataset = dataset.filter(lambda item: item['uid'] in to_filter)
            # dataset = dataset.filter(lambda item: item['uid'] not in to_filter)            
            # small_dataset = dataset.select([0, 10, 20, 30, 40, 50])
            # even_dataset = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
            # bert_dataset = concatenate_datasets([bookcorpus, wiki])
            dataset = dataset.remove_columns("label").add_column('label', list(np.where(certain, preds.flatten(), dataset['label'])))

            # persistence metric? 
            appendix_dataset = dataset.select(np.concatenate([np.flatnonzero(certain), np.flatnonzero(~certain)[:UPDATE_SIZE]]))
            dataset = dataset.select(np.flatnonzero(~certain)[UPDATE_SIZE:])
            model.train()
            return appendix_dataset, dataset
            
        # _, yhat = text_probs.cpu().topk(1, dim=-1)
        # cm = confusion_matrix(ds['label'], yhat.flatten())
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()
        # clustering = AgglomerativeClustering(n_clusters=20)
        # clustering.fit(normalize(cm, norm='l2', axis=1))
        # idx2group = {idx:group  for idx, group in enumerate(clustering.labels_)}
        # group2idx = {v:k for k,v in idx2group.items()}
        model.train()
        print(metrics)
        return metrics

print('train score: ', eval_epoch(ds['train'], model))
print('test score: ', eval_epoch(ds['test'], model))
# print('test2 score: ', eval_epoch(ds['test2']))

class ImageCaptionDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        images = preprocess(self.ds[idx]['image'])
        caption = clip.tokenize(self.ds[idx]['txt'])[0]
        return images,caption

train_dataset = ImageCaptionDataset(ds['train'])
# sample_weights = torch.ones(len(train_dataset), dtype=torch.float32)
train_dataloader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(42))
    ) #Define your own dataloader

# valid_dataset = ImageCaptionDataset(ds['test'])
# valid_dataloader = DataLoader(valid_dataset,batch_size = BATCH_SIZE) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

model.train()
clip.model.convert_weights(model) # convert to fp16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# torch.dot(loss, torch.from_numpy(sample_weight).float()) / len(y_pred)
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

history = []


if HUMAN_IN_THE_LOOP:
    test_ds = test_ds.shuffle(0).train_test_split(test_size=0.2)
for epoch in range(EPOCHS):
    if HUMAN_IN_THE_LOOP:
        if epoch > WARMUP_EPOCH:
            _ = train_epoch(train_dataloader, model)
            eval_metrics = eval_epoch(ds['test'], model)
            test_metrics = eval_epoch(test_ds['test'], model)
            appendix, updated = eval_epoch(test_ds['train'], model, human_in_the_loop=True)
            
            if len(updated): test_ds['train'] = updated
            if len(appendix):
                appendix_dataset = ImageCaptionDataset(appendix)
                # appendix_weights = torch.ones(len(appendix_dataset), dtype=torch.float32) / DENOMINATOR
                updated_dataset = torch.utils.data.ConcatDataset([train_dataset, appendix_dataset])
                # train_dataloader = DataLoader(updated_dataset,batch_size = BATCH_SIZE) #Define your own dataloader
                # sample_weights = torch.cat( [sample_weights, appendix_weights])

                train_dataloader = DataLoader(
                    updated_dataset,
                    batch_size = BATCH_SIZE,
                    # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True, generator=torch.Generator().manual_seed(42))
                    ) #Define your own dataloader

                _ = next(iter(train_dataloader))
        else:
            _ = train_epoch(train_dataloader, model)
            eval_metrics = eval_epoch(ds['test'], model)
            test_metrics = eval_epoch(test_ds['test'], model)            
            # bert_dataset = concatenate_datasets([ds['train'], appendix])    

    else:
        _ = train_epoch(train_dataloader, model)
        eval_metrics = eval_epoch(ds['test'], model)
        test_metrics = eval_epoch(test_ds, model)


    history.append({
        'epoch': epoch,
        'train_size': len(ds['train']),
        'eval_size': len(ds['test']),
        'unlabel_size': len(test_ds['train']),
        'test_size': len(test_ds['test']),
        **{f'eval_{k}':v for k, v in eval_metrics.items()},
        **{f'test_{k}':v for k, v in test_metrics.items()},
    })        

history_dataframe = pd.DataFrame(history)
print(history_dataframe)

dest = Path(cfg.catalog.result) 
Path(dest).mkdir(parents=True, exist_ok=True)
history_dataframe.to_csv(dest, index=False)







