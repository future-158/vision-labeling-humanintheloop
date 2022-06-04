#pip install ftfy regex tqdm
#pip install git+https://github.com/openai/CLIP.git
#pip install datasets transformers omegaconf umaplearn
import os
from collections import OrderedDict

import clip
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import torch
import tqdm
from datasets import load_dataset, load_metric
from PIL import Image
from pkg_resources import packaging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             top_k_accuracy_score)
from torchvision.datasets import CIFAR100
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

clip.available_models()
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size


ds = load_dataset("imagefolder", data_dir="./data", split='train')

id2label = (
    pd.read_csv('data/catalog.csv', index_col=['index'])
    ['en_final']
    .astype(str)
    .str.strip()
    .to_dict()
)

label2id = {v:k for k,v in id2label.items()}

batch_size = 256
steps, residue = np.divmod(len(ds), batch_size)
if residue >0: steps += 1

image_features = []
for step in tqdm.trange(steps):

    images = [
        preprocess(image.resize((224,224)).convert('RGB'))
        for image
        in ds[step*batch_size:(step+1)*batch_size]['image']]

    image_input = torch.tensor(np.stack(images)).cuda()
    with torch.no_grad():
        _image_features = model.encode_image(image_input).float()
    _image_features /= _image_features.norm(dim=-1, keepdim=True)
    image_features.append(_image_features)

image_features = torch.cat(image_features)

# cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
id2txt = {index : f"This is a photo of a {label}" for index,label in id2label.items()}
text_tokens = clip.tokenize(id2txt.values()).cuda()
new_column = [id2txt[index] for index in ds['label']]
ds = ds.add_column("txt", new_column)


with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)
top_k_accuracy_score(ds['label'], text_probs.cpu().numpy(), k=5)

# clustering = AgglomerativeClustering(n_clusters=30)
# clustering.fit(image_features.cpu().numpy())
# clustering.labels_

BATCH_SIZE = 32

class ImageCaptionDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        images = preprocess(self.ds[idx]['image'])
        caption = clip.tokenize(self.ds[idx]['txt']        )[0]
        return images,caption

dataset = ImageCaptionDataset(ds)
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE) #Define your own dataloader

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
clip.model.convert_weights(model)

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params from paper

for batch in tqdm.tqdm(train_dataloader):
    optimizer.zero_grad()
    images,texts = batch #list_images is list of image in numpy array(np.uint8)    
    logits_per_image, logits_per_text = model(images.to(device), texts.to(device))
    # ground_truth = torch.arange(BATCH_SIZE).to(device)
    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
    total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
    total_loss.backward()

    convert_models_to_fp32(model)
    optimizer.step()
    clip.model.convert_weights(model)
   











