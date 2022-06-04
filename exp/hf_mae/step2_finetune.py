#!pip install datasets transformers
import numpy as np
from transformers import TrainingArguments
from transformers import ViTForImageClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, top_k_accuracy_score
from transformers import Trainer
from datasets import load_metric
import torch
from datasets import load_dataset
from transformers.utils.dummy_vision_objects import ImageGPTFeatureExtractor
import random
from PIL import ImageDraw, ImageFont, Image
from transformers import ViTFeatureExtractor
import os
from sklearn.metrics import top_k_accuracy_score
from pathlib import Path
import shutil
from omegaconf import OmegaConf
# from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix

wd = '/content/drive/MyDrive/project/buzzni'
os.chdir(wd)

cfg = OmegaConf.load('exp/hf_mae/config.yaml')

EXP_NAME = cfg.EXP_NAME
PER_DEVICE_TRAIN_BATCH_SIZE =  cfg.PER_DEVICE_TRAIN_BATCH_SIZE
PER_DEVICE_EVAL_BATCH_SIZE =  cfg.PER_DEVICE_EVAL_BATCH_SIZE
K =  cfg.K
THRESHOLD =  cfg.THRESHOLD
LOAD_DOMAIN_ADAPTION =  cfg.LOAD_DOMAIN_ADAPTION

PRETRAIN_OUTPUT_DIR = Path(cfg.PRETRAIN_OUTPUT_DIR)
FINETUNE_OUTPUT_DIR = Path(cfg.FINETUNE_OUTPUT_DIR)
RESULT_DIR = Path(cfg.RESULT_DIR)

PRETRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINETUNE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)







# ds = load_dataset("imagefolder", data_dir="./data", split="test")
train_dataset = load_dataset("imagefolder", data_dir="./data", split='train')
# train_dataset = load_dataset("imagefolder", data_dir="./data", split='train[:80%]')
# validation_dataset = load_dataset("imagefolder", data_dir="./data", split='train[80%:]')
test_dataset = load_dataset("imagefolder", data_dir="./data", split='test')

if LOAD_CUSTOM_FODLER:
    model_name_or_path = PRETRAIN_OUTPUT_DIR
else:
    # model_name_or_path = "facebook/vit-mae-base"
    model_name_or_path = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs

# test_dataset[-1]['image'].resize((224,224)).convert('RGB')
# test_dataset[-1]['image'].convert('RGB').resize((224,224))
def transform(example_batch):
    # inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs = feature_extractor([x.resize((224,224)).convert('RGB') for x in example_batch['image']], return_tensors='pt')
    # inputs['labels'] = example_batch['labels']
    inputs['labels'] = example_batch['label']
    return inputs

prepared_train_ds = train_dataset.with_transform(transform)
prepared_validation_ds = prepared_train_ds.train_test_split(seed=0, train_size=0.8)['test']
prepared_train_ds = prepared_train_ds.train_test_split(seed=0, train_size=0.8)['train']
prepared_test_ds = test_dataset.with_transform(transform)

# save_to_disk
# rename_columns
# rename_column
# filter

"""# Training and Evaluation
- Define a collate function.
- Define an evaluation metric. 
- Load a pretrained checkpoint. 
- Define the training configuration.
"""

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

metric = load_metric("accuracy")
# def compute_metrics(p):
#     return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }


LABELS = [ int(l) for l in prepared_train_ds.features['label'].names]


def compute_metrics(pred):
    labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    num_samples = len(labels)

    # TP_at_1 = top_k_accuracy_score(labels, pred.predictions, k=1, normalize=False, labels=LABELS)
    # TP_at_k = top_k_accuracy_score(labels, pred.predictions, k = K, normalize=False, labels=LABELS) 
    # TP =  TP_at_1 + (TP_at_k - TP_at_1) / K
    # FN =  FP = num_samples - TP_at_k
    # FN =  FP = num_samples - TP_at_k
    # TP =  TP_at_1 + (TP_at_k - TP_at_1) / K
    # FN =  FP = num_samples - TP_at_k
    # buzzni = TP / (TP + FP)


    proba = np.exp(pred.predictions) / np.exp(pred.predictions).sum(axis=1, keepdims=True)
    preds = proba.argmax(axis=-1)
    sure = proba.max(axis=-1) > THRESHOLD
    TP_at_1 = (labels[sure] ==  preds[sure]).sum()
    FP_at_1 = (labels[sure] !=  preds[sure]).sum()
    TP_at_k = top_k_accuracy_score(labels[~sure], proba[~sure], k = K, normalize=False, labels=LABELS) 
    TP =  TP_at_1 + TP_at_k / K

    num_sure = sure.sum()
    num_unsure = (~sure).sum()
    FP_at_k = num_unsure - TP_at_k
    FP = FP_at_1 * K + FP_at_k
    buzzni = TP / (TP + FP)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'buzzni': buzzni
    }


labels = prepared_train_ds.features['label'].names
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

steps_per_epoch = prepared_train_ds.shape[0] // PER_DEVICE_TRAIN_BATCH_SIZE
training_args = TrainingArguments(
  output_dir=FINETUNE_OUTPUT_DIR,    
  per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, # default 16
  per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE, # default 16
  evaluation_strategy="steps",
#   eval_strategy='epoch',
  num_train_epochs=20, # default 10
  fp16=True,
  save_steps=steps_per_epoch,
  eval_steps=steps_per_epoch,
  logging_steps=steps_per_epoch,
  learning_rate=2e-5,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
  metric_for_best_model='eval_buzzni'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_train_ds,
    eval_dataset=prepared_validation_ds,
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate( prepared_validation_ds)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

metrics = trainer.evaluate( prepared_test_ds)
trainer.log_metrics("test", metrics)
trainer.save_metrics("test", metrics)

shutil.copyfile(FINETUNE_OUTPUT_DIR / 'test_results.json', RESULT_DIR / 'test_results.json')
shutil.copyfile(FINETUNE_OUTPUT_DIR / 'config.json', RESULT_DIR / 'config.json')
# kwargs = {
#     "finetuned_from": model.config._name_or_path,
#     "tasks": "image-classification",
#     "dataset": 'beans',
#     "tags": ['image-classification'],
# }


