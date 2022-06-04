#!pip install datasets transformers omegaconf
import os
import random
import numpy as np
import requests
import torch
from datasets import load_dataset, load_metric
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             top_k_accuracy_score)
from transformers import (AutoFeatureExtractor, Trainer, TrainingArguments,
                          ViTFeatureExtractor, ViTForImageClassification,
                          ViTMAEForPreTraining)
from transformers.utils.dummy_vision_objects import ImageGPTFeatureExtractor
from transformers import ViTMAEModel, ViTMAEConfig
from pathlib import Path
from omegaconf import OmegaConf

wd = '/content/drive/MyDrive/project/buzzni'
os.chdir(wd)

cfg = OmegaConf.load('exp/hf_mae/config.yaml')
EXP_NAME = cfg.EXP_NAME
MODEL_NAME = cfg.MODEL_NAME
PER_DEVICE_TRAIN_BATCH_SIZE =  cfg.PER_DEVICE_TRAIN_BATCH_SIZE
PER_DEVICE_EVAL_BATCH_SIZE =  cfg.PER_DEVICE_EVAL_BATCH_SIZE


PRETRAIN_OUTPUT_DIR = Path(cfg.PRETRAIN_OUTPUT_DIR)
PRETRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset("imagefolder", data_dir="./data", split='test')

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

# configuration = ViTMAEConfig(num_channels=3)
# model = ViTMAEForPreTraining(configuration).from_pretrained(MODEL_NAME)
# configuration = model.config
model = ViTMAEForPreTraining.from_pretrained(MODEL_NAME)


def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    return inputs

def transform(example_batch):
    inputs = feature_extractor([x.resize((224,224)).convert('RGB') for x in example_batch['image']], return_tensors='pt')
    return inputs

prepared_ds = ds.with_transform(transform)
prepared_ds = prepared_ds.train_test_split(seed=0, train_size=0.8)

"""# Training and Evaluation
- Define a collate function.
- Define an evaluation metric. 
- Load a pretrained checkpoint. 
- Define the training configuration.
"""

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch])
    }

steps_per_epoch = prepared_ds['train'].shape[0] // PER_DEVICE_TRAIN_BATCH_SIZE

training_args = TrainingArguments(
  output_dir=PRETRAIN_OUTPUT_DIR,
  per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, # default 16
  per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE, # default 16
  evaluation_strategy="epoch",
  save_strategy="epoch",
  logging_strategy='epoch',
  num_train_epochs=3,
  save_steps=1,
  eval_steps=1,
  logging_steps=1,
  fp16=True,
#   save_steps=steps_per_epoch,
#   eval_steps=steps_per_epoch,
#   logging_steps=steps_per_epoch,
  learning_rate=2e-5,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
  metric_for_best_model='eval_loss',  
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    # compute_metrics=compute_metrics,
    train_dataset=prepared_ds['train'],
    eval_dataset=prepared_ds['test'],
    tokenizer=feature_extractor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)



# kwargs = {
#     "finetuned_from": model.config._name_or_path,
#     "tasks": "image-classification",
#     "dataset": 'beans',
#     "tags": ['image-classification'],
# }


