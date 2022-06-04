#!pip install datasets transformers
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

wd = '/content/drive/MyDrive/project/buzzni'
os.chdir(wd)

PRETRAIN_OUTPUT_DIR = './mae_pretrain'

# ds = load_dataset("imagefolder", data_dir="./data", split="test")

train_dataset = load_dataset("imagefolder", data_dir="./data", split='test[:80%]')
validation_dataset = load_dataset("imagefolder", data_dir="./data", split='test[80%:]')

train_dataset[0]['image']


def convert_rgb(example):
    example['image'] = example['image'].convert('RGB')
    return example




model_name_or_path = "facebook/vit-mae-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)

# configuration = ViTMAEConfig(num_channels=3)
# model = ViTMAEForPreTraining(configuration).from_pretrained(model_name_or_path)
# configuration = model.config
model = ViTMAEForPreTraining.from_pretrained(model_name_or_path)


def process_example(example):
    inputs = feature_extractor(example['image'], return_tensors='pt')
    return inputs

def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    return inputs

prepared_train_dataset = train_dataset.with_transform(transform)
prepared_validation_dataset = validation_dataset.with_transform(transform)

{sample['image'].mode for sample in train_dataset}
{sample['image'].mode for sample in validation_dataset}



feature_extractor


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

training_args = TrainingArguments(
  output_dir=PRETRAIN_OUTPUT_DIR,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=10,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-5,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    # compute_metrics=compute_metrics,
    train_dataset=prepared_train_dataset,
    eval_dataset=prepared_validation_dataset,
    tokenizer=feature_extractor,
)


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
metrics = trainer.evaluate(prepared_validation_ds)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

kwargs = {
    "finetuned_from": model.config._name_or_path,
    "tasks": "image-classification",
    "dataset": 'beans',
    "tags": ['image-classification'],
}


