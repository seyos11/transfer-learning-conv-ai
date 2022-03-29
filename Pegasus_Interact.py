"""Script for fine-tuning Pegasus
Example usage:
  # use XSum dataset as example, with first 1000 docs as training data
  from datasets import load_dataset
  dataset = load_dataset("xsum")
  train_texts, train_labels = dataset['train']['document'][:1000], dataset['train']['summary'][:1000]
  
  # use Pegasus Large model as base for fine-tuning
  model_name = 'google/pegasus-large'
  train_dataset, _, _, tokenizer = prepare_data(model_name, train_texts, train_labels)
  trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
  trainer.train()
 
Reference:
  https://huggingface.co/transformers/master/custom_datasets.html

"""

from transformers import PegasusForConditionalGeneration, PegasusModel, PegasusTokenizer, Trainer, TrainingArguments
import torch
from datetime import datetime
import json
import logging
from collections import defaultdict
import os
import tarfile
import tempfile
import socket
from itertools import chain

from transformers import cached_path


 while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            in_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
            batch = tokenizer.prepare_seq2seq_batch(in_text, truncation=True, padding='longest').to(torch_device) 
            translated = model.generate(**batch)
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(tgt_text)
logger.info("Get pretrained model and tokenizer")
tokenizer = PegasusTokenizer.from_pretrained("results1/checkpoint-95500/")
model = PegasusForConditionalGeneration.from_pretrained("results1/checkpoint-95500/")  
  # Check results
in_text = [in_df['allTextReprocess'].iloc[3]]
batch = tokenizer.prepare_seq2seq_batch(in_text, truncation=True, padding='longest').to(torch_device) 

translated = model.generate(min_length=min_length, max_length=max_length, **batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(tgt_text)