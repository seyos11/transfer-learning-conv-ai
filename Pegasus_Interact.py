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
from argparse import ArgumentParser

from transformers import cached_path

def get_data_loaders():
    """ Prepare the dataset for training and evaluation """
    dataset_path = ""
    dataset_cache = None
    personachat = get_dataset(dataset_path, dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    personality = []
    history_complete = []
    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(1, num_candidates)
        for dialog in dataset:
            persona = dialog["persona_info"].copy()
            #datasets[personality].append(persona)
            count_history = 0
            for utterance in dialog["utterances"]:
                count_history = count_history + 1
                history = utterance["history"][-(2*2+1):]
                #history_complete.append(history)
                if len(history) > 4:
                    instance = build_input_from_segments(persona, history)     
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array) 
    return datasets

def run():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="results2_3epochs_2batch/checkpoint-143500", help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()
    tokenizer = PegasusTokenizer.from_pretrained(args.model_checkpoint)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_checkpoint) 
    model.to("cpu")
    dataset = get_data_loaders()
    count= 0
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        #batch = tokenizer.prepare_seq2seq_batch(raw_text, truncation=True, padding='longest')
        batch = tokenizer(raw_text, truncation=True, padding="longest", return_tensors="pt").to('cpu')
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        #tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(tgt_text)
        print(dataset['valid'])
        count = count +1

if __name__ == "__main__":
    run()
