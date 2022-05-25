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

from transformers import PegasusForConditionalGeneration, PegasusModel, PegasusTokenizer, Trainer, TrainingArguments, GPT2Tokenizer, OpenAIGPTTokenizer
import torch
from datetime import datetime
import json
import logging
from collections import defaultdict
import pickle
import os
import tarfile
import tempfile
import socket
import time
from itertools import chain
from datasets import load_metric
from transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

logger = logging.getLogger(__file__)

class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

  
def get_dataset(dataset_path, dataset_cache=None):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache) 
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open('data_personachat.json', "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset

def get_data_loaders():
    """ Prepare the dataset for training and evaluation """
    dataset_path = ""
    dataset_cache = None
    personachat = get_dataset(dataset_path, dataset_cache)

    tokenizer_selected = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    personality = []
    history_complete = []
    count_persona = 0
    with open('data_faiss_pegasus_2sentences_finalgenerated.pkl', 'rb') as f:
        persona_selected_list = pickle.load(f)
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
                history = utterance["history"][-(2*2+5):]
                
                #history_complete.append(history)
                if len(persona) == 4:
                    if len(history) > (len(persona)+3):
                        history_chatbot = history[1::2]
                        persona_selected = persona_selected_list[count_persona]
                        instance = build_input_from_segments_faiss_2(persona_selected, history_chatbot)     
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                        count_persona = count_persona + 1
    return datasets

def get_data_loaders_1sentence():
    """ Prepare the dataset for training and evaluation """
    dataset_path = ""
    dataset_cache = None
    personachat = get_dataset(dataset_path, dataset_cache)

    tokenizer_selected = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    personality = []
    history_complete = []
    count_persona = 0
    with open('data_faiss_pegasus_1_sentence_final_generated.pkl', 'rb') as f:
        persona_selected_list = pickle.load(f)
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
                if len(history) > 3:
                    history_chatbot = history[1]
                    persona_selected = persona_selected_list[count_persona]
                    instance = build_input_from_segments_faiss(persona_selected, history_chatbot)     
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                    count_persona = count_persona + 1
    return datasets
def get_data_loaders_2sentences():
    """ Prepare the dataset for training and evaluation """
    dataset_path = ""
    dataset_cache = None
    personachat = get_dataset(dataset_path, dataset_cache)

    tokenizer_selected = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    personality = []
    history_complete = []
    count_persona = 0
    with open('data_faiss_pegasus_2generated.pkl', 'rb') as f:
        persona_selected_list = pickle.load(f)
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
                  history_chatbot = history[1::2]

                  persona_selected = persona_selected_list[count_persona]
                  instance = build_input_from_segments_faiss_2(persona_selected, history_chatbot)     
                  for input_name, input_array in instance.items():
                      datasets[dataset_name][input_name].append(input_array)
                  count_persona = count_persona + 1
    return datasets
def get_data_loaders_3sentences():
    """ Prepare the dataset for training and evaluation """
    dataset_path = ""
    dataset_cache = None
    personachat = get_dataset(dataset_path, dataset_cache)

    tokenizer_selected = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    personality = []
    history_complete = []
    count_persona = 0
    with open('data_faiss_pegasus_3generated.pkl', 'rb') as f:
        persona_selected_list = pickle.load(f)
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
                history = utterance["history"][-(2*2+3):]
                #history_complete.append(history)
                if len(history) > 6:
                        history_chatbot = history[1::2]
                        persona_selected = persona_selected_list[count_persona]
                        instance = build_input_from_segments_faiss_2(persona_selected, history_chatbot)     
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(input_array)
                        count_persona = count_persona + 1
    return datasets


def get_data_loaders_4sentence():
    """ Prepare the dataset for training and evaluation """
    dataset_path = ""
    dataset_cache = None
    personachat = get_dataset(dataset_path, dataset_cache)

    tokenizer_selected = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    personality = []
    history_complete = []
    count_persona = 0
    with open('data_faiss_pegasus_1generated.pkl', 'rb') as f:
        persona_selected_list = pickle.load(f)
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
                history = utterance["history"]
                #history_complete.append(history)
                if len(history_splitted) > (len(persona)-1):
                    history_chatbot = history[1::2]
                    persona_selected = persona_selected_list[count_persona]
                    instance = build_input_from_segments_faiss(persona_selected, history_chatbot)     
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                    count_persona = count_persona + 1
    return datasets
def build_input_from_segments(persona, history, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    #bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    #sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = history[1] + ' ' + history[3]
    #instance["input_ids"] = " ".join(history[-1])    
    instance["decoder_input_ids"] = " ".join(persona)
    return instance


def build_input_from_segments_faiss(persona_faiss, history, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    #bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    #sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    #instance["input_ids"] = " ".join(persona_faiss)
    #instance["input_ids"] = " ".join(history[-1])
    instance["input_ids"] = history  
    instance["decoder_input_ids"] = persona_faiss
    return instance

def build_input_from_segments_faiss_2(persona_faiss, history_chatbot, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    #bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    #sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    #instance["input_ids"] = " ".join(persona_faiss)
    #instance["input_ids"] = " ".join(history[-1])
    instance["input_ids"] = ".".join(history_chatbot)   
    instance["decoder_input_ids"] = " ".join(persona_faiss)
    return instance

def build_input_from_segments_faiss_4(persona_faiss, history_chatbot, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    #bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    #sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    #instance["input_ids"] = " ".join(persona_faiss)
    #instance["input_ids"] = " ".join(history[-1])
    instance["input_ids"] = ".".join(history_chatbot)   
    instance["decoder_input_ids"] = " ".join(persona_faiss)
    return instance

def prepare_data(model_name, 
                 train_texts, train_labels, 
                 val_texts=None, val_labels=None, 
                 test_texts=None, test_labels=None):
  """
  Prepare input data for model fine-tuning
  """
  tokenizer = PegasusTokenizer.from_pretrained(model_name)

  prepare_val = False if val_texts is None or val_labels is None else True
  prepare_test = False if test_texts is None or test_labels is None else True

  def tokenize_data(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True)
    decodings = tokenizer(labels, truncation=True, padding=True)
    dataset_tokenized = PegasusDataset(encodings, decodings)
    return dataset_tokenized

  train_dataset = tokenize_data(train_texts, train_labels)
  val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
  test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

  return train_dataset, val_dataset, test_dataset, tokenizer


def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./result_final_normal_4x4'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
  metric = load_metric("accuracy")
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=1,           # total number of training epochs
      per_device_train_batch_size=8,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=8,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='epoch',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      logging_dir='./logs1',            # directory for storing logs
      logging_steps=10,
      learning_rate = 0.0005

    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=5,           # total number of training epochs
      per_device_train_batch_size=16,   # batch size per device during training, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      #weight_decay=0.01,
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      #learning_rate=0.1
      learning_rate = 0.0005
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer

def prepare_fine_tuning_faiss1x1(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./result_final_faiss_1x1'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
  metric = load_metric("accuracy")
  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=1,           # total number of training epochs
      per_device_train_batch_size=8,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=8,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='epoch',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      logging_dir='./logs1',            # directory for storing logs
      logging_steps=10,
      learning_rate = 0.0005
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=5,           # total number of training epochs
      per_device_train_batch_size=16,   # batch size per device during training, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      #weight_decay=0.01,
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      #learning_rate=0.1
      learning_rate = 0.0005
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer

def prepare_fine_tuning_faiss2x2(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./result_final_faiss_2x2'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
  metric = load_metric("accuracy")
  def compute_metrics(eval_pred):
        
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)
  
  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=1,           # total number of training epochs
      per_device_train_batch_size=8,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=8,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='epoch',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      logging_dir='./logs2',            # directory for storing logs
      logging_steps=10,
      learning_rate = 0.0005
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=1,           # total number of training epochs
      per_device_train_batch_size=16,   # batch size per device during training, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=5,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      #weight_decay=0.01,
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      #learning_rate=0.1
      learning_rate = 0.0005
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer

def prepare_fine_tuning_faiss3x3(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./result_final_faiss_3x3'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
  metric = load_metric("accuracy")
  def compute_metrics(eval_pred):
        
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)
  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=1,           # total number of training epochs
      per_device_train_batch_size=8,   # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=8,    # batch size for evaluation, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='epoch',     # evaluation strategy to adopt during training
      eval_steps=100,                  # number of update steps before evaluation
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      logging_dir='./logs1',            # directory for storing logs
      logging_steps=10,
      learning_rate = 0.0005
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=5,           # total number of training epochs
      per_device_train_batch_size=16,   # batch size per device during training, can increase if memory allows
      save_steps=500,                  # number of updates steps before checkpoint saves
      warmup_steps=500,                # number of warmup steps for learning rate scheduler
      weight_decay=0.1,               # strength of weight decay
      #weight_decay=0.01,
      logging_dir='./logs',            # directory for storing logs
      logging_steps=10,
      #learning_rate=0.1
      learning_rate = 0.0005
    )

    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer

if __name__=='__main__':
  # use XSum dataset as example, with first 1000 docs as training data
  
  dataset = get_data_loaders()
  #dataset = load_dataset("xsum")
  train_texts, train_labels = dataset['train']['input_ids'], dataset['train']['decoder_input_ids']
  valid_texts, valid_labels = dataset['valid']['input_ids'], dataset['valid']['input_ids']

  # use Pegasus Large model as base for fine-tuning
  model_name = 'google/pegasus-large'

  #train_dataset, valid_dataset, _, tokenizer = prepare_data(model_name, train_texts, train_labels,val_texts=valid_texts, val_labels=valid_labels)
  #trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset,val_dataset=valid_dataset)
  #trainer.train()
  files = os.listdir('result_final_normal_4x4/')
  model_name_1 = './result_final_normal_4x4/' + files[0]
  dataset = get_data_loaders_1sentence()
  train_texts, train_labels = dataset['train']['input_ids'], dataset['train']['decoder_input_ids']
  valid_texts, valid_labels = dataset['valid']['input_ids'], dataset['valid']['input_ids']
  #train_dataset, valid_dataset, _, tokenizer = prepare_data(model_name_1, train_texts, train_labels,val_texts=valid_texts, val_labels=valid_labels)
  #trainer2 = prepare_fine_tuning_faiss1x1(model_name_1, tokenizer, train_dataset,val_dataset=valid_dataset)
  #trainer2.train()
  
  files = os.listdir('result_final_faiss_1x1/')
  model_name_2 = './result_final_faiss_1x1/' + files[0]
  dataset = get_data_loaders_2sentences()
  #dataset = load_dataset("xsum")
  train_texts, train_labels = dataset['train']['input_ids'], dataset['train']['decoder_input_ids']
  valid_texts, valid_labels = dataset['valid']['input_ids'], dataset['valid']['input_ids']
  train_dataset, valid_dataset, _, tokenizer = prepare_data(model_name_2, train_texts, train_labels,val_texts=valid_texts, val_labels=valid_labels)
  trainer3 = prepare_fine_tuning_faiss2x2(model_name_2, tokenizer, train_dataset,val_dataset=valid_dataset)
  trainer3.train()
  
  files = os.listdir('result_final_faiss_2x2/')
  model_name_3 = './result_final_faiss_2x2/' + files[0]
  time.sleep(100)
  dataset = get_data_loaders_3sentences()
  #dataset = load_dataset("xsum")
  train_texts, train_labels = dataset['train']['input_ids'], dataset['train']['decoder_input_ids']
  valid_texts, valid_labels = dataset['valid']['input_ids'], dataset['valid']['input_ids']
  train_dataset, valid_dataset, _, tokenizer = prepare_data(model_name_3, train_texts, train_labels,val_texts=valid_texts, val_labels=valid_labels)
  trainer4 = prepare_fine_tuning_faiss3x3(model_name_3, tokenizer, train_dataset,val_dataset=valid_dataset)
  trainer4.train()
  
  
  
'''   # Check results
in_text = [in_df['allTextReprocess'].iloc[3]]
batch = tokenizer.prepare_seq2seq_batch(in_text, truncation=True, padding='longest').to(torch_device) 

translated = model.generate(min_length=min_length, max_length=max_length, **batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(tgt_text) '''