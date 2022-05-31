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
from itertools import chain

from transformers import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

logger = logging.getLogger(__file__)


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
def get_data_loaders4x4():
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

def get_data_loaders2x2():
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

def get_data_loaders3x3():
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
                if len(history) > (len(persona)-1):
                    history_chatbot = history[1::2]
                    if len(persona)  < 4:  
                        history_chatbot = history[1]
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
    with open('faiss_035threshold_1sentence.pkl', 'rb') as f:
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
                    history_chatbot = history[3]
                    persona_selected = persona_selected_list[count_persona]
                    if (persona_selected != '<None>'):
                        instance = build_input_from_segments_faiss(persona_selected, history_chatbot)     
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


if __name__=='__main__':
      # use XSum dataset as example, with first 1000 docs as training data
    from datasets import load_dataset
    dataset = get_data_loaders_1sentence()
  
    with open('dataload_pegasus_1x1_threshold035.pkl', 'wb') as f:
        pickle.dump(dataset, f)