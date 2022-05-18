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
import os
import tarfile
import tempfile
import socket
from itertools import chain
from argparse import ArgumentParser
import pickle
from transformers import cached_path


PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

logger = logging.getLogger(__file__)

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

def build_input_from_segments_faiss(persona, persona_faiss, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    #bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    #sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = " ".join(persona_faiss)
    #instance["input_ids"] = " ".join(history[-1])    
    instance["decoder_input_ids"] = " ".join(persona)
    return instance
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
                history = utterance["history"][-(2*2+1):]
                #history_complete.append(history)
                if len(history) > 1:
                    history_chatbot = history[1]
                    persona_selected = persona_selected_list[count_persona]
                    instance = build_input_from_segments_faiss(persona_selected, history_chatbot, persona)     
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
                history = utterance["history"][-(2*2+1):]
                #history_complete.append(history)
                if len(history) > 1:
                    history_chatbot = history[1]
                    persona_selected = persona_selected_list[count_persona]
                    instance = build_input_from_segments_faiss(persona_selected, history_chatbot, persona)     
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


def build_input_from_segments_faiss(persona_faiss, history, persona, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    #bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    #sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    #instance["input_ids"] = " ".join(persona_faiss)
    #instance["input_ids"] = " ".join(history[-1])
    instance["input_ids"] = history  
    instance["decoder_input_ids"] = persona_faiss
    instance['total_persona'] = persona
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

def run():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="results2_3epochs_2batch/checkpoint-143500", help="model checkpoint to use")
    parser.add_argument("--data_faiss", type=str, default="data_faiss_pegasus_1generated.pkl", help="pickle data to recover faiss data")
    parser.add_argument("--n_sentences", type=int, default= 1, help="sentences used to get faiss personality")

    args = parser.parse_args()
    tokenizer = PegasusTokenizer.from_pretrained(args.model_checkpoint)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_checkpoint) 
    model.to("cpu")
    if args.n_sentences == 2:
        dataset = get_data_loaders()
    else:
        dataset= get_data_loaders_1sentence()
    count= 0
    while True:
        print("History  input:")
        print(random.choice(dataset['valid']['input_ids']))
        print("\n Persona Faiss Input:")
        print(random.choice(dataset['valid']['decoder_input_ids']))
        print("\n Persona total input:")
        print(random.choice(dataset['valid']['total_persona']))
        count = count +1
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        #batch = tokenizer.prepare_seq2seq_batch(raw_text, truncation=True, padding='longest')
        batch = tokenizer(raw_text, truncation=True, padding="longest", return_tensors="pt").to('cpu')
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        #tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        print("Result of decoding")
        print(tgt_text)


if __name__ == "__main__":
    run()
