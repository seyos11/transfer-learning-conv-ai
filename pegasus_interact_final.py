

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
import random
from datasets import load_metric
from tqdm import tqdm
import itertools
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
                        instance = build_input_from_segments_faiss_2(persona, history_chatbot)     
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

def run():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="results2_3epochs_2batch/checkpoint-143500", help="model checkpoint to use")
    parser.add_argument("--data_faiss", type=str, default="data_faiss_pegasus_1generated.pkl", help="pickle data to recover faiss data")
    parser.add_argument("--n_sentences", type=int, default= 1, help="sentences used to get faiss personality")
    parser.add_argument("--metric", type=str, default= 'bleu', help="Metric to eval Pegasus Model")

    args = parser.parse_args()
    tokenizer = PegasusTokenizer.from_pretrained(args.model_checkpoint)
    model = PegasusForConditionalGeneration.from_pretrained(args.model_checkpoint) 
    model.to("cuda")
    dataset = get_data_loaders()        
    predictedTokens4x4 = []  
    predicciones = []
    predicted_tokens1 = [[1,2,3,4]]
    references = []
    metric_bleu = load_metric('bleu')
    metric_rouge = load_metric('rouge')
    metric_cosine_similarity = load_metric('bertscore')
    metric4x4 = metric_rouge
    print(metric_bleu.inputs_description)
    print(metric_rouge.inputs_description)
    print(metric_cosine_similarity.inputs_description)
    prueba = 1
    #metric4x4.add_batch(predictions=predicted_tokens1, references=references)    
    #metric4x4.compute(predicionts=predicted_tokens1, references = references) 
    decoded_preds = []
    decoded_labels = []
    decoded_preds_bleu = []
    if (prueba == 1):
        count = 0
        
        def postprocess_text(preds, labels):
            preds = [pred.split() for x in preds for pred in x]
            #labels = [[label.split()] for label in labels]

            return preds, labels
        def postprocess_text2(preds, labels):
            preds = [pred.strip() for x in preds for pred in x]
            #labels = [[label.strip()] for label in labels]

            return preds, labels
        for i in tqdm(dataset['valid']['input_ids'][:30]):
            batch = tokenizer(i, truncation=True, padding="longest", return_tensors="pt").to('cuda')
            output = model.generate(**batch)
            predicciones.append(output)
        #for i in tqdm(dataset['valid']['input_ids'][:100]):
            #print (i)
        #batch = tokenizer(i, truncation=True, padding="longest", return_tensors="pt").to('cuda')
        #batch2 = tokenizer(dataset['valid']['decoder_input_ids'][count],truncation=True, padding="longest", return_tensors="pt").to('cuda')
        #output = model.generate(**batch)

        for i in predicciones:
            decoded_preds.append(tokenizer.batch_decode(i, skip_special_tokens=True))
        for i in dataset['valid']['decoder_input_ids'][:10]:
            decoded_labels.append([i.split()])
        for i in decoded_preds:
            for j in i:
                decoded_preds_bleu.append([j.split()])
        print(decoded_preds)
        print(decoded_preds_bleu)
        print(decoded_labels)
        result1 = metric_bleu.compute(predictions=decoded_preds,references=decoded_labels)  
        print(result1)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        decoded_labels = [" ".join(i) for j in decoded_labels for i in j]
        decoded_preds = [" ".join(i) for i in decoded_preds]
        #result2 = metric_rouge.compute(predictions=decoded_preds,references=decoded_labels)  

        #decoded_preds = list(itertools.chain(*decoded_preds))
        #print(predicciones)
        #decoded_preds = tokenizer.batch_decode(predicciones, skip_special_tokens=True)
        #decoded_labels = [dataset['valid']['decoder_input_ids'][:30]]
        #decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        #print(output)
        #print(batch2)
        #print(batch)
        #predictedTokens4x4.append(model.generate(**batch))
        #print(decoded_preds)
        #print("decoded labels")
        #print(decoded_labels)
        #predictedTokens4x4.append(decoded_preds)
        #references.append(decoded_labels)
            ##metric4x4.add(prediction=output, reference=batch2['input_ids'])    
        #metric4x4 = load_metric('bleu')
        #metric4x4.add_batch(predictions=predicted_tokens1, references=dataset['valid']['decoder_input_ids'])    
        #result = metric4x4.compute(predictions=decoded_preds,references=decoded_labels)  
        print("resultado 1")
        #print(result1)
        decoded_preds = ["".join(decoded_preds)]
        decoded_labels =["".join(decoded_labels)]
        #print(decoded_preds)
        #print("decoded labels")
        #print(decoded_labels)
        result3 = metric_cosine_similarity.compute(predictions=decoded_preds,references=decoded_labels, lang='en')  
        #print(result3)
        
        #model = SentenceTransformer('all-mpnet-base-v2')
        #embeddings_pred = model.encode(decoded_preds, show_progress_bar=False)   
            # Step 1: Change data type
        #embeddings_pred = np.array([embedding for embedding in embeddings_pred]).astype("float32")
        #embeddings_label = model.encode(decoded_labels, show_progress_bar=False)   
            # Step 1: Change data type
        #embeddings_label = np.array([embedding for embedding in embeddings_label]).astype("float32")
        #print(embeddings_pred)
        #result = cosine_similarity(embeddings_pred, embeddings_label)
        
        
        '''     dataset = get_data_loaders()        
        predictedTokens1x1 = []  
        for i in dataset['valid']['input_ids']:
            batch = tokenizer(i, truncation=True, padding="longest", return_tensors="pt").to('cpu')
            predictedTokens1x1.append(model.generate(**batch))
        metric1x1 = load_metric('bleu')
        metric1x1.compute(predicionts=predictedTokens1x1, references = dataset['valid']['decoder_input_ids'])


        dataset = get_data_loaders_2sentences()     
        predictedTokens2x2 = []  
        for i in dataset['valid']['input_ids']:
            batch = tokenizer(i, truncation=True, padding="longest", return_tensors="pt").to('cpu')
            predictedTokens2x2.append(model.generate(**batch))
        metric2x2 = load_metric('bleu')
        metric2x2.compute(predicionts=predictedTokens2x2, references = dataset['valid']['decoder_input_ids'])

        dataset = get_data_loaders_3sentences()       
        predictedTokens3x3 = []  
        for i in dataset['valid']['input_ids']:
            batch = tokenizer(i, truncation=True, padding="longest", return_tensors="pt").to('cpu')
            predictedTokens3x3.append(model.generate(**batch))
        metric3x3 = load_metric('bleu')
        metric3x3.compute(predicionts=predictedTokens3x3, references = dataset['valid']['decoder_input_ids'])
        print(metric1x1)
        print(metric2x2) '''
        
        ''' def compute_metrics(eval_pred):
        
        logits, labels = eval_pred

        predictions = np.argmax(logits, axis=-1)

        return metric.compute(predictions=predictions, references=labels)) '''
             
        
    else:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        
        for i in tqdm(dataset['valid']['input_ids'][:10]):
            batch = tokenizer(i, truncation=True, padding="longest", return_tensors="pt").to('cuda')
            output = model.generate(**batch)
            predicciones.append(output)
        #print(predicciones)
        
        for i in predicciones:
            decoded_preds.append(tokenizer.batch_decode(i, skip_special_tokens=True))
        for i in dataset['valid']['decoder_input_ids'][:10]:
            decoded_labels.append([i])
        decoded_labels = [" ".join(i) for i in decoded_labels]
        decoded_preds = [" ".join(i) for i in decoded_preds]
        decoded_labels = " ".join(decoded_labels)
        decoded_preds = " ".join(decoded_preds)
        print(decoded_preds)
        print(decoded_labels)
        sentences = [i for i in nlp(decoded_preds).sents]
        print(sentences)
        references = [i for i in nlp(decoded_labels).sents]   

if __name__ == "__main__":
    run()
