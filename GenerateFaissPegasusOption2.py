# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
import pickle
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events 
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from utils_faiss import get_dataset, make_logdir, get_dataset_with_no_tokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import sys, os

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
SPECIAL_TOKENS_2 = ["<bos>","<eos>","<persona1>","<persona2>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>',
                         'pad_token': '<pad>', 'additional_special_tokens': ['<speaker1>', '<speaker2>']}
ATTR_TO_SPECIAL_TOKEN_2 = {'bos_token': '<bos>', 'eos_token': '<eos>',
                         'pad_token': '<pad>', 'additional_special_tokens': ['<persona1>','<persona2>','<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)
''' def getDocumentsByScore(score=0.5,index):
    T = []
    D, I = search(charset, criterion[, ...])
    for i in D:
        if i >0.5:
            T.append(i) '''
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__sdout__
    
def parse_data(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            line = line.strip()

            if len(line) == 0:
                continue

            space_idx = line.find(' ')
            if space_idx == -1:
                dialog_idx = int(line)
            else:
                dialog_idx = int(line[:space_idx])

            #if int(dialog_idx) == 1:
                #data.append({'persona_info': [], 'dialog': []})
                #data.append({'personalities': []})
            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data.append(persona_info)
                
            if dialog_line[0].startswith("partner's persona:"):
                persona_info2 = dialog_line[0].replace("partner's persona: ",'')
                data.append(persona_info2)
          # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
        
        return data


def get_persona_faiss_selected(args):
    """ Prepare the dataset for training and evaluation """
    tokenizer=""
    personachat = get_dataset_with_no_tokenizer(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    persona_faiss_selected = []
    history_faiss_selected = []
    persona_faiss_index = []
    history_faiss_index = []
    persona_complete = parse_data('./Dataset/train_self_original.txt')
    model = SentenceTransformer('all-mpnet-base-v2')
    count = 0



    for dataset_name, dataset in personachat.items():
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for dialog in tqdm(dataset):
            #persona = dialog["personality"].copy()
            persona = dialog["persona_info"].copy()
            #persona2 = dialog["persona_info2"].copy()
            #persona_selected = faiss(replyanddialog)
            #index: all persona1 sentences or all personalities
            #model1 = SentenceTransformer('bert-large-nli-mean-tokens')
            #model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased')
            embeddings_persona = model.encode(persona, show_progress_bar=False)   
            # Step 1: Change data type
            embeddings_persona = np.array([embedding for embedding in embeddings_persona]).astype("float32")

            # Step 2: Instantiate the index
            index = faiss.IndexFlatIP(embeddings_persona.shape[1])

            #index = faiss.IndexFlatL2(embeddings_persona.shape[1])
            #Sklearn normalize from preprocessing (embeddings)
            # Step 3: Pass the index to IndexIDMap
            index = faiss.IndexIDMap2(index)
            #index = faiss.IndexIDMap(index)
            # Step 4: Add vectors and their IDs
            index.add_with_ids(embeddings_persona, np.array(list(range(0,embeddings_persona.shape[0])))) 
            #if count==4:
            #    break
            count = count + 1
            #data_train list of set of list of all personalities (not duplicated)
            for _ in range(args.personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*args.max_history+1):]
                    if len(history) > 1:                   
                        #history_encoded = model.encode(history,show_progress_bar=False)
                        #history_splitted = " ".join(history)
                        #history_splitted = history[1] + ' ' + history[3]                                     
                        history_splitted = history[1]

                        history_encoded = model.encode(history_splitted,show_progress_bar=False)
                        D, I = index.search(np.array(history_encoded), k=len(persona))
                        persona_list = []
                        persona_list = persona[I[0][0]]
                        history_faiss_selected.append(history)
                        #persona_faiss_selected.append(persona[I[0][0]])
                        persona_faiss_selected.append(persona_list)
                            
                #persona = [persona[-1]] + persona[:-1]  # permuted personalities
        #break
    return persona_faiss_selected


def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="openai-gpt", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()
    data_obtained = get_persona_faiss_selected(args)
    with open('data_faiss_pegasus_1generated.pkl', 'wb') as f:
        pickle.dump(data_obtained, f)

if __name__ == "__main__":
    train()

