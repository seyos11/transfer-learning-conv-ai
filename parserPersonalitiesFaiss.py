#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import torch
import json
from itertools import chain
from torch.utils.data import Dataset
import os
import math
import logging
import json
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

from utils import get_dataset,get_dataset_personalities
# Used to import data from local.
import pandas as pd

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer

# Used to create and store the Faiss index.
import faiss
import numpy as np
import pickle
from pathlib import Path

# Used to do vector searches and display the results.
#from vector_engine.utils import vector_search, id2details

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

data_train = parse_data('./Dataset/train_both_revised.txt')
#data_test = parse_data_searched('valid_both_revised.txt')
data = data_train
#save_as_json(data,'firstjson.json')
#with open('Personalidades2.json', 'w') as outfile:
    #json.dump(data, outfile, indent=4)      
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)
# Step 1: Change data type
embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

# Step 2: Instantiate the index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)

# Step 4: Add vectors and their IDs
index.add_with_ids(embeddings, numpy.array(list(range(0,embeddings.shape[0]))))

D, I = index.search(np.array([embeddings[5415]]), k=10)
print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')