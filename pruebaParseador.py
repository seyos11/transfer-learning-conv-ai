# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
import json
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

from utils import get_dataset,get_dataset_personalities
from parser import parse_data,save_as_json 

data_train = parse_data('train_both_original.txt')
data_test = parse_data('valid_both_original.txt')
data = {'train' : data_train,'valid': data_test}
#save_as_json(data,'firstjson.json')
with open('data_personachat_original.json', 'w') as outfile:
    json.dump(data, outfile, indent=4)