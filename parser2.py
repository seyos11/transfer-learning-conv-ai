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
from searching import Searcher

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

            if int(dialog_idx) == 1:
                #data.append({'persona_info': [], 'dialog': []})
                data.append({'persona_info': [],'persona_info2':[], 'utterances': []})
            
            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['persona_info'].append(persona_info)
                
            if dialog_line[0].startswith("partner's persona:"):
                persona_info2 = dialog_line[0].replace("partner's persona: ",'')
                data[-1]['persona_info2'].append(persona_info2)

            elif len(dialog_line) > 1:
                history = []
                if len(data[-1]['utterances']) > 0:
                    
                    old_history = data[-1]['utterances'][-1]['history']
                    gold_reply = data[-1]['utterances'][-1]['candidates'][-1]
                    for sentence in old_history:
                        history.append(sentence)
                    history.append(gold_reply)
                    history.append(dialog_line[0])
                else:
                    history.append(dialog_line[0])
                    
                data[-1]['utterances'].append({'history': [], 'candidates': []})
                for sentence in history:
                    data[-1]['utterances'][-1]['history'].append(sentence)
                candidates_list  = dialog_line[3].split('|')
                for candidate in candidates_list:
                    data[-1]['utterances'][-1]['candidates'].append(candidate)
                #data[-1]['dialog'].append(dialog_line[1])
                    

        return data
def parse_data_searched(path):
    searcher = Searcher(100,'jobsearch')
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

            if int(dialog_idx) == 1:
                #data.append({'persona_info': [], 'dialog': []})
                data.append({'persona_info': [],'persona_info2':[], 'utterances': []})
            
            dialog_line = line[space_idx + 1:].split('\t')
            dialog_line = [l.strip() for l in dialog_line]

            if dialog_line[0].startswith('your persona:'):
                persona_info = dialog_line[0].replace('your persona: ', '')
                data[-1]['persona_info'].append(persona_info)
                
            if dialog_line[0].startswith("partner's persona:"):
                #persona_info2 = dialog_line[0].replace("partner's persona: ",'')
                #data[-1]['persona_info2'].append(persona_info2)
                continue    
            elif len(dialog_line) > 1:
                history = []
                if len(data[-1]['utterances']) > 0:
                    
                    old_history = data[-1]['utterances'][-1]['history']
                    gold_reply = data[-1]['utterances'][-1]['candidates'][-1]
                    for sentence in old_history:
                        history.append(sentence)
                    history.append(gold_reply)
                    history.append(dialog_line[0])
                else:
                    history.append(dialog_line[0])
                    
                data[-1]['utterances'].append({'history': [], 'candidates': []})
                for sentence in history:
                    data[-1]['utterances'][-1]['history'].append(sentence)
                candidates_list  = dialog_line[3].split('|')
                for candidate in candidates_list:
                    data[-1]['utterances'][-1]['candidates'].append(candidate)
                #data[-1]['dialog'].append(dialog_line[1])
                persona_id = searcher(dialog_line[0])
                persona_info2   = searcher.search(dialog_line[0])
                if len(data[-1]['persona_info2']) > 5:
                    data[-1]['persona_info2'][randrange(5)] = persona_info2
                else:
                    data[-1]['persona_info2'].append(persona_info2)

        return data
    

def save_as_json(data, filename):
    with open('filename', 'w') as outfile:
        json.dump(data, outfile)