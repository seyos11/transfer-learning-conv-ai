# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train_faiss_option1 import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def build_input_from_segments1(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [1 if i % 2 else 0 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        if option_faiss == 3:
            instance = build_input_from_segments1(personality, history, current_output, tokenizer, with_eos=False)
        else:
            instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
            
        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--option_faiss", type=int, default=0, help="What faiss option is selected")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    logger.info("Sample a personality")
    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    personalities = [dialog["persona_info"] for dataset in dataset.values() for dialog in dataset]
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    model_faiss = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings_persona = model_faiss.encode(personality, show_progress_bar=False)   
    # Step 1: Change data type
    embeddings_persona = np.array([embedding for embedding in embeddings_persona]).astype("float32")

    # Step 2: Instantiate the index
    index = faiss.IndexFlatL2(embeddings_persona.shape[1])

    # Step 3: Pass the index to IndexIDMap
    index = faiss.IndexIDMap(index)
    # Step 4: Add vectors and their IDs
    index.add_with_ids(embeddings_persona, np.array(list(range(0,embeddings_persona.shape[0]))))
    history = []
    personality_decoded = []
    for i in personality:
        personality_decoded.append(tokenizer.decode(i))
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        selected_personality = []
        history_decoded = []
        for i in history[-5:]:
            history_decoded.append(tokenizer.decode(i))
        if args.option_faiss == 1:
            #Búsqueda Faiss:
            D, I = index.search(np.array(history_decoded), k=len(personality_decoded))
            history_faiss_selected.append(history)
            persona_faiss_selected.append(persona_complete[I[0][0]])
            selected_personality = personality_decoded[personality_decoded[I[0][0]]]
        elif args.option_faiss == 2:
            if len(history) > 1:
                history_encoded = model_faiss.encode([history_decoded[-2]],show_progress_bar=False)
            else:
                history_encoded = model_faiss.encode([history_decoded[-1]],show_progress_bar=False)
            D, I = index.search(np.array(history_encoded, k=len(personality_decoded)))
            selected_personality = personality_decoded[I[0][0]]
        elif args.option_faiss == 3:
            if len(history) > 1:
                history_encoded = model_faiss.encode([history_decoded[-2]], show_progress_bar=False)
            else:
                history_encoded = model_faiss.encode([history_decoded[-1]],show_progress_bar=False)
            D, I = index.search(np.array(history_encoded), k=len(personality_decoded))
            persona_list = []
            for i in I[0][1:-1]:
                selected_personality.append(personality_decoded[i])
        elif args.option_faiss == 4:
            history_encoded_user = model_faiss.encode([history_decoded[-1]],show_progress_bar=False)
            D, I = index.search(np.array(history_encoded_user), k=len(personality_decoded))            
            
            index_to_be_removed = I[0][0]

            persona2 = personality_decoded[:index_to_be_removed] + personality_decoded[index_to_be_removed+1:]
            
            
            embeddings_persona2 = model_faiss.encode(persona2, show_progress_bar=False)   
            # Step 1: Change data type
            embeddings_persona2 = np.array([embedding for embedding in embeddings_persona2]).astype("float32")

            # Step 2: Instantiate the index
            index2 = faiss.IndexFlatL2(embeddings_persona2.shape[1])

            # Step 3: Pass the index to IndexIDMap
            index2 = faiss.IndexIDMap(index2)

            # Step 4: Add vectors and their IDs
            index2.add_with_ids(embeddings_persona2, np.array(list(range(0,embeddings_persona2.shape[0])))) 
            persona_list = []
            for i in I[0][1:-1]:
                persona_list.append(personality_decoded[i])
            if len(history) >1:
                history_encoded_chatbot = model_faiss.encode([history_decoded[-2]], show_progress_bar=False)
            else:
                history_encoded_chatbot = model_faiss.encode([history_decoded[-1]], show_progress_bar=False)
            T, J = index2.search(np.array(history_encoded_chatbot), k=len(persona2))
            #persona_faiss_selected.append(persona2[J[0][0]])
            selected_personality = persona2[J[0][0]]
        else:
            selected_personality = personality_decoded
        selected_personality_encoded = []
        for i in selected_personality:
            selected_personality_encoded.append(tokenizer.encode(i))
        with torch.no_grad():
            out_ids = sample_sequence(selected_personality_encoded, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(personality)
        print(selected_personality)
        print(out_text)

if __name__ == "__main__":
    run()
