# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
import pickle
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
import numpy as np
import json

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
SPECIAL_TOKENS_2 = ["<bos>","<eos>","<persona1>","<persona2>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>',
                         'pad_token': '<pad>', 'additional_special_tokens': ['<speaker1>', '<speaker2>']}
ATTR_TO_SPECIAL_TOKEN_2 = {'bos_token': '<bos>', 'eos_token': '<eos>',
                         'pad_token': '<pad>', 'additional_special_tokens': ['<persona1>','<persona2>','<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

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

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()

#This method equalize vectors. All must have the same length. Thus, the longest vector is taken as reference
#The rest of vectors are going to be filled 
def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN_2) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
def tokenize(tokenizer_selected,obj):
    if isinstance(obj, str):
        return tokenizer_selected.convert_tokens_to_ids(tokenizer_selected.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)
def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + persona] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[1 if (len(sequence)-i) % 2 else 0] + s for i, s in enumerate(sequence[1:])]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [1 if i % 2 else 0 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance

def build_input_from_segments2(persona1, persona2, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, persona1_token,persona2_token,speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS_2[:-1])
    #Its created a new list of list where the first token makes reference to the speaker, except the persona list, where is pointed
    #out with <bos>. The last token of the last list is <eos>.
    selected_persona = persona1[I[1]]
    instance = {}
    sequence = [[bos] + [speaker1]+ list(chain(*persona1))]+ [[speaker2] + list(chain(*persona2))] + history + [reply + ([eos] if with_eos else [])]
    #sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    #Dialog_State
    sequence = [sequence[0]] + [sequence[1]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[2:])]
    
    #Saved all the tokens including special tokens
    instance["input_ids"] = list(chain(*sequence))
    #token_vector_persona = [persona2_token if i %2 else persona1_token for i, s in enumerate(sequence[:2]) for _ in s]
    token_vector_persona = [speaker2 if i %2 else speaker1 for i, s in enumerate(sequence[:2]) for _ in s]
    token_vector_historial = [speaker1 if i % 2 else speaker2 for i, s in enumerate(sequence[2:]) for _ in s]
    
    instance["token_type_ids"] = token_vector_persona + token_vector_historial
    #instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    #Length: Number of tokens, i.e nº of words
    #Reiniciar indices
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    #N = length. N times [1]. if lm_labels = False
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
    return instance, sequence

#This method charges the data and split it into train,validation.
def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    #Dataset is charged in variable. It is already tokenidez
    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    personachat_raw = get_dataset_with_no_tokenizer(tokenizer, args.dataset_path, args.dataset_cache)
    with open(args.dataset_pkl, 'rb') as f:
        persona_selected_list = pickle.load(f)
    count_persona=0
    #personachat_personalities = get_dataset_personalities(tokenizer,args.dataset_path,args.dataset_cache)
    logger.info("Build inputs and labels")
    #Dictionary inside dictionary
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    persona_info_raw={"train": [], "valid": []}
    personas = {"persona_tokens": personachat, "persona_history": personachat_raw}
    for dataset_name, dataset in personachat.items():
        for persona_index, personaset in personachat_raw.items():
            for dialog in personaset:
                persona_info_raw[persona_index].append(dialog["persona_info"].copy())
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        count = 0
        for dialog in dataset:
            #persona1_raw = persona_info_raw[dataset_name][count].copy()
            #persona1 = dialog["persona_info"].copy()
            #persona_selected = get_persona_faiss_selected(args,tokenizer)
            #persona2 = dialog["persona_info2"].copy()
            #persona_selected = faiss(replyanddialog)
            #index: all persona1 sentences or all personalities
            #for _ in range(args.personality_permutations):
            for utterance in dialog["utterances"]:
                history = utterance["history"][-(2*args.max_history+1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates-1)
                    #D, I = index.search(np.array([history]), k=10)
                    #print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')
                    persona_selected = persona_selected_list[count_persona]
                    persona_selected_tokenized = tokenize(tokenizer,persona_selected)
                    instance = build_input_from_segments(persona_selected_tokenized, history, candidate, tokenizer, lm_labels)
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)
                    count_persona = count_persona + 1
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                datasets[dataset_name]["n_candidates"] = num_candidates
                #count_persona = count_persona + 1
                #persona1 = [persona1[-1]] + persona1[:-1]  # permuted personalities
                #persona2 = [persona2[-1]] + persona2[:-1]  # permuted personalities
            count = count + 1
    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS_2[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(*tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler

#Method that is called and calls the rest of methods. Via Terminal we must choose the parameters of training the network, and the path
#of the dataset
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
    parser.add_argument("--data_faiss", type=str, default="data_persona_faiss_fase1_opcion4", help="list of the personalities selected with faiss according to the strategy selected")
    parser.add_argument("--dataset_pkl", type=str, default="data_faiss_fase1.pkl", help="File where is saved the data from faiss (personalities)")    
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    model_class = GPT2DoubleHeadsModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        output_loss= model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, labels=lm_labels
        )
        loss = (output_loss.loss * args.lm_coef + output_loss.mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            #logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            output_gpt = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            lm_logits_flat_shifted = output_gpt.logits[..., :-1, :].contiguous().view(-1, output_gpt.logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, output_gpt.mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), global_step_transform=global_step_from_engine(trainer)), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()

