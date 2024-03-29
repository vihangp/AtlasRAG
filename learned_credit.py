# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import List
import argparse
import time
from collections import defaultdict
import numpy as np
import random
import torch
import torch.cuda
import logging
import sys
from src.torchrun_utils import init_distributed_mode_torchrun
from src import dist_utils, slurm, util
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model, save_atlas_model
from src.options import get_options
import torch.distributed as dist
from src.tasks import get_task
from torch.nn import MultiheadAttention, Linear
from src.util import WarmupLinearScheduler, CosineScheduler, FixedScheduler
from sklearn.metrics import confusion_matrix


os.environ["TOKENIZERS_PARALLELISM"] = "true"
NCONTEXT: str = "5"
PBSZ: str = "3"
PRECISION: str = "bf16"
GOLD_SCORE_MODE: str = "ppmean"
GPU_MAX_LENGTH: str = "384"
GEN_MAX_LENGTH: str = "32"
EPSILON: str = "0.01"
SMALL_EPSILON: str = "4e-4"
DROPOUT: str = "0.1"
WARMUP_STEPS: str = "5"
EVAL_FREQ: str = "50"
LOG_FREQ: str = "5"
NO_REFRESH: str = "-1"
CHECK_FREQS: List[str] = ["--warmup_steps", "--save_freq", "--eval_freq"]
PORT: str = str(random.randrange(15000, 16000))
GRAD_SCALE_UPPER_BOUND_MEAN: int = 1000
GRAD_SCALE_LOWER_BOUND_MEAN: float = 0.01
THRESHOLD_GRAD_STATS: int = 100

logger = logging.getLogger(__name__)


def set_optim(opt, model):
    from src.AdamWFP32Copy import AdamWFP32Copy

    retr_optimizer = None
    optim_class = AdamWFP32Copy
    optim_args = {"weight_decay": opt.weight_decay, "betas": (0.9, opt.beta2), "eps": opt.epsilon}
    if opt.is_distributed and opt.shard_optim:
        from fairscale.optim.oss import OSS

        optim_args["optim"] = optim_class
        optim_args["force_broadcast_object"] = True
        optim_class = OSS
    optimizer = optim_class(params=model.parameters(), lr=opt.lr, **optim_args)

    retr_scheduler = None
    scheduler_args = {"warmup": opt.warmup_steps, "total": opt.total_steps, "ratio": 0.1}
    if opt.scheduler == "linear":
        scheduler_class = WarmupLinearScheduler
    elif opt.scheduler == "cosine":
        scheduler_class = CosineScheduler
    elif opt.scheduler == "fixed":
        scheduler_class = FixedScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)

    return optimizer, scheduler, retr_optimizer, retr_scheduler

class LCAModelAllDocGen(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.num_heads = 16
        self.embed_dim = 1024 #768

        self.num_heads = 16
        self.embed_dim = 1024 #768
        self.opt = opt

        self.self_attention = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0, batch_first=True).cuda()
        self.post_attention_layer = Linear(self.embed_dim, self.embed_dim, bias=False).cuda()
        self.pred_layer = Linear(self.embed_dim, 1, bias=False).cuda()

        # added for softmax
        self.soft_max_preds = Linear(self.embed_dim, 1, bias=False).cuda()        
    
    def forward(self, encoder_last_hidden_state, gen_tokens_embeddings):

#        print(encoder_last_hidden_state.shape, gen_tokens_embeddings.shape)

        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context , 512, 1024) # 768
        gen_tokens_embeddings = torch.repeat_interleave(gen_tokens_embeddings, repeats=self.opt.n_context, dim=0)
        gen_tokens_embeddings = gen_tokens_embeddings[:, -1, :].unsqueeze(dim=1)
#        print(gen_tokens_embeddings.shape, att_input.shape)
        fnn_input, _ = self.self_attention(gen_tokens_embeddings, att_input, att_input, need_weights=False)
        fnn_input = self.post_attention_layer(fnn_input)
        # average over output tokens
        fnn_input = torch.mean(fnn_input, dim=1, keepdim=False)

        # performance prediction
        logit = self.pred_layer(fnn_input)
        prediction_passages = torch.sigmoid(logit).squeeze()        
        # reshape to batch_size, n_context
        prediction_passages = prediction_passages.view(-1, self.opt.n_context)

        # sum over n_context
#        prediction = torch.sum(prediction, dim=1)

        # softmax prediction
        soft_logit = self.soft_max_preds(fnn_input)
        soft_logit = soft_logit.view(-1, self.opt.n_context)
        soft_max_prediction = torch.softmax(soft_logit, dim=1)

        prediction = torch.sum(prediction_passages * soft_max_prediction, dim=1)

        return prediction, prediction_passages, soft_max_prediction

class LCAModelAllDocGenSimplified(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.num_heads = 12
        self.embed_dim = 768

        self.num_heads = 12
        self.embed_dim = 768
        self.opt = opt

        self.post_attention_layer = Linear(2 * self.embed_dim, 2 * self.embed_dim, bias=False).cuda()
        self.pred_layer = Linear(2 * self.embed_dim, 1, bias=False).cuda()

        # added for softmax
        self.soft_max_preds = Linear(2 * self.embed_dim, 1, bias=False).cuda()        
    
    def forward(self, encoder_last_hidden_state, gen_tokens_embeddings):

        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context , 512, 768)
        gen_tokens_embeddings = torch.repeat_interleave(gen_tokens_embeddings, repeats=self.opt.n_context, dim=0)

        # use only the last token of the generated_token_embeddings
        gen_tokens_embeddings = gen_tokens_embeddings[:, -1, :]

        # mean of the tokens in the passage
        att_input = torch.mean(att_input, dim=1, keepdim=False)

        # concatenate the two embeddings
        gen_passage_tokens_embeddings = torch.cat((gen_tokens_embeddings, att_input), dim=1)

        fnn_input = self.post_attention_layer(gen_passage_tokens_embeddings)

        # performance prediction
        logit = self.pred_layer(fnn_input)
        prediction_passages = torch.sigmoid(logit).squeeze()        
        # reshape to batch_size, n_context
        prediction_passages = prediction_passages.view(-1, self.opt.n_context)

        # sum over n_context
#        prediction = torch.sum(prediction, dim=1)

        # softmax prediction
        soft_logit = self.soft_max_preds(fnn_input)
        soft_logit = soft_logit.view(-1, self.opt.n_context)
        soft_max_prediction = torch.softmax(soft_logit, dim=1)

        prediction = torch.sum(prediction_passages * soft_max_prediction, dim=1)

        return prediction, prediction_passages, soft_max_prediction
    

class LCAModelAllDocGenOnlySoftMax(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.num_heads = 12
        self.embed_dim = 768

        self.num_heads = 12
        self.embed_dim = 768
        self.opt = opt

        self.self_attention = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0, batch_first=True).cuda()
        self.post_attention_layer = Linear(self.embed_dim, self.embed_dim, bias=False).cuda()

        # added for softmax
        self.soft_max_preds = Linear(self.embed_dim, 1, bias=False).cuda()        
    
    def forward(self, encoder_last_hidden_state, gen_tokens_embeddings):

        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context , 512, 768)
        gen_tokens_embeddings = torch.repeat_interleave(gen_tokens_embeddings, repeats=self.opt.n_context, dim=0)
        fnn_input, _ = self.self_attention(gen_tokens_embeddings, att_input, att_input, need_weights=False)
        fnn_input = self.post_attention_layer(fnn_input)
        # average over output tokens
        fnn_input = torch.mean(fnn_input, dim=1, keepdim=False)

        # softmax prediction
        soft_logit = self.soft_max_preds(fnn_input)
        soft_logit = soft_logit.view(-1, self.opt.n_context)
#        soft_max_prediction = torch.softmax(soft_logit, dim=1)
        soft_max_prediction = soft_logit/torch.sum(soft_logit, dim=1)

        return soft_max_prediction
    

class LCAModelAllDocGenSum(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.num_heads = 12
        self.embed_dim = 768

        self.num_heads = 12
        self.embed_dim = 768
        self.opt = opt

        self.self_attention = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0, batch_first=True).cuda()
        self.post_attention_layer = Linear(self.embed_dim, self.embed_dim, bias=False).cuda()
        self.pred_layer = Linear(self.embed_dim, 1, bias=False).cuda()
    
    def forward(self, encoder_last_hidden_state, gen_tokens_embeddings):

        # b, number_of_passages, 512, 768
        # b, 16, 768
        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context , 512, 768)
        # b * number_of_passages, 512, 768

        gen_tokens_embeddings = torch.repeat_interleave(gen_tokens_embeddings, repeats=self.opt.n_context, dim=0)

        fnn_input, _ = self.self_attention(gen_tokens_embeddings, att_input, att_input, need_weights=False)
        fnn_input = self.post_attention_layer(fnn_input)
        # average over output tokens
        fnn_input = torch.mean(fnn_input, dim=1, keepdim=False)

        # performance prediction
        logit = self.pred_layer(fnn_input)
        prediction = torch.sigmoid(logit).squeeze()        
        # reshape to batch_size, n_context
        prediction_passages = prediction.view(-1, self.opt.n_context)

        # sum over n_context
        prediction = torch.sum(prediction_passages, dim=1)

        return prediction, prediction_passages

class LCABaselinePerfFunction(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.num_heads = 12
        self.embed_dim = 768

        self.num_heads = 12
        self.embed_dim = 768
        self.opt = opt

        self.cross_attention = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0, batch_first=True).cuda()
        self.post_attention_layer = Linear(self.embed_dim, self.embed_dim, bias=False).cuda()
        self.pred_layer = Linear(self.embed_dim, 1, bias=False).cuda()
    
    def forward(self, encoder_last_hidden_state, gen_tokens_embeddings):

        # b, number_of_passages, 512, 768
        # b, 16, 768
        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) , self.opt.n_context * 512, 768)
        # b, number_of_passages * 512, 768

        # b, 16, 768 : gen_tokens_embeddings
        
        fnn_input, _ = self.cross_attention(gen_tokens_embeddings, att_input, att_input, need_weights=False)
        fnn_input = self.post_attention_layer(fnn_input)
        # average over output tokens
        fnn_input = torch.mean(fnn_input, dim=1, keepdim=False)

        # performance prediction
        logit = self.pred_layer(fnn_input)
        prediction = torch.sigmoid(logit).squeeze()        

        return prediction


class LCAModelDocGen(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.num_heads = 12
        self.embed_dim = 768

        self.num_heads = 12
        self.embed_dim = 768
        self.opt = opt

        self.self_attention = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0, batch_first=True).cuda()
        self.pred_layer = Linear(self.embed_dim, 1, bias=False).cuda()        
    
    def forward(self, encoder_last_hidden_state, gen_tokens_embeddings):

        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context, 512, 768)

        fnn_input, _ = self.self_attention(gen_tokens_embeddings, att_input, att_input, need_weights=False)

        fnn_input = torch.mean(fnn_input, dim=1, keepdim=False)
        logit = self.pred_layer(fnn_input)
        prediction = torch.sigmoid(logit).squeeze()        

        return prediction

class LCAModel(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_heads = 12
        self.embed_dim = 768

        self.num_heads = 12
        self.embed_dim = 768
        self.opt = opt

        self.self_attention = MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=0.0, batch_first=True).cuda()
        self.pred_layer = Linear(self.embed_dim, 1, bias=False).cuda()        
    
    def forward(self, encoder_last_hidden_state):
        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context, 512, 768)

        fnn_input, _ = self.self_attention(att_input, att_input, att_input, need_weights=False)

        fnn_input = torch.mean(fnn_input, dim=1, keepdim=False)
        logit = self.pred_layer(fnn_input)
        prediction = torch.sigmoid(logit).squeeze()        

        return prediction


def get_argument_value(all_args: List[str], argument_name: str) -> int:

    argument_idx = all_args.index(argument_name)
    return int(all_args[argument_idx + 1])


def check_valid_input_params(all_args: List[str], total_steps: int) -> None:

    for freq in CHECK_FREQS:
        try:
            arg_val = get_argument_value(all_args, freq)
        except ValueError:
            print(f"List does not contain value {freq}")

        assert arg_val < total_steps, f"The {freq} cannot be higher than the total steps {total_steps}. "


def set_parser_options(parser: argparse.Namespace, passed_args: List[str]) -> argparse.ArgumentParser:
    """
    Sets the default options for finetuning an Atlas model for a q&a task.
    """

    total_steps = get_argument_value(passed_args, "--total_steps")

    all_args = [
        "--write_results",
        "--use_gradient_checkpoint_reader",
        "--temperature_gold",
        EPSILON,
        "--temperature_score",
        EPSILON,
        "--dropout",
        DROPOUT,
        "--lr",
        SMALL_EPSILON,
        "--lr_retriever",
        SMALL_EPSILON,
        "--scheduler",
        "linear",
        "--weight_decay",
        EPSILON,
        "--generation_max_length",
        GEN_MAX_LENGTH,
        "--target_maxlength",
        GEN_MAX_LENGTH,
        "--gold_score_mode",
        GOLD_SCORE_MODE,
        "--precision",
        PRECISION,
        "--text_maxlength",
        GPU_MAX_LENGTH,
        "--per_gpu_batch_size",
        PBSZ,
        "--n_context",
        NCONTEXT,
        "--task",
        "qa_retrieved",
        "--warmup_steps",
        WARMUP_STEPS,
        "--eval_freq",
        EVAL_FREQ,
        "--log_freq",
        LOG_FREQ,
    ] + passed_args

    check_valid_input_params(all_args, total_steps)
    return parser.parse_args(all_args)

def _get_eval_data_iterator(opt, data_path, task):
    data_iterator = task.data_iterator(data_path, opt.global_rank, opt.world_size, opt=opt, is_eval=True)
    data_iterator = filter(None, map(task.process, data_iterator))
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_batch_size_eval))

    if dist.is_initialized():
        len_data = torch.tensor(len(data_iterator), device=torch.device("cuda"))
        dist.all_reduce(len_data, torch.distributed.ReduceOp.MAX)
        dist.barrier()
        if len(data_iterator) < len_data.item():
            data_iterator.extend([{} for _ in range(len_data.item() - len(data_iterator))])

    return data_iterator

@torch.no_grad()
def evaluate(model, atlas_model, opt, data_path, step=None):
    model.eval()
    atlas_model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []

    unwrapped_model = util.get_unwrapped_model_if_wrapped(atlas_model)
    task = get_task(opt, unwrapped_model.reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        target_tokens = batch.get("target_tokens")
        scores = batch.get("scores")
        generations = batch.get("generation")

        f1_label = []
        for score_dict in scores:
            f1_label.append(score_dict["f1"])

        f1_labels = torch.tensor(f1_label, dtype=torch.float32).cuda()

        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)
        # generated_tokens["input_ids"] shape: batch_size, max_length
        generated_tokens = unwrapped_model.reader_tokenizer(
                                                            generations,
                                                            max_length=opt.target_maxlength,
                                                            padding="max_length",
                                                            truncation=True,
                                                            return_tensors="pt",
                                                            add_special_tokens=False,
                                                        )
        generated_tokens = generated_tokens["input_ids"].cuda()
        with torch.no_grad():
            gen_tokens_embeddings = unwrapped_model.reader.shared(generated_tokens)

        retrieved_passages = [p[: opt.n_context] for p in batch["passages"]]

        reader_tokens, _ = unwrapped_model.tokenize_passages(query, retrieved_passages)
        # reader_tokens: input_ids, attention_mask (batch_size, some_number, 512)
        # some_number = the max number of passages retrieved for any example in the batch

        cfg = unwrapped_model.reader.encoder.config
        cfg.bsz = reader_tokens["input_ids"].size(0)
        cfg.n_context = min(opt.n_context, reader_tokens["input_ids"].size(1))

        with torch.no_grad():
            # what is happening here? decoder input ids, what are these? 
            output = unwrapped_model.reader(
                input_ids=reader_tokens["input_ids"].cuda().view(reader_tokens["input_ids"].size(0), -1),
                attention_mask=reader_tokens["attention_mask"].cuda().view(reader_tokens["attention_mask"].size(0), -1),
                decoder_input_ids=decoder_input_ids.cuda(),
                use_cache=False,
                output_hidden_states = True
                )
            
            # encoder_last_hidden_state size: batch_size, num_passages * 512, 768
            encoder_last_hidden_state = output["encoder_last_hidden_state"]        
            decoder_hidden_state = output["decoder_hidden_states"][-1 * opt.decoder_features_layer]
            
        if opt.decoder_features_layer != 0:
            gen_tokens_embeddings = decoder_hidden_state            

        if opt.lca_one_document:
            prediction = model(encoder_last_hidden_state)
        elif opt.lca_one_document_generation:
            prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)
        elif opt.lca_multiple_document_generation:            
            prediction, prediction_passages = model(encoder_last_hidden_state, gen_tokens_embeddings) 
        elif opt.lca_multiple_document_generation_softmax:
            prediction, prediction_passages, softmax_passages = model(encoder_last_hidden_state, gen_tokens_embeddings)           
        elif opt.lca_multiple_document_generation_softmax_only:
            prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)
        elif opt.lca_multiple_document_generation_simplified:
            prediction, prediction_passages, softmax_passages = model(encoder_last_hidden_state, gen_tokens_embeddings)
        elif opt.lca_baseline_perf_function:
            prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)

        else:
            raise NotImplementedError
        
        if f1_labels.size()[0] < opt.per_gpu_batch_size_eval or prediction.size()[0] < opt.per_gpu_batch_size_eval:
            continue

        eval_loss = torch.nn.functional.mse_loss(prediction.unsqueeze(dim=1), f1_labels.unsqueeze(dim=1))

        metrics["eval_loss"].append(eval_loss.cpu())

        prediction = prediction.unsqueeze(dim=1).cpu().numpy().tolist()
        if opt.lca_multiple_document_generation or opt.lca_multiple_document_generation_softmax or opt.lca_multiple_document_generation_simplified:
            prediction_passages = prediction_passages.cpu().numpy().tolist()
            if opt.lca_multiple_document_generation_softmax or opt.lca_multiple_document_generation_simplified:
                softmax_passages = softmax_passages.cpu().numpy().tolist()
#        print(prediction.shape, f1_labels.unsqueeze(dim=1).shape, len(generations))

        for k in range(len(generations)):
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]

            ex = {"query": query[k], "answers": gold, "generation": generations[k], "scores": scores[k], "f1_pred": prediction[k]}
            if not opt.dont_write_passages:
                ex["passages"] = retrieved_passages[k]
            if "id" in batch:
                ex["id"] = batch["id"][k]
            if opt.lca_multiple_document_generation or opt.lca_multiple_document_generation_softmax or opt.lca_multiple_document_generation_simplified:
                ex["f1_pred_passages"] = prediction_passages[k]
            if opt.lca_multiple_document_generation_softmax or opt.lca_multiple_document_generation_simplified:
                ex["softmax_passages"] = softmax_passages[k]
            
            dataset_wpred.append(ex)

    all_em_scores = [dataset_wpred[i]["scores"]["exact_match"] for i in range(len(dataset_wpred))]    
    all_f1_preds = [0 if dataset_wpred[i]["f1_pred"][0] < 0.5 else 1 for i in range(len(dataset_wpred))]

    # compute confusion matrix between f1_pred
    cm = confusion_matrix(all_em_scores, all_f1_preds)
    accuracy = np.trace(cm) / np.sum(cm)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}
    metrics["accuracy"] = accuracy
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics

def train(
    model,
    atlas_model,
    optimizer,
    scheduler,
    step,
    opt,
    checkpoint_path,
):
    atlas_model.eval()
    tb_logger = util.init_tb_logger(os.path.join(opt.checkpoint_dir, opt.name), is_main=opt.is_main)
    run_stats = util.WeightedAvgStats()

    # different seed for different sampling depending on global_rank
    torch.manual_seed(opt.global_rank + opt.seed)

    scale = 2.0
    grad_stats = defaultdict(lambda: [])
    unwrapped_model = util.get_unwrapped_model_if_wrapped(atlas_model)
    task = get_task(opt, unwrapped_model.reader_tokenizer)

    while step < opt.total_steps:
        data_iterator = task.data_iterator(
            opt.train_data, opt.global_rank, opt.world_size, repeat_if_less_than_world_size=True, opt=opt
        )
        data_iterator = filter(None, map(task.process, data_iterator))
        data_iterator = task.batch_iterator(data_iterator, opt.per_gpu_batch_size, drop_last=True, shuffle=opt.shuffle)
        for i, batch in enumerate(data_iterator):

            iter_stats = {}
            model.train()
            step += 1
            train_step_start = time.time()

            query = batch.get("query", [""])
            answers = batch.get("target", [""])
            target_tokens = batch.get("target_tokens")
            scores = batch.get("scores")
            generations = batch.get("generation")

            f1_label = []
            for score_dict in scores:
                f1_label.append(score_dict["f1"])

            f1_labels = torch.tensor(f1_label, dtype=torch.float32).cuda()

            query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)
            # generated_tokens["input_ids"] shape: batch_size, max_length
            if opt.lca_one_document_generation or opt.lca_multiple_document_generation or opt.lca_multiple_document_generation_softmax or opt.lca_multiple_document_generation_softmax_only or opt.lca_multiple_document_generation_simplified:
                generated_tokens = unwrapped_model.reader_tokenizer(
                                                                    generations,
                                                                    max_length=opt.target_maxlength,
                                                                    padding="max_length",
                                                                    truncation=True,
                                                                    return_tensors="pt",
                                                                    add_special_tokens=False,
                                                                )
                generated_tokens = generated_tokens["input_ids"].cuda()
                with torch.no_grad():
                    gen_tokens_embeddings = unwrapped_model.reader.shared(generated_tokens)

            retrieved_passages = [p[: opt.n_context] for p in batch["passages"]]

            reader_tokens, _ = unwrapped_model.tokenize_passages(query, retrieved_passages)
            # reader_tokens: input_ids, attention_mask (batch_size, some_number, 512)
            # some_number = the max number of passages retrieved for any example in the batch

            cfg = unwrapped_model.reader.encoder.config
            cfg.bsz = reader_tokens["input_ids"].size(0)
            cfg.n_context = min(opt.n_context, reader_tokens["input_ids"].size(1))

            with torch.no_grad():
                output = unwrapped_model.reader(
                    input_ids=reader_tokens["input_ids"].cuda().view(reader_tokens["input_ids"].size(0), -1),
                    attention_mask=reader_tokens["attention_mask"].cuda().view(reader_tokens["attention_mask"].size(0), -1),
                    decoder_input_ids=decoder_input_ids.cuda(),
                    use_cache=False,
                    output_hidden_states = True
                    )
                
                # encoder_last_hidden_state size: batch_size, num_passages * 512, 768
                encoder_last_hidden_state = output["encoder_last_hidden_state"]

                decoder_hidden_state = output["decoder_hidden_states"][-1 * opt.decoder_features_layer]
            if opt.decoder_features_layer != 0:
                gen_tokens_embeddings = decoder_hidden_state

            if opt.lca_one_document:
                prediction = model(encoder_last_hidden_state)
            elif opt.lca_one_document_generation:
                prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)
            elif opt.lca_multiple_document_generation:            
                prediction, _ = model(encoder_last_hidden_state, gen_tokens_embeddings)
            elif opt.lca_multiple_document_generation_softmax:
                prediction, _, _ = model(encoder_last_hidden_state, gen_tokens_embeddings)     
            elif opt.lca_multiple_document_generation_softmax_only:
                prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)
            elif opt.lca_multiple_document_generation_simplified:
                prediction, _, _ = model(encoder_last_hidden_state, gen_tokens_embeddings)     
            elif opt.lca_baseline_perf_function:
                prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)
            else:
                raise NotImplementedError
            train_loss = torch.nn.functional.mse_loss(prediction, f1_labels)

            iter_stats["loss/train_loss"] = (train_loss.item(), len(batch["query"]))

            backward_start = time.time()
            train_loss = scale * train_loss
            train_loss.backward()
            iter_stats["runtime/backward"] = (time.time() - backward_start, 1)

            model_update_start = time.time()

            stats = util.compute_grad_stats_finetune_head(model)
            if stats["skip_example"]:
                model.module.zero_grad()
                # continue
            else:
                for k, v in stats.items():
                    grad_stats[k].append(v)

            if len(grad_stats["max"]) >= THRESHOLD_GRAD_STATS:
                if np.mean(grad_stats["max"]) > GRAD_SCALE_UPPER_BOUND_MEAN:
                    scale /= 2
                elif np.mean(grad_stats["mean"]) < GRAD_SCALE_LOWER_BOUND_MEAN:
                    scale *= 2
                # print(f'Scale: {scale}')
                grad_stats.clear()

            if step % opt.accumulation_steps == 0 and not stats["skip_example"]:
                if opt.is_distributed and opt.shard_optim:
                    optimizer.clip_grad_norm(scale * opt.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), scale * opt.clip)

                optimizer.step(scale=scale)
                scheduler.step()
                model.module.zero_grad()
            iter_stats["runtime/model_update"] = (time.time() - model_update_start, 1)
            iter_stats["runtime/train_step"] = (time.time() - train_step_start, 1)
            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3g}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.2g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"
                if tb_logger:
                    tb_logger.add_scalar("lr", scheduler.get_last_lr()[0], step)

                logger.info(log)
                run_stats.reset()

            if step % opt.eval_freq == 0:
                for data_path in opt.eval_data:
                    dataset_name = os.path.basename(data_path)

                    metrics = evaluate(model, atlas_model, opt, data_path, step)
                    log_message = f"Dataset: {dataset_name}"
                    for k, v in metrics.items():
                        log_message += f" | {v:.3f} {k}"
                        if tb_logger:
                            tb_logger.add_scalar(f"{dataset_name}/{k}", v, step)
                    logger.info(log_message)

            if step % opt.save_freq == 0:
                checkpoint = {
                    "step": step,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "opt": opt,
                }
                name = f"step-{step}" + "model.pth.tar"
                path = os.path.join(checkpoint_path)
#                epoch_path = os.path.join(path, name)  # "step-%s" % step)
                fp = os.path.join(path, name)
                torch.save(checkpoint, fp)

            if step > opt.total_steps:
                exit()


if __name__ == "__main__":
    options = get_options()
    opt = set_parser_options(options.parser, sys.argv[1:])

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    atlas_model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt)

    # actual model
    if opt.lca_one_document:
        model = LCAModel(opt)
    elif opt.lca_one_document_generation:
        model = LCAModelDocGen(opt)
    elif opt.lca_multiple_document_generation:
        model = LCAModelAllDocGenSum(opt)
    elif opt.lca_multiple_document_generation_softmax:
        model = LCAModelAllDocGen(opt)
    elif opt.lca_multiple_document_generation_softmax_only:
        model = LCAModelAllDocGenOnlySoftMax(opt)
    elif opt.lca_multiple_document_generation_simplified:
        model = LCAModelAllDocGenSimplified(opt)
    elif opt.lca_baseline_perf_function:
        model = LCABaselinePerfFunction(opt)
    else:
        raise NotImplementedError
    
    optimizer, scheduler, _, _ = set_optim(opt, model)

    if opt.is_distributed:
        atlas_model = torch.nn.parallel.DistributedDataParallel(
            atlas_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=True,
        )
        atlas_model._set_static_graph()

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=True,
        )
        model._set_static_graph()

    logger.info("Start finetuning")
    dist_utils.barrier()
    train(
        model,
        atlas_model,
        optimizer,
        scheduler,
        step,
        opt,
        checkpoint_path,
    )
