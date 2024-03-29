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
EVAL_FREQ: str = "10"
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

class LCAModelAllDocGen(torch.nn.Module):
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

        # added for softmax
        self.soft_max_preds = Linear(self.embed_dim, 1, bias=False).cuda()        
    
    def forward(self, encoder_last_hidden_state, gen_tokens_embeddings):

        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context , 512, 768)
        gen_tokens_embeddings = torch.repeat_interleave(gen_tokens_embeddings, repeats=self.opt.n_context, dim=0)
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

        att_input = encoder_last_hidden_state.view(encoder_last_hidden_state.size(0) * self.opt.n_context , 512, 768)
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
    data_iterator = list(task.batch_iterator(data_iterator, opt.per_gpu_batch_size))

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
            output = unwrapped_model.reader(
                input_ids=reader_tokens["input_ids"].cuda().view(reader_tokens["input_ids"].size(0), -1),
                attention_mask=reader_tokens["attention_mask"].cuda().view(reader_tokens["attention_mask"].size(0), -1),
                decoder_input_ids=decoder_input_ids.cuda(),
                use_cache=False,
                output_hidden_states = True
                )
            
            # encoder_last_hidden_state size: batch_size, num_passages * 512, 768
            encoder_last_hidden_state = output["encoder_last_hidden_state"]        

        if opt.lca_one_document:
            prediction = model(encoder_last_hidden_state)
        elif opt.lca_one_document_generation:
            prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)
        elif opt.lca_multiple_document_generation:            
            prediction, prediction_passages = model(encoder_last_hidden_state, gen_tokens_embeddings) 
        elif opt.lca_multiple_document_generation_softmax:
            prediction, prediction_passages, softmax_passages = model(encoder_last_hidden_state, gen_tokens_embeddings)           
        elif opt.lca_baseline_perf_function:
            prediction = model(encoder_last_hidden_state, gen_tokens_embeddings)
        else:
            raise NotImplementedError
        
        if f1_labels.size()[0] < opt.per_gpu_batch_size or prediction.size()[0] < opt.per_gpu_batch_size:
            continue

        eval_loss = torch.nn.functional.mse_loss(prediction.unsqueeze(dim=1), f1_labels.unsqueeze(dim=1))

        metrics["eval_loss"].append(eval_loss.cpu())

        prediction = prediction.unsqueeze(dim=1).cpu().numpy().tolist()
        if opt.lca_multiple_document_generation or opt.lca_multiple_document_generation_softmax:
            prediction_passages = prediction_passages.cpu().numpy().tolist()
            if opt.lca_multiple_document_generation_softmax:
                softmax_passages = softmax_passages.cpu().numpy().tolist()        
#        print(prediction.shape, f1_labels.unsqueeze(dim=1).shape, len(generations))

        for k in range(len(generations)):
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]

            ex = {"query": query[k], "answers": gold, "generation": generations[k], "scores": scores[k], "f1_pred": prediction[k]}
            if not opt.dont_write_passages:
                ex["passages"] = retrieved_passages[k]
            if "id" in batch:
                ex["id"] = batch["id"][k]
            if opt.lca_multiple_document_generation or opt.lca_multiple_document_generation_softmax:
                ex["f1_pred_passages"] = prediction_passages[k]
            if opt.lca_multiple_document_generation_softmax:
                ex["softmax_passages"] = softmax_passages[k]

            dataset_wpred.append(ex)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics



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
    elif opt.lca_baseline_perf_function:
        model = LCABaselinePerfFunction(opt)        
    else:
        raise NotImplementedError

    optimizer, scheduler, _, _ = set_optim(opt, model)

    checkpoint = torch.load("/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_baseline_perf_function_20231129-211522/step-1250model.pth.tar")
    model.load_state_dict(checkpoint["model"])
    step = checkpoint["step"]
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

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")

        metrics = evaluate(model, atlas_model, opt, data_path, step)
        log_message = f"Dataset: {dataset_name}"
        for k, v in metrics.items():
            log_message += f" | {v:.3f} {k}"
        logger.info(log_message)    

