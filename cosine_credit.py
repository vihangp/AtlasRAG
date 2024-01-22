

# get the input_id embeddings using the encoder.ember_tokens layer
# use torch.autograd.grad to compute the gradient of the loss with respect to the input_id embeddings
# take the product of the embedding and the gradient
# take the norm of the gradients
# normalize the norms over all the tokens in the input_ids
# sum the normalized norms over passages to get the final passage credit score

import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist

from src import dist_utils, slurm, util
from src.index_io import load_or_initialize_index, save_embeddings_and_index
from src.model_io import create_checkpoint_directories, load_or_initialize_atlas_model
from src.options import get_options
from src.tasks import get_task

os.environ["TOKENIZERS_PARALLELISM"] = "true"


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
def evaluate(model, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        unwrapped_model.reader.zero_grad()

        query_ = batch.get("query", [""])
        answers = batch.get("target", [""])
        nn_scores = batch.get("nn_scores")
        batch_metadata = batch.get("metadata")
        target_tokens = batch.get("target_tokens")
        generation = batch.get("generation")
        scores = batch.get("scores")
        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query_, answers, target_tokens=target_tokens)

        query = []
        for q, a in zip(query_, generation):
            query.append(q[:-12] + a)

        assert "passages" in batch, "cant use use_file_passages mode without passing in passages"
#        retrieved_passages = [p[: opt.n_context] for p in batch["passages"]]
        retrieved_passages = []
        for p in batch["passages"]:
            _passages = []
            for _p in p:
                _passages.append(_p["text"])
            retrieved_passages.append(_passages)

        query_enc = model.retriever_tokenize(query)
        passage_tokens = []
        for passages in retrieved_passages:
            passage_tokens.append(model.retriever_tokenize(passages))
        
        # query_emb = self.retriever(query_ids_retriever, query_mask_retriever, is_passages=False)
        query_emb = model.retriever(query_enc["input_ids"], query_enc["attention_mask"], is_passages=False)

        passage_emb = []
        for passages in passage_tokens:
            passage_emb.append(model.retriever(passages["input_ids"], passages["attention_mask"], is_passages=True))
        
        # compute cosine similarity between query and passages
        # query_emb: (batch_size, 768)
        # passage_emb: (batch_size, num_passages, 768)
        # cosine_sim: (batch_size, num_passages)
        cosine_sim = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(1), torch.stack(passage_emb), dim=-1)

        # normalize the cosine similarity scores
        cosine_sim_normlz = cosine_sim / torch.sum(cosine_sim, dim=-1).unsqueeze(-1)
        cosine_sim_normlz = cosine_sim_normlz.detach().cpu().numpy()

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        
        # copied from atlas generate function

        for k, g in enumerate(generation):
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            
            if opt.write_results:
                ex = {"query": query_[k], "answers": gold, "generation": g, "scores": scores[k], "nn_scores": nn_scores[k], "cosine_credit": cosine_sim_normlz[k].tolist()}
                if not opt.dont_write_passages:
                    ex["passages"] = batch["passages"][k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    metrics, dataset_wpred = task.evaluation_postprocessing(metrics, dataset_wpred)
    metrics = util.avg_dist_dict(task.metrics, metrics)
    metrics = {key: value if key == "eval_loss" else 100 * value for key, value in metrics.items()}
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}-eval"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics        

if __name__ == "__main__":
    options = get_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()

    checkpoint_path, saved_index_path = create_checkpoint_directories(opt)

    logger = util.init_logger(opt.is_main, opt.is_distributed, os.path.join(checkpoint_path, "run.log"))
    if opt.is_main:
        options.print_options(opt)

    logger.info(f"world size: {dist_utils.get_world_size()}")

    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)

    logger.info("Start Evaluation")
    dist_utils.barrier()
    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")

        metrics = evaluate(model, opt, data_path, step)










