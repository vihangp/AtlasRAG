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
def run_retrieval_only(model, index, opt, data_path, step=None):
    model.eval()
    metrics = defaultdict(lambda: [])
    dataset_wpred = []
    unwrapped_model = util.get_unwrapped_model_if_wrapped(model)
    reader_tokenizer = unwrapped_model.reader_tokenizer

    task = get_task(opt, reader_tokenizer)
    data_iterator = _get_eval_data_iterator(opt, data_path, task)

    for i, batch in enumerate(data_iterator):
        query = batch.get("query", [""])
        answers = batch.get("target", [""])
        batch_metadata = batch.get("metadata")
        query_enc = model.retriever_tokenize(query)
        retrieved_passages, sim_scores = unwrapped_model.retrieve(
            index,
            opt.n_context,
            query,
            query_enc["input_ids"].cuda(),
            query_enc["attention_mask"].cuda(),
            batch_metadata=batch_metadata,
            filtering_fun=task.filter,
        )
        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue
        # list of lists of passages

        for k in range(len(retrieved_passages)):
            num_queries_added = 0
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            
            # use all retrieved passages
            ex = {"query": query[k], "answers": gold, "passages": retrieved_passages[k], "nn_scores": sim_scores[k]}
            if batch_metadata is not None:
                ex["metadata"] = batch_metadata[k]
            if "id" in batch:
                ex["id"] = batch["id"][k]
            dataset_wpred.append(ex)
            num_queries_added += 1

            if opt.use_all_passages_only:
                # create a separate entry for each passage
                for p in range(len(retrieved_passages[k])):
                    current_retrieved_passages = [retrieved_passages[k][p]]
                    nn_scores = [sim_scores[k][p]]
                    ex = {"query": query[k], "answers": gold, "passages": current_retrieved_passages, "nn_scores": nn_scores}
                    if batch_metadata is not None:
                        ex["metadata"] = batch_metadata[k]
                    if "id" in batch:
                        ex["id"] = batch["id"][k]
                    dataset_wpred.append(ex)
                    num_queries_added += 1
                
                # drop one passage at a time
                for p in range(len(retrieved_passages[k])):
                    current_retrieved_passages = [s for ind, s in enumerate(retrieved_passages[k]) if ind!=p]
                    nn_scores = [s for ind, s in enumerate(sim_scores[k]) if ind!=p]
                    ex = {"query": query[k], "answers": gold, "passages": current_retrieved_passages, "nn_scores": nn_scores}
                    if batch_metadata is not None:
                        ex["metadata"] = batch_metadata[k]
                    if "id" in batch:
                        ex["id"] = batch["id"][k]
                    dataset_wpred.append(ex)
                    num_queries_added += 1

                # drop two passages at a time
                for p in range(len(retrieved_passages[k])):
                    for q in range(p+1, len(retrieved_passages[k])):
                        current_retrieved_passages = [s for ind, s in enumerate(retrieved_passages[k]) if ind!=p and ind!=q]
                        nn_scores = [s for ind, s in enumerate(sim_scores[k]) if ind!=p and ind!=q]
                        ex = {"query": query[k], "answers": gold, "passages": current_retrieved_passages, "nn_scores": nn_scores}
                        if batch_metadata is not None:
                            ex["metadata"] = batch_metadata[k]
                        if "id" in batch:
                            ex["id"] = batch["id"][k]
                        dataset_wpred.append(ex)
                        num_queries_added += 1
                
                # drop three passages at a time
                for p in range(len(retrieved_passages[k])):
                    for q in range(p+1, len(retrieved_passages[k])):
                        for r in range(q+1, len(retrieved_passages[k])):
                            current_retrieved_passages = [s for ind, s in enumerate(retrieved_passages[k]) if ind!=p and ind!=q and ind!=r]
                            nn_scores = [s for ind, s in enumerate(sim_scores[k]) if ind!=p and ind!=q and ind!=r]
                            ex = {"query": query[k], "answers": gold, "passages": current_retrieved_passages, "nn_scores": nn_scores}
                            if batch_metadata is not None:
                                ex["metadata"] = batch_metadata[k]
                            if "id" in batch:
                                ex["id"] = batch["id"][k]
                            dataset_wpred.append(ex)
                            num_queries_added += 1
            

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
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

    index, passages = load_or_initialize_index(opt)
    model, _, _, _, _, opt, step = load_or_initialize_atlas_model(opt, eval_only=True)

    logger.info("Start Evaluation")
    dist_utils.barrier()

    if not opt.use_file_passages and opt.load_index_path is None:
        indexing_start = time.time()
        model.build_index(index, passages, opt.per_gpu_embedder_batch_size, logger)

        if opt.save_index_path is not None:
            save_embeddings_and_index(index, opt)

    for data_path in opt.eval_data:
        dataset_name = os.path.basename(data_path)
        logger.info(f"Start Evaluation on {data_path}")

        run_retrieval_only(model, index, opt, data_path, step)








