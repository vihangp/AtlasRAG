# advantage of using a passage: Expected Return with retrieved passages - Expected Return without retrieved passages

# Top k passages: k = 5
# We look at assigning credit to the top 5 passages retrieved. We do this by
# 1- comparing the return when we use each of the passages separately to the return when we do generation closed book. (TODO)

# Result: A file with the following format:
# { "query": "", "answers": [], "passages": [], "generation": [], "scores": [] }
# Where, answers is the list of possible correct answers
# passages is the list of passages retrieved
# generation is the list of generated answers, as we use only one passage at a time, this list will have elements = number of passages retrieved
# scores is the list of scores for each generated answer, as we use only one passage at a time, this list will have elements = number of passages retrieved

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
        retrieved_passages, _ = unwrapped_model.retrieve(
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
        for k in range(len(retrieved_passages)):
            if opt.write_results:
                gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
                ex = {"query": query[k], "answers": gold, "passages": retrieved_passages[k]}
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)

    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    return metrics


@torch.no_grad()
def evaluate(model, index, opt, data_path, step=None):
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
        target_tokens = batch.get("target_tokens")
        query_enc, labels, decoder_input_ids = unwrapped_model.tokenize(query, answers, target_tokens=target_tokens)
        if not opt.use_file_passages:
            query_ids_retriever = query_enc["input_ids"].cuda()
            query_mask_retriever = query_enc["attention_mask"].cuda()
            # retrieved passages is a list of documents/passages
            retrieved_passages, sim_scores = unwrapped_model.retrieve(
                index,
                opt.n_context,
                query,
                query_ids_retriever,
                query_mask_retriever,
                batch_metadata=batch_metadata,
                filtering_fun=task.filter,
            )
        else:
            assert "passages" in batch, "cant use use_file_passages mode without passing in passages"
            retrieved_passages = [p[: opt.n_context] for p in batch["passages"]]

        # If example is a padding example then skip step
        if (len(query) == 0) or (len(query[0]) == 0):
            continue

        all_predictions = [[] for _ in retrieved_passages]
        all_answers = [[] for _ in retrieved_passages]
        all_metrics = [[] for _ in retrieved_passages]

        # generate closed book solution
        # get tokens for the query only
        reader_tokens, _ = unwrapped_model.tokenize_passages(query, [[{'id':"", "title":"", "text":""}] for _ in retrieved_passages])

        if "eval_loss" in task.metrics:
            eval_loss, logits = unwrapped_model.compute_reader_loss_and_logits(reader_tokens, decoder_input_ids, labels)
            metrics["eval_loss"].append(eval_loss)

        closed_book_generation = unwrapped_model.generate(
                reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
            )

        for k, g in enumerate(closed_book_generation):
            if opt.decoder_prompt_format is not None:
                query_ids = reader_tokenizer.encode(
                    opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                )
                g = g[len(query_ids) + 1 :]
            pred = reader_tokenizer.decode(g, skip_special_tokens=True)
            gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
            sample_metrics = task.evaluation(pred, gold)

            for key, value in sample_metrics.items():
                metrics[key].append(value)
            all_predictions[k].append(pred)
            all_answers[k].append(gold)
            all_metrics[k].append(sample_metrics)

        # generate open book solutions
        for current_context in range(opt.n_context):
            current_retrieved_passages = [[p for ind, p in enumerate(passages) if ind!=current_context] for passages in retrieved_passages]

            # this function converts the passages to tokens, which can then be given to a LLM for generation
            reader_tokens, _ = unwrapped_model.tokenize_passages(query, current_retrieved_passages)

            if "eval_loss" in task.metrics:
                eval_loss, logits = unwrapped_model.compute_reader_loss_and_logits(reader_tokens, decoder_input_ids, labels)
                metrics["eval_loss"].append(eval_loss)

            generation = unwrapped_model.generate(
                reader_tokens, query, choices=batch["choices"] if "choices" in batch else None
            )

            for k, g in enumerate(generation):
                if opt.decoder_prompt_format is not None:
                    query_ids = reader_tokenizer.encode(
                        opt.decoder_prompt_format.format_map({"query": query[k]}), add_special_tokens=False
                    )
                    g = g[len(query_ids) + 1 :]
                pred = reader_tokenizer.decode(g, skip_special_tokens=True)
                gold = [answers[k]] if not "answers" in batch else batch["answers"][k]
                sample_metrics = task.evaluation(pred, gold)

                for key, value in sample_metrics.items():
                    metrics[key].append(value)
                all_predictions[k].append(pred)
                all_answers[k].append(gold)
                all_metrics[k].append(sample_metrics)
        
        if opt.write_results:
            # this is looping over the batch
            for k in range(len(retrieved_passages)):
                ex = {"query": query[k], "answers": all_answers[k], "generation": all_predictions[k], "scores": all_metrics[k], "nn_scores": sim_scores[k]}
                if not opt.dont_write_passages:
                    ex["passages"] = retrieved_passages[k]
                if batch_metadata is not None:
                    ex["metadata"] = batch_metadata[k]
                if opt.task == "multiple_choice":
                    # TODO: correct this for multiple choice questions
                    ex["choice_logits"] = task.get_choice_logits(logits[k])
                if "id" in batch:
                    ex["id"] = batch["id"][k]
                dataset_wpred.append(ex)
        
    if opt.write_results:
        dataset_name, _ = os.path.splitext(os.path.basename(data_path))
        dataset_name = f"{dataset_name}-step-{step}"
        util.save_distributed_dataset(dataset_wpred, dataset_name, opt)

    # the returned metrics here are n_context = 1
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
        if opt.retrieve_only:
            run_retrieval_only(model, index, opt, data_path, step)
        else:
            metrics = evaluate(model, index, opt, data_path, step)