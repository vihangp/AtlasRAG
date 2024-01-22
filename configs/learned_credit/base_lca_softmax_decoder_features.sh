#!/bin/bash
FINETUNED="TRUE"
port=$(shuf -i 15000-16000 -n 1)
DATA_DIR='/data/projects/monet/atlas'
SAVE_DIR=${DATA_DIR}/experiments/
size=base
PRETRAINED_MODEL=${DATA_DIR}/models/atlas_nq/${size}
PRETRAINED_INDEX=/data2/projects/monet/atlas/indices/atlas_nq/wiki/${size}

RETRIEVE_DIR=base_t5_model_lm_TRUE_generate_lca_multiple_document_generation
PRECISION="fp32" # "bf16"


# Train on the generated data
TRAIN_FILES="${SAVE_DIR}${RETRIEVE_DIR}/train_nq_trivia.jsonl"
EVAL_FILES="${SAVE_DIR}${RETRIEVE_DIR}/dev_nq_trivia.jsonl ${SAVE_DIR}${RETRIEVE_DIR}/test_nq_trivia.jsonl"

EXPERIMENT_NAME="${RETRIEVE_DIR}_normalize_$(date '+%Y%m%d-%H%M%S')"
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 13
# /data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_generate_lca_multiple_document_generation_normalize_20231114-002045/run.log
# /data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_generate_lca_multiple_document_generation_normalize_20231114-071838/test_nq_trivia-step-1000.jsonl
# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 13

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 6

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 7

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 8

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 9

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 10

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 11

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 12

# EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
# CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 13
