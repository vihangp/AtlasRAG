#!/bin/bash
FINETUNED="TRUE"
port=$(shuf -i 15000-16000 -n 1)
DATA_DIR='/data/projects/monet/atlas'
SAVE_DIR=${DATA_DIR}/experiments/
size=large
RETRIEVE_DIR=${size}_t5_model_lm_${FINETUNED}_lca_multiple_document_generation
RETRIEVE_FILES="${DATA_DIR}/nq_data/train.jsonl ${DATA_DIR}/nq_data/nq_dev.jsonl ${DATA_DIR}/nq_data/nq_test.jsonl ${DATA_DIR}/triviaqa_data/trivia_train.jsonl ${DATA_DIR}/triviaqa_data/triviaqa_dev.jsonl ${DATA_DIR}/triviaqa_data/triviaqa_test.jsonl"
PRETRAINED_MODEL=${DATA_DIR}/models/atlas_nq/${size}
PRETRAINED_INDEX=/data2/projects/monet/atlas/indices/atlas_nq/wiki/${size}
EXPERIMENT_NAME="${RETRIEVE_DIR}_$(date '+%Y%m%d-%H%M%S')"
PRECISION="fp32" # "bf16"

# Retrieve documents and store them in a directory
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 generate_query_set_lca.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --eval_data ${RETRIEVE_FILES} --per_gpu_batch_size 8 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa" --write_results --local_rank 0

# Generate the reponse and the performance
EVAL_FILES="${SAVE_DIR}${EXPERIMENT_NAME}/trivia_train-step-0.jsonl ${SAVE_DIR}${EXPERIMENT_NAME}/triviaqa_dev-step-0.jsonl ${SAVE_DIR}${EXPERIMENT_NAME}/triviaqa_test-step-0.jsonl ${SAVE_DIR}${EXPERIMENT_NAME}/nq_dev-step-0.jsonl ${SAVE_DIR}${EXPERIMENT_NAME}/train-step-0.jsonl ${SAVE_DIR}${EXPERIMENT_NAME}/nq_test-step-0.jsonl"
CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 generate.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0

# combine json files
python combine_json_files.py --input_dir "${SAVE_DIR}${EXPERIMENT_NAME}"

# Train on the generated data
TRAIN_FILES="${SAVE_DIR}${EXPERIMENT_NAME}/train_nq_trivia.jsonl"
EVAL_FILES="${SAVE_DIR}${EXPERIMENT_NAME}/dev_nq_trivia.jsonl ${SAVE_DIR}${EXPERIMENT_NAME}/test_nq_trivia.jsonl"

CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 64 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32
