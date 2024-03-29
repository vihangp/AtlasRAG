#!/bin/bash

# size types: base, large, xl, xxl
for size in large
do 

    DATA_DIR='/data/projects/monet/atlas'
    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/large_t5_model_lm_TRUE_generate_lca_multiple_document_generation'
#/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_retrieve/nq_test-step-0.jsonl
    port=$(shuf -i 15000-16000 -n 1)
#    EVAL_FILES="${DATA_DIR}/nq_data/train.jsonl ${DATA_DIR}/nq_data/nq_test.jsonl"
#    EVAL_FILES="${DATA_DIR}/triviaqa_data/trivia_train.jsonl ${DATA_DIR}/triviaqa_data/triviaqa_dev.jsonl ${DATA_DIR}/triviaqa_data/triviaqa_test.jsonl"
#    EVAL_FILES="${RETRIEVE_DIR}/nq_dev-step-0.jsonl ${RETRIEVE_DIR}/train-step-0.jsonl ${RETRIEVE_DIR}/trivia_train-step-0.jsonl"
    TRAIN_FILES="${RETRIEVE_DIR}/train-step-0-step-0-eval.jsonl"
    EVAL_FILES="${RETRIEVE_DIR}/nq_dev-step-0-step-0-eval.jsonl ${RETRIEVE_DIR}/nq_test-step-0-step-0-eval.jsonl" 
#    TRAIN_FILES="${RETRIEVE_DIR}/train_nq_trivia.jsonl"
#    EVAL_FILES="${RETRIEVE_DIR}/dev_nq_trivia.jsonl"
    FINETUNED="TRUE"
    if [[ "${FINETUNED}" == "TRUE" ]]; then
        PRETRAINED_MODEL=${DATA_DIR}/models/atlas_nq/${size}
        if [[ "${size}" == "xxl" ]]; then
            PRETRAINED_INDEX=/data/projects/monet/atlas/indices/atlas_nq/wiki/xxl
        else
            PRETRAINED_INDEX=/data2/projects/monet/atlas/indices/atlas_nq/wiki/${size}
        fi        
    else
        PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
        PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
    fi
    SAVE_DIR=${DATA_DIR}/experiments/
    EXPERIMENT_NAME="${size}_t5_model_lm_${FINETUNED}_lca_multiple_document_generation_softmax"
    # add date and time to experiment name
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_$(date '+%Y%m%d-%H%M%S')"
    PRECISION="fp32" # "bf16"
# n_context changed to 5 for multi document training
    CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 learned_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 28 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 100 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 12 --decoder_features_layer 5

done