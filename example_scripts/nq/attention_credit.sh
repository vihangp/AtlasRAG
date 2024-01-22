#!/bin/bash

# size types: base, large, xl, xxl
for size in large
do 

    DATA_DIR='/data/projects/monet/atlas'
    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_retrieve_gradient'
#    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_retrieve_robustness'

    port=$(shuf -i 15000-16000 -n 1)
    EVAL_FILES="${RETRIEVE_DIR}/nq_test-step-0.jsonl"
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
    EXPERIMENT_NAME="${size}_t5_model_lm_${FINETUNED}_attention_credit"
    PRECISION="fp32" # "bf16"

    CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 attention_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "adist" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --eval_data ${EVAL_FILES} --per_gpu_batch_size 24 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0
# /data/projects/monet/atlas/experiments/large_t5_model_lm_TRUE_attention_credit/nq_test-step-0-step-0-eval.jsonl
# /data/projects/monet/atlas/experiments/large_t5_model_lm_TRUE_lca_multiple_document_generation_softmax_20231214-111236/nq_test-step-0-step-0-eval-step-1000.jsonl
done