#!/bin/bash

# ###### Create a query, document set#########


# # size types: base, large, xl, xxl
# for size in base
# do 

#     DATA_DIR='/data/projects/monet/atlas'

#     port=$(shuf -i 15000-16000 -n 1)
# #    EVAL_FILES="${DATA_DIR}/nq_data/train.jsonl ${DATA_DIR}/nq_data/nq_dev.jsonl"
# #    EVAL_FILES="${DATA_DIR}/triviaqa_data/trivia_train.jsonl ${DATA_DIR}/triviaqa_data/triviaqa_dev.jsonl ${DATA_DIR}/triviaqa_data/triviaqa_test.jsonl"
#     EVAL_FILES="${DATA_DIR}/nq_data/nq_test.jsonl"
#     FINETUNED="TRUE"
#     if [[ "${FINETUNED}" == "TRUE" ]]; then
#         PRETRAINED_MODEL=${DATA_DIR}/models/atlas_nq/${size}
#         if [[ "${size}" == "xxl" ]]; then
#             PRETRAINED_INDEX=/data/projects/monet/atlas/indices/atlas_nq/wiki/xxl
#         else
#             PRETRAINED_INDEX=/data2/projects/monet/atlas/indices/atlas_nq/wiki/${size}
#         fi        
#     else
#         PRETRAINED_MODEL=${DATA_DIR}/models/atlas/${size}
#         PRETRAINED_INDEX=${DATA_DIR}/indices/atlas/wiki/${size}
#     fi
#     SAVE_DIR=${DATA_DIR}/experiments/
#     EXPERIMENT_NAME="${size}_t5_model_lm_${FINETUNED}_query_all_doc_set"
#     PRECISION="fp32" # "bf16"

#     CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 generate_query_set.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --eval_data ${EVAL_FILES} --per_gpu_batch_size 8 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa" --write_results --local_rank 0

# done

###### Retrieve and generate #########

# size types: base, large, xl, xxl
for size in large
do 

    DATA_DIR='/data/projects/monet/atlas'
    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_generate_lca_multiple_document_generation'
    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/large_t5_model_lm_TRUE_generate_lca_multiple_document_generation'
    
#    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_retrieve_robustness'
        
#/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_retrieve/nq_test-step-0.jsonl
    port=$(shuf -i 15000-16000 -n 1)
    EVAL_FILES="${RETRIEVE_DIR}/nq_test-step-0-step-0-eval.jsonl"
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
    EXPERIMENT_NAME="${size}_t5_model_lm_${FINETUNED}_cosine_credit"
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_$(date '+%Y%m%d-%H%M%S')"
    PRECISION="fp32" # "bf16"

    CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 cosine_credit.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --eval_data ${EVAL_FILES} --per_gpu_batch_size 6 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0

done