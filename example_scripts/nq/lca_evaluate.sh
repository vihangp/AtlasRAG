#!/bin/bash

# size types: base, large, xl, xxl
for size in base
do 

    DATA_DIR='/data/projects/monet/atlas'
    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_generate_lca_multiple_document_generation'
#    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_retrieve_robustness'
#    RETRIEVE_DIR='/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_attention_credit_robustness'    

    port=$(shuf -i 15000-16000 -n 1)

    EVAL_FILES="${RETRIEVE_DIR}/test_nq_trivia-step-0-eval_attention.jsonl" 
    EVAL_FILES="${RETRIEVE_DIR}/test_nq_trivia.jsonl"
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
    EXPERIMENT_NAME="base_t5_model_lm_TRUE_generate_lca_multiple_document_generation_softmax_robustness"
    EXPERIMENT_NAME="base_t5_model_lm_TRUE_generate_lca_multiple_document_generation_performance"
    PRECISION="fp32" # "bf16"

    #CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 evaluate_lca.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --train_data ${TRAIN_FILES} --eval_data ${EVAL_FILES} --per_gpu_batch_size 32 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 20000 --save_freq 1000 --lca_multiple_document_generation_softmax --decoder_features_layer 4
#    CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 evaluate_lca.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --eval_data ${EVAL_FILES} --per_gpu_batch_size 32 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 1000 --save_freq 50 --lca_multiple_document_generation_softmax --per_gpu_batch_size_eval 32 --decoder_features_layer 1
    CUDA_LAUNCH_BLOCKING=1 python -m torch.distributed.launch --nproc_per_node 8 evaluate_lca.py --name ${EXPERIMENT_NAME} --generation_max_length 32 --target_maxlength 32 --gold_score_mode "ppmean" --precision ${PRECISION} --reader_model_type google/t5-${size}-lm-adapt --text_maxlength 512 --target_maxlength 16 --model_path ${PRETRAINED_MODEL} --load_index_path ${PRETRAINED_INDEX} --eval_data ${EVAL_FILES} --per_gpu_batch_size 12 --n_context 5 --retriever_n_context 40 --checkpoint_dir ${SAVE_DIR} --index_mode "flat" --task "qa_retrieved" --qa_prompt_format "{question}" --write_results --local_rank 0 --total_steps 5000 --save_freq 50 --lca_baseline_perf_function --per_gpu_batch_size_eval 32 --decoder_features_layer 1    

done