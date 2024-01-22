import json
import numpy as np


def advantage(data):
    """
    Advantage is defined as : credit_i = score_i - Expected_score
    Where, Expected_score = nn_score_1 * score_1 + ...+ nn_score_k * score_k
    The nn_scores are normalized to sum to 1 and are the nearest neighbour scores
    from retrieval.
    """
    epsilon = 0.05
    for i in range(len(data)):
        
        # assuming that the nn_scores are already greater than 0
        # normalize the nn_scoress to sum to 1
        closed_book_epsilon = epsilon * sum(data[i]['nn_scores'])
        sum_nn_scores = sum(data[i]['nn_scores']) + closed_book_epsilon
        for j in range(len(data[i]['nn_scores'])):
            data[i]['nn_scores'][j] = data[i]['nn_scores'][j] / sum_nn_scores
        closed_book_epsilon = closed_book_epsilon / sum_nn_scores

        # expected score for the model
        expected_score_em = 0
        expected_score_f1 = 0
        for j in range(len(data[i]['nn_scores'])):
            expected_score_em += data[i]['nn_scores'][j] * data[i]['scores'][j + 1]['exact_match']
            expected_score_f1 += data[i]['nn_scores'][j] * data[i]['scores'][j + 1]['f1']
        
        expected_score_em += closed_book_epsilon * data[i]['scores'][0]['exact_match']
        expected_score_f1 += closed_book_epsilon * data[i]['scores'][0]['f1']


        for j in range(len(data[i]['scores'])):
            # compare the closed book score with top k passage scores
            passage_exact_match = data[i]['scores'][j]['exact_match']
            passage_f1 = data[i]['scores'][j]['f1']

            passage_credit_em = passage_exact_match - expected_score_em + 1e-8
            passage_credit_f1 = passage_f1 - expected_score_f1 + 1e-8

            data[i]['scores'][j]['advantage_credit_em'] = passage_credit_em  
            data[i]['scores'][j]['advantage_credit_f1'] = passage_credit_f1


    return data


# closed book comparison
def closed_book(data):
    """
    Compare the closed book scores with scores from top k passages.
    """
    for i in range(len(data)):
        # 0th index is the closed book score
        closed_book_exact_match = data[i]['scores'][0]['exact_match']
        closed_book_f1 = data[i]['scores'][0]['f1']

        all_passage_credit_em = []
        all_passage_credit_f1 = []

        for j in range(len(data[i]['scores_one'])):
            # compare the closed book score with top k passage scores
            passage_exact_match = data[i]['scores_one'][j]['exact_match']
            passage_f1 = data[i]['scores_one'][j]['f1']

            passage_credit_em = passage_exact_match - closed_book_exact_match + 1e-8
            passage_credit_f1 = passage_f1 - closed_book_f1 + 1e-8

            data[i]['scores_one'][j]['passage_credit_em_koi'] = passage_credit_em  
            data[i]['scores_one'][j]['passage_credit_f1_koi'] = passage_credit_f1 

            all_passage_credit_em.append(data[i]['scores_one'][j]['passage_credit_em_koi'])
            all_passage_credit_f1.append(data[i]['scores_one'][j]['passage_credit_f1_koi'])

        # make values >= 0
        min_passage_credit_em = min(all_passage_credit_em) 
        min_passage_credit_f1 = min(all_passage_credit_f1)

        sum_em_scores = 0
        sum_f1_scores = 0
        for j in range(len(data[i]['scores_one'])):
            data[i]['scores_one'][j]['passage_credit_em_koi'] = data[i]['scores_one'][j]['passage_credit_em_koi'] - min_passage_credit_em
            data[i]['scores_one'][j]['passage_credit_f1_koi'] = data[i]['scores_one'][j]['passage_credit_f1_koi'] - min_passage_credit_f1

            sum_em_scores += data[i]['scores_one'][j]['passage_credit_em_koi'] + 1e-8
            sum_f1_scores += data[i]['scores_one'][j]['passage_credit_f1_koi'] + 1e-8
        
        # sum of passage credit should be 1
        for j in range(len(data[i]['scores_one'])):
            data[i]['scores_one'][j]['passage_credit_em_koi'] = data[i]['scores_one'][j]['passage_credit_em_koi'] / sum_em_scores
            data[i]['scores_one'][j]['passage_credit_f1_koi'] = data[i]['scores_one'][j]['passage_credit_f1_koi'] / sum_f1_scores

    return data

def input_reduction(data):
    """
    Compare the scores when a document is dropped from the input. 
    """
    for i in range(len(data)):
        # 0th index is the closed book score
        closed_book_exact_match = data[i]['scores'][0]['exact_match']
        closed_book_f1 = data[i]['scores'][0]['f1']

        data[i]['scores'][0]['passage_credit_em'] = closed_book_exact_match  
        data[i]['scores'][0]['passage_credit_f1'] = closed_book_f1

        all_passage_credit_em = []
        all_passage_credit_f1 = []

        for j in range(1, len(data[i]['scores'])):
            # compare the closed book score with top k passage scores
            passage_exact_match = data[i]['scores'][j]['exact_match']
            passage_f1 = data[i]['scores'][j]['f1']

            inclusion_exact_match = 0
            inclusion_f1 = 0
            for k in range(1, len(data[i]['scores'])):
                if k != j:
                    inclusion_exact_match += data[i]['scores'][k]['exact_match']
                    inclusion_f1 += data[i]['scores'][k]['f1']

            inclusion_exact_match = inclusion_exact_match / (len(data[i]['scores']) - 2)
            inclusion_f1 = inclusion_f1 / (len(data[i]['scores']) - 2)

            # credit = average_score_when_included - score_when_dropped
            passage_credit_em = inclusion_exact_match -  passage_exact_match + 1e-8 + data[i]['scores_one'][j-1]['exact_match']
            passage_credit_f1 = inclusion_f1 - passage_f1 + 1e-8 + data[i]['scores_one'][j-1]['f1']
            
            data[i]['scores'][j]['passage_credit_em'] = passage_credit_em  
            data[i]['scores'][j]['passage_credit_f1'] = passage_credit_f1 

            all_passage_credit_em.append(data[i]['scores'][j]['passage_credit_em'])
            all_passage_credit_f1.append(data[i]['scores'][j]['passage_credit_f1'])

        # make values >= 0
        min_passage_credit_em = min(all_passage_credit_em) 
        min_passage_credit_f1 = min(all_passage_credit_f1)

        sum_em_scores = 0
        sum_f1_scores = 0
        for j in range(1, len(data[i]['scores'])):
            data[i]['scores'][j]['passage_credit_em'] = data[i]['scores'][j]['passage_credit_em'] - min_passage_credit_em
            data[i]['scores'][j]['passage_credit_f1'] = data[i]['scores'][j]['passage_credit_f1'] - min_passage_credit_f1

            sum_em_scores += data[i]['scores'][j]['passage_credit_em'] + 1e-8
            sum_f1_scores += data[i]['scores'][j]['passage_credit_f1'] + 1e-8
        
        # sum of passage credit should be 1
        for j in range(1, len(data[i]['scores'])):
            data[i]['scores'][j]['passage_credit_em'] = data[i]['scores'][j]['passage_credit_em'] / sum_em_scores
            data[i]['scores'][j]['passage_credit_f1'] = data[i]['scores'][j]['passage_credit_f1'] / sum_f1_scores

    return data

def gradient(data):
    """
    Credit is given by norm of the product of the input and the gradient of the output wrt input. Further, normalized to sum to 1. 
    """
    for i in range(len(data)):
        # 0th index is the closed book score
        closed_book_exact_match = data[i]['scores'][0]['exact_match']
        closed_book_f1 = data[i]['scores'][0]['f1']

        data[i]['scores'][0]['passage_credit_em'] = closed_book_exact_match  
        data[i]['scores'][0]['passage_credit_f1'] = closed_book_f1

        all_passage_credit_em = []
        all_passage_credit_f1 = []

        for j in range(1, len(data[i]['scores'])):
            # compare the closed book score with top k passage scores
            passage_exact_match = data[i]['scores'][j]['exact_match']
            passage_f1 = data[i]['scores'][j]['f1']

            inclusion_exact_match = 0
            inclusion_f1 = 0
            for k in range(1, len(data[i]['scores'])):
                if k != j:
                    inclusion_exact_match += data[i]['scores'][k]['exact_match']
                    inclusion_f1 += data[i]['scores'][k]['f1']

            inclusion_exact_match = inclusion_exact_match / (len(data[i]['scores']) - 2)
            inclusion_f1 = inclusion_f1 / (len(data[i]['scores']) - 2)

            # credit = average_score_when_included - score_when_dropped
            passage_credit_em = inclusion_exact_match -  passage_exact_match + 1e-8
            passage_credit_f1 = inclusion_f1 - passage_f1 + 1e-8
            
            data[i]['scores'][j]['passage_credit_em'] = passage_credit_em  
            data[i]['scores'][j]['passage_credit_f1'] = passage_credit_f1 

            all_passage_credit_em.append(data[i]['scores'][j]['passage_credit_em'])
            all_passage_credit_f1.append(data[i]['scores'][j]['passage_credit_f1'])

        # make values >= 0
        min_passage_credit_em = min(all_passage_credit_em) 
        min_passage_credit_f1 = min(all_passage_credit_f1)

        sum_em_scores = 0
        sum_f1_scores = 0
        for j in range(1, len(data[i]['scores'])):
            data[i]['scores'][j]['passage_credit_em'] = data[i]['scores'][j]['passage_credit_em'] - min_passage_credit_em
            data[i]['scores'][j]['passage_credit_f1'] = data[i]['scores'][j]['passage_credit_f1'] - min_passage_credit_f1

            sum_em_scores += data[i]['scores'][j]['passage_credit_em'] + 1e-8
            sum_f1_scores += data[i]['scores'][j]['passage_credit_f1'] + 1e-8
        
        # sum of passage credit should be 1
        for j in range(1, len(data[i]['scores'])):
            data[i]['scores'][j]['passage_credit_em'] = data[i]['scores'][j]['passage_credit_em'] / sum_em_scores
            data[i]['scores'][j]['passage_credit_f1'] = data[i]['scores'][j]['passage_credit_f1'] / sum_f1_scores

    return data

def get_credit_from_path(data_dir, data_file, dataset_name="nq_data", model_name="base_ft", type="credit"):
    # load jsonl file
    with open(data_dir + data_file, "r") as f:
        data = f.readlines()
    
    # extract dict from jsonl
    data = [json.loads(line) for line in data]
    # keys: query, answers, generation, scores, passages, nn_scores, metadata

    if type == "credit":
        data = closed_book(data)
        data = advantage(data)
    elif type == "input_reduction":
        data = input_reduction(data)
        data = closed_book(data)
    elif type == "gradient":
        data = gradient(data)

    if type == "credit":
        # sum passage credits across all queries, for each index in top k
        all_em_scores_sum_k = [0 for i in range(len(data[0]['scores']))]
        all_f1_scores_sum_k = [0 for i in range(len(data[0]['scores']))]

        all_em_scores_sum_adv = [0 for i in range(len(data[0]['scores']))]
        all_f1_scores_sum_adv = [0 for i in range(len(data[0]['scores']))]

        for i in range(len(data)):
            for j in range(len(data[i]['scores'])):
                all_em_scores_sum_k[j] += data[i]['scores'][j]['passage_credit_em']
                all_f1_scores_sum_k[j] += data[i]['scores'][j]['passage_credit_f1']
                
                all_em_scores_sum_adv[j] += data[i]['scores'][j]['advantage_credit_em']
                all_f1_scores_sum_adv[j] += data[i]['scores'][j]['advantage_credit_f1']
            
        # print the average passage credit for each index in top k
        print(dataset_name, model_name, "Average closed book credit em for each index in top k:", np.array(all_em_scores_sum_k)/ len(data))
        print(dataset_name, model_name, "Average closed book credit f1 for each index in top k:", np.array(all_f1_scores_sum_k)/ len(data))

        print(dataset_name, model_name, "Average advantage credit em for each index in top k:", np.array(all_em_scores_sum_adv)/ len(data))
        print(dataset_name, model_name, "Average advantage credit f1 for each index in top k:", np.array(all_f1_scores_sum_adv)/ len(data))
    
    elif type == "input_reduction":
        all_em_scores_sum_k = [0 for i in range(1, len(data[0]['scores']))]
        all_f1_scores_sum_k = [0 for i in range(1, len(data[0]['scores']))]

        for i in range(len(data)):
            for j in range(1, len(data[i]['scores'])):
                all_em_scores_sum_k[j-1] += data[i]['scores'][j]['passage_credit_em']
                all_f1_scores_sum_k[j-1] += data[i]['scores'][j]['passage_credit_f1']

        print(dataset_name, model_name, "Average credit em for each index in top k:", np.array(all_em_scores_sum_k)/ len(data))
        print(dataset_name, model_name, "Average credit f1 for each index in top k:", np.array(all_f1_scores_sum_k)/ len(data))



if __name__ == "__main__":
    type = "input_reduction"

    credit_file_loo = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_leave_one_out/nq_test-step-0_input_reduction.jsonl"
    credit_file_gradient = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_gradient_credit/nq_test-step-0-step-0-eval.jsonl"
    credit_file_attention = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_attention_credit/nq_test-step-0-step-0-eval.jsonl"
    credit_file_lca_multi_doc = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_multiple_document_generation_sum_eval/nq_test-step-0-step-0-eval-step-1100.jsonl"
    credit_file_lca_multi_doc_soft = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_multiple_document_generation_softmax_eval/nq_test-step-0-step-0-eval-step-1000.jsonl"
    credit_file_lca_multi_doc_soft = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_generate_lca_multiple_document_generation_normalize_20231114-002045/test_nq_trivia-step-1000.jsonl"    

    with open(credit_file_loo, "r") as f:
        data_loo = f.readlines()
    
    data_loo = [json.loads(line) for line in data_loo]

    with open(credit_file_gradient, "r") as f:
        data_gradient = f.readlines()
    
    data_gradient = [json.loads(line) for line in data_gradient]
    
    with open(credit_file_attention, "r") as f:
        data_attention = f.readlines()
        
    data_attention = [json.loads(line) for line in data_attention]

    with open(credit_file_lca_multi_doc, "r") as f:
        data_lca_multi_doc = f.readlines()
    
    data_lca_multi_doc = [json.loads(line) for line in data_lca_multi_doc]

    with open(credit_file_lca_multi_doc_soft, "r") as f:
        data_lca_multi_doc_soft = f.readlines()
    
    data_lca_multi_doc_soft = [json.loads(line) for line in data_lca_multi_doc_soft]

    all_em_scores_sum_k = [[] for i in range(1, len(data_loo[0]['scores']))]
    all_f1_scores_sum_k = [[] for i in range(1, len(data_loo[0]['scores']))]


    import scipy.stats as ss

    # store rank of passage_credit_f1 scores
    loo_dict = {}
    for i in range(len(data_loo)):
        loo_dict[data_loo[i]['query']] = []
        for j in range(1, len(data_loo[i]['scores'])):
            loo_dict[data_loo[i]['query']].append(data_loo[i]['scores'][j]['passage_credit_f1'])


    gradient_dict = {}
    for i in range(len(data_gradient)):
        for j in range(1, len(data_gradient[i]['scores'])):
            gradient_dict[data_gradient[i]['query']] = data_gradient[i]['gradient_credit']


    attention_dict = {}
    for i in range(len(data_attention)):
        for j in range(1, len(data_attention[i]['scores'])):
            attention_dict[data_attention[i]['query']] = data_attention[i]['attention_credit']

    lca_dict = {}
    for i in range(len(data_lca_multi_doc)):
        lca_dict[data_lca_multi_doc[i]['query']] = data_lca_multi_doc[i]['f1_pred_passages']

    lca_dict_soft = {}
    for i in range(len(data_lca_multi_doc_soft)):
        lca_dict_soft[data_lca_multi_doc_soft[i]['query']] = []
        for j in range(len(data_lca_multi_doc_soft[i]['f1_pred_passages'])):
            lca_dict_soft[data_lca_multi_doc_soft[i]['query']].append(data_lca_multi_doc_soft[i]['f1_pred_passages'][j] * data_lca_multi_doc_soft[i]['softmax_passages'][j])
    
    grad_mrr = []
    att_mrr = []
    lca_mrr = []
    lca_mrr_soft = []

    for key in attention_dict.keys():
        loo_ranks = loo_dict[key]
        gradient_ranks = gradient_dict[key]
        attention_ranks = attention_dict[key]
#        lca_ranks = lca_dict[key]
        lca_ranks_soft = lca_dict_soft[key]

        grad_mrr.extend(np.abs(np.array(loo_ranks) - np.array(gradient_ranks)).tolist())
        att_mrr.extend(np.abs(np.array(loo_ranks) - np.array(attention_ranks)).tolist())
#        lca_mrr.extend(np.abs(np.array(loo_ranks) - np.array(lca_ranks)).tolist())
        lca_mrr_soft.extend(np.abs(np.array(loo_ranks) - np.array(lca_ranks_soft)).tolist())


    print("Gradient MRR:", np.mean(grad_mrr))
    print("Attention MRR:", np.mean(att_mrr))
#    print("LCA MRR:", np.mean(lca_mrr))
    print("LCA MRR Softmax:", np.mean(lca_mrr_soft))

