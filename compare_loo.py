import json
import numpy as np
import copy
from sklearn.metrics import ndcg_score

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


def compute_ap(loo_scores: list, baseline_ranks: list):

    # loo scores set values greater than 0 to 1
    loo_scores = [1 if score > 0 else 0 for score in loo_scores]

    prod_rel_ranks = [rank * score for rank, score in zip(baseline_ranks, loo_scores)]
    count = 0
    total_precsion = 0
    for prod in prod_rel_ranks:
        if prod > 0:
            count += 1
            total_precsion += count/prod

    total_relevant_documents = sum(loo_scores)

    # compute average precision
    # precision @k = (# relevant documents @k) / k
    # average precision = sum(precision @k) / total_relevant_documents
    if total_relevant_documents == 0:
        ap = 0
    else:
        ap = total_precsion / total_relevant_documents

    return ap

def compute_weighted_ap(loo_scores: list, baseline_ranks: list):

    prod_rel_ranks = [rank * score for rank, score in zip(baseline_ranks, loo_scores)]
    count = 0
    total_precsion = 0
    for prod, rank in zip(prod_rel_ranks, baseline_ranks):
        if prod > 0:
            count += 1
            total_precsion += prod *  count/rank

    total_relevant_documents = sum(loo_scores)

    # compute average precision
    # precision @k = (# relevant documents @k) / k
    # average precision = sum(precision @k) / total_relevant_documents
    if total_relevant_documents == 0:
        ap = 0
    else:
        ap = total_precsion / total_relevant_documents

    return ap

def compute_mrr(loo_ranks, baseline_ranks):
    copy_loo_ranks = copy.deepcopy(loo_ranks)
    mrr = 0
    for i in range(len(loo_ranks)):
        index_min = np.argmin(copy_loo_ranks)
        mrr += 1/baseline_ranks[index_min]
        copy_loo_ranks = 10000
        
    
    return mrr / len(loo_ranks)


if __name__ == "__main__":
    type = "input_reduction"

    credit_file_loo = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_leave_one_out/nq_test-step-0_input_reduction.jsonl"
    credit_file_gradient = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_gradient_credit/nq_test-step-0-step-0-eval.jsonl"
    credit_file_attention = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_attention_credit/nq_test-step-0-step-0-eval.jsonl"
    credit_file_lca_multi_doc = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_multiple_document_generation_sum_eval/nq_test-step-0-step-0-eval-step-1100.jsonl"
    credit_file_lca_multi_doc_soft = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_multiple_document_generation_softmax_eval/nq_test-step-0-step-0-eval-step-1000.jsonl"
    credit_file_cosine = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_cosine_credit_20231207-204753/test_nq_trivia-step-0-eval.jsonl"

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

    with open(credit_file_cosine, "r") as f:
        data_cosine = f.readlines()
    
    data_cosine = [json.loads(line) for line in data_cosine]

    all_em_scores_sum_k = [[] for i in range(1, len(data_loo[0]['scores']))]
    all_f1_scores_sum_k = [[] for i in range(1, len(data_loo[0]['scores']))]

    import scipy.stats as ss

    # store rank of passage_credit_f1 scores
    loo_dict = {}
    loo_dict_scores = {}
    for i in range(len(data_loo)):
        temp = []
        for j in range(1, len(data_loo[i]['scores'])):
            temp.append(data_loo[i]['scores'][j]['passage_credit_f1'])
        rank = len(temp) - ss.rankdata(temp, method='max').astype(int) + 1
        loo_dict[data_loo[i]['query']] = rank
        loo_dict_scores[data_loo[i]['query']] = temp


    gradient_dict = {}
    gradient_scores = {}
    for i in range(len(data_gradient)):
        for j in range(1, len(data_gradient[i]['scores'])):
            temp = len(data_gradient[i]['gradient_credit']) - ss.rankdata(np.array(data_gradient[i]['gradient_credit']) * data_gradient[i]['scores']['f1'], method='max').astype(int) + 1
            gradient_dict[data_gradient[i]['query']] = temp
            gradient_scores[data_gradient[i]['query']] = np.array(data_gradient[i]['gradient_credit']) * data_gradient[i]['scores']['f1']

    
    cosine_dict = {}
    cosine_scores = {}
    for i in range(len(data_cosine)):
        for j in range(1, len(data_cosine[i]['scores'])):
            temp = len(data_cosine[i]['cosine_credit']) - ss.rankdata(np.array(data_cosine[i]['cosine_credit']) * data_cosine[i]['scores']['f1'], method='max').astype(int) + 1
            cosine_dict[data_cosine[i]['query']] = temp
            cosine_scores[data_cosine[i]['query']] = np.array(data_cosine[i]['cosine_credit']) * data_cosine[i]['scores']['f1']
    
    attention_dict = {}
    attention_scores = {}
    for i in range(len(data_attention)):
        for j in range(1, len(data_attention[i]['scores'])):
            temp = len(data_attention[i]['attention_credit']) - ss.rankdata(np.array(data_attention[i]['attention_credit']) * data_attention[i]['scores']['f1'], method='max').astype(int) + 1
            attention_dict[data_attention[i]['query']] = temp
            attention_scores[data_attention[i]['query']] = np.array(data_attention[i]['attention_credit']) * data_attention[i]['scores']['f1']

    lca_dict = {}
    lca_scores = {}
    for i in range(len(data_lca_multi_doc)):
        temp = len(data_lca_multi_doc[i]['f1_pred_passages']) - ss.rankdata(data_lca_multi_doc[i]['f1_pred_passages'], method='max').astype(int) + 1 
        lca_dict[data_lca_multi_doc[i]['query']] = temp
        lca_scores[data_lca_multi_doc[i]['query']] = data_lca_multi_doc[i]['f1_pred_passages']


    lca_dict_soft = {}
    lca_scores_soft = {}
    for i in range(len(data_lca_multi_doc_soft)):
        temp = []
        lca_scores_soft[data_lca_multi_doc_soft[i]['query']] = []
        for j in range(len(data_lca_multi_doc_soft[i]['f1_pred_passages'])):
            temp.append(data_lca_multi_doc_soft[i]['f1_pred_passages'][j] * data_lca_multi_doc_soft[i]['softmax_passages'][j])
            lca_scores_soft[data_lca_multi_doc_soft[i]['query']].append(data_lca_multi_doc_soft[i]['f1_pred_passages'][j] * data_lca_multi_doc_soft[i]['softmax_passages'][j])
        
        rank = len(temp) - ss.rankdata(temp, method='max').astype(int) + 1
        lca_dict_soft[data_lca_multi_doc_soft[i]['query']] = rank
        
    
    grad_mrr = []
    att_mrr = []
    lca_mrr = []
    lca_mrr_soft = []
    cos_mrr = []

    grad_map = []
    att_map = []
    lca_map = []
    lca_map_soft = []
    cos_map = []

    grad_wmap = []
    att_wmap = []
    lca_wmap = []
    lca_wmap_soft = []
    cos_wmap = []

    grad_ndcg = []
    att_ndcg = []
    lca_ndcg = []
    lca_ndcg_soft = []
    cos_ndcg = []

    grad_mae = []
    att_mae = []
    lca_mae = []
    lca_mae_soft = []
    cos_mae = []

    for key in lca_dict.keys():
        loo_ranks = loo_dict[key]
        gradient_ranks = gradient_dict[key]
        attention_ranks = attention_dict[key]
        lca_ranks = lca_dict[key]
        lca_ranks_soft = lca_dict_soft[key]
        cos_ranks = cosine_dict[key]

        grad_mrr.append(compute_mrr(loo_ranks, gradient_ranks))
        att_mrr.append(compute_mrr(loo_ranks, attention_ranks))
        lca_mrr.append(compute_mrr(loo_ranks, lca_ranks))
        lca_mrr_soft.append(compute_mrr(loo_ranks, lca_ranks_soft))
        cos_mrr.append(compute_mrr(loo_ranks, cos_ranks))

        loo_scores = loo_dict_scores[key]
        
        grad_map.append(compute_ap(loo_scores, gradient_ranks))
        att_map.append(compute_ap(loo_scores, attention_ranks))
        lca_map.append(compute_ap(loo_scores, lca_ranks))
        lca_map_soft.append(compute_ap(loo_scores, lca_ranks_soft))
        cos_map.append(compute_ap(loo_scores, cos_ranks))

        grad_wmap.append(compute_weighted_ap(loo_scores, gradient_ranks))
        att_wmap.append(compute_weighted_ap(loo_scores, attention_ranks))
        lca_wmap.append(compute_weighted_ap(loo_scores, lca_ranks))
        lca_wmap_soft.append(compute_weighted_ap(loo_scores, lca_ranks_soft))
        cos_wmap.append(compute_weighted_ap(loo_scores, cos_ranks))


        grad_ndcg.append(ndcg_score(np.expand_dims(np.array(loo_scores), 0), np.expand_dims(np.array(gradient_scores[key]), 0) ))
        att_ndcg.append(ndcg_score(np.expand_dims(np.array(loo_scores), 0), np.expand_dims(np.array(attention_scores[key]), 0) ))
        lca_ndcg.append(ndcg_score(np.expand_dims(np.array(loo_scores), 0), np.expand_dims(np.array(lca_scores[key]), 0) ))
        lca_ndcg_soft.append(ndcg_score(np.expand_dims(np.array(loo_scores), 0), np.expand_dims(np.array(lca_scores_soft[key]), 0) ))
        cos_ndcg.append(ndcg_score(np.expand_dims(np.array(loo_scores), 0), np.expand_dims(np.array(cosine_scores[key]), 0) ))

        grad_mae.append(np.mean(np.abs(np.array(loo_scores) - np.array(gradient_scores[key]))))
        att_mae.append(np.mean(np.abs(np.array(loo_scores) - np.array(attention_scores[key]))))
        lca_mae.append(np.mean(np.abs(np.array(loo_scores) - np.array(lca_scores[key]))))
        lca_mae_soft.append(np.mean(np.abs(np.array(loo_scores) - np.array(lca_scores_soft[key]))))
        cos_mae.append(np.mean(np.abs(np.array(loo_scores) - np.array(cosine_scores[key]))))




    print("Gradient MRR:", np.mean(grad_mrr))
    print("Attention MRR:", np.mean(att_mrr))
    print("LCA MRR:", np.mean(lca_mrr))
    print("LCA MRR Softmax:", np.mean(lca_mrr_soft))
    print("Cosine MRR:", np.mean(cos_mrr))

    print("Gradient MAP:", np.mean(grad_map))
    print("Attention MAP:", np.mean(att_map))
    print("LCA MAP:", np.mean(lca_map))
    print("LCA MAP Softmax:", np.mean(lca_map_soft))
    print("Cosine MAP:", np.mean(cos_map))

    # print("Gradient WMAP:", np.mean(grad_wmap))
    # print("Attention WMAP:", np.mean(att_wmap))
    # print("LCA WMAP:", np.mean(lca_wmap))
    # print("LCA WMAP Softmax:", np.mean(lca_wmap_soft))
    # print("Cosine WMAP:", np.mean(cos_wmap))

    print("Gradient NDCG:", np.mean(grad_ndcg))
    print("Attention NDCG:", np.mean(att_ndcg))
    print("LCA NDCG:", np.mean(lca_ndcg))
    print("LCA NDCG Softmax:", np.mean(lca_ndcg_soft))
    print("Cosine NDCG:", np.mean(cos_ndcg))

    print("Gradient MAE:", np.mean(grad_mae))
    print("Attention MAE:", np.mean(att_mae))
    print("LCA MAE:", np.mean(lca_mae))
    print("LCA MAE Softmax:", np.mean(lca_mae_soft))
    print("Cosine MAE:", np.mean(cos_mae))
    

    


