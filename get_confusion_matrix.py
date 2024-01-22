import numpy as np
import json

if __name__ == "__main__":


    lca_single_eval = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_single_eval/nq_dev-step-0-step-0-eval-step-3000.jsonl"

    with open(lca_single_eval, "r") as f:
        data_single_eval = f.readlines()
    
    data_single_eval = [json.loads(d) for d in data_single_eval]

    # f1_pred, exact_match
    all_em_scores = [data_single_eval[i]["scores"]["exact_match"] for i in range(len(data_single_eval))]
    all_f1_scores = [data_single_eval[i]["scores"]["f1"] for i in range(len(data_single_eval))]
    all_f1_preds = [0 if data_single_eval[i]["f1_pred"][0] < 0.5 else 1 for i in range(len(data_single_eval))]

    # compute confusion matrix between f1_pred and exact_match
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_em_scores, all_f1_preds)
    print(cm)

    # get accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(accuracy)

    # mse on f1_pred and f1_score
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(all_f1_scores, all_f1_preds)
    print("mse:", mse)

    lca_one_doc_gen = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_one_document_generation/nq_dev-step-0-step-0-eval-step-960.jsonl"

    with open(lca_one_doc_gen, "r") as f:
        data_one_doc_gen_eval = f.readlines()
    
    data_one_doc_gen_eval = [json.loads(d) for d in data_one_doc_gen_eval]

    # f1_pred, exact_match
    all_em_scores = [data_one_doc_gen_eval[i]["scores"]["exact_match"] for i in range(len(data_one_doc_gen_eval))]
    all_f1_scores = [data_one_doc_gen_eval[i]["scores"]["f1"] for i in range(len(data_one_doc_gen_eval))]
    all_f1_preds = [0 if data_one_doc_gen_eval[i]["f1_pred"][0] < 0.5 else 1 for i in range(len(data_one_doc_gen_eval))]

    # compute confusion matrix between f1_pred and exact_match
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_em_scores, all_f1_preds)
    print(cm)

    # get accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(accuracy)

    # mse on f1_pred and f1_score
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(all_f1_scores, all_f1_preds)
    print("mse:", mse)

    threshold = 0.55

    lca_multi_doc_gen = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_multiple_document_generation_softmax_20231020-001600/dev_nq_trivia-step-1100.jsonl"

    with open(lca_multi_doc_gen, "r") as f:
        data_multi_doc_gen_eval = f.readlines()
    
    data_multi_doc_gen_eval = [json.loads(d) for d in data_multi_doc_gen_eval]

    # f1_pred, exact_match
    all_em_scores = [data_multi_doc_gen_eval[i]["scores"]["exact_match"] for i in range(len(data_multi_doc_gen_eval))]
    all_f1_scores = [data_multi_doc_gen_eval[i]["scores"]["f1"] for i in range(len(data_multi_doc_gen_eval))]
    all_f1_preds = [0 if data_multi_doc_gen_eval[i]["f1_pred"][0] < threshold else 1 for i in range(len(data_multi_doc_gen_eval))]

    # compute confusion matrix between f1_pred and exact_match
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_em_scores, all_f1_preds)
    print(cm)

    # get accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(accuracy)

    # mse on f1_pred and f1_score
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(all_f1_scores, all_f1_preds)
    print("mse:", mse)

    lca_multi_doc_gen_softmax = "/data/projects/monet/atlas/experiments/base_t5_model_lm_TRUE_lca_multiple_document_generation_softmax_20231020-052901/dev_nq_trivia-step-850.jsonl"

    with open(lca_multi_doc_gen_softmax, "r") as f:
        data_multi_doc_gen_softmax_eval = f.readlines()

    data_multi_doc_gen_softmax_eval = [json.loads(d) for d in data_multi_doc_gen_softmax_eval]

    # f1_pred, exact_match
    all_em_scores = [data_multi_doc_gen_softmax_eval[i]["scores"]["exact_match"] for i in range(len(data_multi_doc_gen_softmax_eval))]
    all_f1_scores = [data_multi_doc_gen_softmax_eval[i]["scores"]["f1"] for i in range(len(data_multi_doc_gen_softmax_eval))]
    all_f1_preds = [0 if data_multi_doc_gen_softmax_eval[i]["f1_pred"][0] < threshold else 1 for i in range(len(data_multi_doc_gen_softmax_eval))]

    # compute confusion matrix between f1_pred and exact_match
    # https://en.wikipedia.org/wiki/Confusion_matrix
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_em_scores, all_f1_preds)
    print(cm)

    # get accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(accuracy)

    # mse on f1_pred and f1_score
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(all_f1_scores, all_f1_preds)
    print("mse:", mse)

