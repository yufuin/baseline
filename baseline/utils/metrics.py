def compute_precision_recall_f(true_positive, real_positive, pred_positive, prefix=None, eps=1e-10):
    """
    true_positive, real_positive, pred_positive : int
    prefix : str (default="").

    output : {'precision':precision, 'recall':recall, 'f-score':f_score}.
    if prefix is given, keys are with prefix. e.g., if prefix=='foo/', then key for precision is 'foo/precision'
    """
    if prefix is None:
        prefix = ""
    output = dict()
    output[prefix+"precision"] = precision = true_positive / max(1, pred_positive)
    output[prefix+"recall"] = recall = true_positive / max(1, real_positive)
    output[prefix+"f-score"] = f_score = 2*precision*recall / max(eps, precision+recall)
    return output
