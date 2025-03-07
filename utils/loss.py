import torch.nn.functional as F

def computeCrossEntropyLoss(query_logits, query_labels):
    """ Calculate the CrossEntropy Loss for query set
    """
    return F.cross_entropy(query_logits, query_labels)