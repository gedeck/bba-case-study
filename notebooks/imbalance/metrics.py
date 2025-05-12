from scipy import stats
import numpy as np
from imbalance_degree import imbalance_degree as imb


def entropy(groups: list[int]) -> float:
    """ Use Shannon entropy to measure group imbalance """
    total = sum(groups)
    return stats.entropy([group / total for group in groups]) / np.log(len(groups))


def relative_difference(groups: list[int]) -> float:
    """ https://stats.stackexchange.com/a/532719 """
    return (np.max(groups) - np.min(groups)) / (np.sum(groups) - len(groups))


def L1_distance(groups: list[int]) -> float:
    """ L1 norm from all groups maximum """
    max_group = np.max(groups)
    return np.linalg.norm([max_group - group for group in groups], ord=1) / len(groups)


def mean_replica(groups: list[int]) -> float:
    """ Mean of the number of replicas """
    return np.mean(groups)


def imbalance_degree(groups: list[int]) -> float:
    """ 
        https://bird.bcamath.org/bitstream/handle/20.500.11824/716/PRL%20Jonathan.pdf
        https://github.com/mjuez/py-imbalance-degree
    """
    return imb(groups)


def lrid(groups: list[int]) -> float:
    """ 
        https://www.sciencedirect.com/science/article/pii/S0167865518305907
        https://github.com/sahutkarsh/likelihood-ratio-imbalance-degree
    """
    raise NotImplementedError('Implementation in this github repo')
    return np.sum([np.abs(group - np.mean(groups)) for group in groups]) / np.sum(groups)
