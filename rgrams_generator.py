from collections import Counter
from itertools import tee, zip_longest

"""
This is an implementation of the rgrams described by Ekgren A. et al.
in R-grams: Unsupervised Learning of Semantic Units in Natural Language (2018)
https://arxiv.org/abs/1808.04670
"""

def pairwise(tokens, longest=False):
    """ Generate pairs from a list
    Example :
        pairwise(['spam', 'bacon', 'eggs'])
        > [('spam', 'bacon'), ('bacon', 'eggs')]

        pairwise(['spam', 'bacon', 'eggs'], True)
        > [('spam', 'bacon'), ('bacon', 'eggs'), ('eggs', None)]
    :param tokens: list of tokens to process
    :param longest: boolean value if false use zip, otherwise use zip_longest
    :return: iterable on a list of tuples
    """
    a, b = tee(tokens)
    next(b, None)
    return zip(a, b) if longest is False else zip_longest(a, b)


def get_most_common_pair(tokens):
    """ Get the most common pair of tokens
    Example:
        get_most_common_pair(['spam', 'spam', 'eggs', 'spam', 'spam'])
         > ('spam spam', 2)
    :param tokens: tokens
    :return: a tuple containing the most common token and its frequency
    """
    pairs = pairwise(tokens)
    return Counter([' '.join(p) for p in pairs]).most_common(1)[0]


def concatenate_pair(tokens, pair):
    """
    Find all the pairs of tokens similar to 'pair' and replace them by
    the pair
    Example:
        concatenate_pair(['spam', 'spam', 'eggs', 'spam', 'spam'], 'spam spam')
        > ['spam spam', 'eggs', 'spam spam']
    :param tokens: list of tokens
    :param pair: pair of tokens to concatenate
    :return: list of token
    """
    pairs = pairwise(tokens, longest=True)
    new_tokens = []
    for p in pairs:
        current_pair = " ".join(p) if p[1] else p[0]
        if current_pair == pair:
            new_tokens.append(pair)
            next(pairs, None)
        else:
            new_tokens.append(p[0])
    return new_tokens


def generate_rgrams(tokens, min_freq=4, current_iter=0, max_iter=50000):
    """ Generate rgrams using BPE algorithm
    Example:
        generate_rgrams(['spam', 'spam', 'eggs', 'spam', 'spam'], 2)
        > ['spam spam', 'eggs', 'spam spam']

        generate_rgrams(['spam', 'spam', 'eggs', 'spam', 'spam'], 3)
        > ['spam', 'spam', 'eggs', 'spam', 'spam']
    :param tokens: list of unigrams
    :param min_freq: minimum frequency of occurrence of the rgram
    :param current_iter: current number of iterations processed
    :param max_iter: maximum number of iterations to perform
    :return: list of generated rgrams
    """
    if current_iter > max_iter:
        return tokens
    pair, freq = get_most_common_pair(tokens)
    if freq < min_freq:
        return tokens
    tokens = concatenate_pair(tokens, pair)
    return generate_rgrams(tokens, min_freq, current_iter + 1, max_iter)