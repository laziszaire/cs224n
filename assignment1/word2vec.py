# why normalize row?
import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def softmax_loss(predicted, outputVectors, target):
    vc = predicted
    U = outputVectors
    scores = U.dot(vc)
    probs = softmax(scores)
    cost = -np.log(probs[target])
    d_score = probs
    d_score[target] -= 1   # gradient of score
    grad_vc = U.T.dot(d_score)
    grad_U = d_score[:, np.newaxis].dot(vc[np.newaxis, :]) # gradient of U
    return cost, grad_vc, grad_U

def negative_samples(target, dataset, K):

    # sample a word that is not the target
    indices = [None]*K
    for k in xrange(K):
        while True:
            newidx = dataset.sampleTokenIdx()
            if newidx != target:
                break
        indices[k] = newidx
    return indices


def negsampling(predicted, target, outputVectors, dataset, K=10):

    indices = [target]
    indices += negative_samples(target, dataset, K)
    labels = -np.ones_like(indices) # indicator of negative samples, sigmoid(-z)
    labels[0] = 1

    # 两部分合在一起了
    _outputVectors = outputVectors[indices, :]
    z = labels*_outputVectors.dot(predicted)
    prob = sigmoid(z)
    cost = -np.sum(np.log(prob))

    # grad
    _prob = (prob-1)*labels
    grad_pred = _outputVectors.dot(_prob)
    _grad = _prob[:, np.newaxis].dot(predicted[np.newaxis, :])
    grad_U = np.zeros_like(outputVectors)
    grad_U[indices, :] = _grad
    return cost, grad_pred, grad_U
