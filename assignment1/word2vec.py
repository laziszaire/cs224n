#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad
from q3_word2vec import normalizeRows
from collections import Counter


def softmax_loss_grad(predicted, outputVectors, target, dataset):
    vc = predicted
    U = outputVectors
    scores = U.dot(vc)
    probs = softmax(scores)      # |V|个相加， 计算词库中每个词的概率
    cost = -np.log(probs[target])
    d_score = probs
    d_score[target] -= 1   # gradient of score（softmax-CE(score))： p - y

    # score = U.dot(vc)
    grad_vc = U.T.dot(d_score)
    grad_U = d_score[:, np.newaxis].dot(vc[np.newaxis, :])  # gradient of U
    return cost, grad_vc, grad_U


def negative_samples(target, dataset, K):

    # sample a word that is not the target
    # 输入（contex)和输出(target)之间是可以重复的 (context, target)
    # negative samples 不能和 target相同        (context, !target)

    indices = [None]*K
    for k in xrange(K):
        while True:
            new_idx = dataset.sample_idx()
            if new_idx != target:
                break
        indices[k] = new_idx
    return indices


def negsampling(predicted, outputVectors, target, dataset, K=10):
    """
    输入一个词 对应 [一个目标词，一些非目标词]
    输入词和目标词可以相同
     d(log_sigmoid(z)/dz = (1-sigmoid(z))
     d(log_sigmoid(-z)/dz = -(1-sigmoid(-z))
    """
    indices = [target]
    indices += negative_samples(target, dataset, K)
    labels = -np.ones_like(indices)  # indicator of negative samples, sigmoid(-z),
    labels[0] = 1
    grad_U = np.zeros_like(outputVectors)
    _outputVectors = outputVectors[indices, :]
    z = labels*_outputVectors.dot(predicted)    # 只用计算采用的概率
    prob = sigmoid(z)
    cost = -np.sum(np.log(prob))

    # grad
    _prob = (prob-1)*labels
    grad_pred = _outputVectors.T.dot(_prob)
    _grad_U = _prob[:, np.newaxis].dot(predicted[np.newaxis, :])

    # sample可能重复
    for _grad_U_idx, output_idx in enumerate(indices):
        grad_U[output_idx, :] += _grad_U[_grad_U_idx, :]

    return cost, grad_pred, grad_U


def skipgram(current_word, window_size, context_words, word2idx, input_vectors, output_vectors,
             dataset, cost_grad=softmax_loss_grad):
    """
    one step skip-gram word2vec model
    一个window多对（len(context_words)）训练样本
    """
    cost = .0
    grad_in = np.zeros_like(input_vectors)
    grad_out = np.zeros_like(output_vectors)

    vc_idx = word2idx[current_word]
    predicted = input_vectors[vc_idx, :]
    for context_word in context_words:
        context_word_idx = word2idx[context_word]
        target = context_word_idx
        _cost, _grad_in, _grad_out = cost_grad(predicted, output_vectors, target, dataset)
        cost += _cost
        grad_in[vc_idx, ] += _grad_in
        grad_out += _grad_out

    return cost, grad_in, grad_out


def cbow(current_word, windowsize, context_words, word2idx, input_vectors, output_vectors,
         dataset, cost_grad=softmax_loss_grad):
    """
    one step continuous bag of words
    一个窗口（中心词）只有一条样本（一对（contex, target)）
    """

    grad_in = np.zeros_like(input_vectors)
    context_indices = [word2idx[word] for word in context_words]
    predicted = np.sum(input_vectors[context_indices, :], axis=0)
    target = word2idx[current_word]
    # cost and grad
    cost, _grad_in, grad_out = cost_grad(predicted, output_vectors, target, dataset)


    # 从vhat向后传播梯度到input
    # for context_idx in context_indices:  # 可能重复, context x = [0, 0, 2, 1, 0, 3], W' = delta*x
    #     grad_in[context_idx, :] += _grad_in  # sum 每个input vector都是一样grad

    _counter = Counter(context_indices)  # 可能重复
    ucontext_idx = _counter.keys()
    num_occurrence = np.asarray(_counter.values())[:, np.newaxis]
    grad_in[ucontext_idx, :] += num_occurrence.dot(_grad_in[np.newaxis, :])

    return cost, grad_in, grad_out


def word2vec_sgd(word2vec_model, word2indx, word_vectors, dataset, window_size,
                 cost_grad=softmax_loss_grad, batch_size=50):
    cost = .0   # avg cost per word
    grad = np.zeros_like(word_vectors)  # avg grad per word
    N = word_vectors.shape[0]
    input_vectors = word_vectors[:N/2, :]
    output_vectors = word_vectors[N/2:, :]
    for i in xrange(batch_size):
        center_word, context = dataset.get_random_context(window_size)   # 生成数据
        _cost, g_in, g_out = word2vec_model(center_word, window_size, context, word2indx, input_vectors, output_vectors,
                                            dataset, cost_grad)
        cost += _cost/batch_size
        grad[:N/2, :] += g_in/batch_size
        grad[N/2:, :] += g_out/batch_size
    return cost, grad


def test_word2vec():

    dataset = type('dummy', (), {})()  # class name, bases, namespace| __name__, __bases__, __dict__

    def sample_idx():
        return random.randint(0, 4)

    def get_random_context(window_size):
        tokens = ['a', 'b', 'c', 'd', 'e']
        predicted = tokens[random.randint(0, 4)]
        context = [tokens[random.randint(0, 4)] for i in xrange(2*window_size)]
        return predicted, context
    dataset.sample_idx = sample_idx
    dataset.get_random_context = get_random_context

    random.seed(1)
    np.random.seed(1)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])

    gradcheck_naive(lambda vec: word2vec_sgd(skipgram, dummy_tokens, vec, dataset, 5, softmax_loss_grad),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd(skipgram, dummy_tokens, vec, dataset, 5, negsampling),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd(cbow, dummy_tokens, vec, dataset, 5, softmax_loss_grad),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd(cbow, dummy_tokens, vec, dataset, 5, negsampling),
                    dummy_vectors)



    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negsampling)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negsampling)


if __name__ == "__main__":
    test_word2vec()