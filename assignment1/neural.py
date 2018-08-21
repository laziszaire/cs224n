import numpy as np
import random
from q1_softmax import softmax
from q2_sigmoid import sigmoid
from q2_gradcheck import gradcheck_naive


class NeuralNetwork:
    """
    input + hidden + output
    """
    def __init__(self, dimensions, params):

        # dimensions
        Dx, H, Dy = dimensions

        # init paramaters
        ofs = 0
        self.W1 = np.reshape(params[ofs:ofs + Dx*H], (Dx, H))
        ofs += Dx*H
        self.b1 = np.reshape(params[ofs:ofs + H], (1, H))
        ofs += H
        self.W2 = np.reshape(params[ofs:ofs + H*Dy], (H, Dy))
        ofs += H*Dy
        self.b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    def forward_backward(self, X, labels):
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        local_grad_z1 = a1*(1-a1)
        self.scores = a1.dot(W2) + b2
        self.probs = softmax(self.scores)
        cost = self.cross_entropy(labels, self.probs).sum()

        #backward
        delta_scores = self.probs - labels
        grad_b2 = delta_scores.sum(axis=0)
        grad_W2 = a1.T.dot(delta_scores)
        delta_z1 = local_grad_z1 * (delta_scores.dot(W2.T))
        grad_W1 = X.T.dot(delta_z1)
        grad_b1 = delta_z1.sum(axis=0)
        grads = np.concatenate((grad_W1.flatten(), grad_b1.flatten(),
                               grad_W2.flatten(), grad_b2.flatten()))
        return cost, grads

    @staticmethod
    def cross_entropy(p_true, p_hat, axis=1):
        """
        ce = -y.dot(log(yhat))
        :param p_true:
        :param p_hat:
        :return:
        """
        ce = -np.sum(p_true*np.log(p_hat), axis=axis)
        return ce


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    def grad_c(params):
        nn = NeuralNetwork(dimensions, params)
        return nn.forward_backward(data, labels)
    gradcheck_naive(grad_c, params)


if __name__ == "__main__":
    sanity_check()