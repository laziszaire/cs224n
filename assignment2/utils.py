import numpy as np

def mini_batch(data, batch_size, shuffle=True):
    """
    mini batch generator
    :param data:
    :param batch_size:
    :param shuffle:
    :return:
    """

    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for mini_start in np.arange(0, data_size, batch_size):
        mini_batch_indice = indices[mini_start: mini_start + batch_size]
        yield [_minibatch(d, mini_batch_indice) for d in data] if list_data \
            else _minibatch(data, mini_batch_indice)


def _minibatch(data, indices):
    return data[indices] if type(data) is np.ndarray else [data[i] for i in indices]

