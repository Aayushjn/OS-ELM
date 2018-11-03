from math import ceil

import numpy as np


def activation(x):
    return 1 / (1 + np.exp(-x))


def mean_squared_error(a, b):
    return ((a - b) ** 2).mean(axis=None)


class OS_ELM(object):
    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes):
        self.__n_input_nodes = n_input_nodes
        self.__n_hidden_nodes = n_hidden_nodes
        self.__n_output_nodes = n_output_nodes

        self.__is_finished_init_train = False

        # alpha = learning rate
        self.__alpha = np.array([[np.random.uniform(-1, 1) for _ in range(self.__n_hidden_nodes)] for i in range(self.__n_input_nodes)])
        self.__bias = np.array([np.random.uniform(-1, 1) for _ in range(self.__n_hidden_nodes)])
        # beta is the weight of precision over recall in F-beta score
        self.__beta = np.zeros(shape=[self.__n_hidden_nodes, self.__n_output_nodes])
        # model parameter
        self.__p = np.zeros(shape=[self.__n_hidden_nodes, self.__n_hidden_nodes])

    def predict(self, x):
        return np.dot(activation(np.dot(x, self.__alpha) + self.__bias), self.__beta)

    def evaluate(self, x, t, metrics=None):
        if metrics is None:
            metrics = ['loss']
        met = []
        for m in metrics:
            if m == 'loss':
                met.append(mean_squared_error(self.predict(x), t))
            elif m == 'accuracy':
                tp = fp = 0
                for i in range(len(x)):
                    if ceil(self.predict(x)[i]) == t[i]:
                        tp += 1
                    else:
                        fp += 1
                met.append((tp / (tp + fp)))
            else:
                return ValueError('An unknown metric \'{}\' was given.'.format(m))

        return met

    def init_train(self, x, t):
        if self.__is_finished_init_train:
            raise Exception('The initial training phase has already finished. Please call \'seq_train\' method for '
                            'further training.')
        if len(x) < self.__n_hidden_nodes:
            raise ValueError('In the initial training phase, the number of training samples must be greater than the '
                             'number of hidden nodes. But this time len(x)={}, while n_hidden_nodes={}'
                             .format(len(x), self.__n_hidden_nodes))

        self.__build_init_train_graph(x, t)
        self.__is_finished_init_train = True

    def seq_train(self, x, t):
        if not self.__is_finished_init_train:
            raise Exception('You have not gone through the initial training phase yet. Please first initialize the '
                            'model\'s weights by \'init_train\' method before calling \'seq_train\' method.')
        self.__build_seq_train_graph(x, t)

    def __build_init_train_graph(self, x, t):
        # hidden layer output matrix
        H = activation(np.dot(x, self.__alpha) + self.__bias)
        HT = np.transpose(H)
        HTH = np.dot(HT, H)
        self.__p = np.linalg.inv(HTH)
        pHT = np.dot(self.__p, HT)
        self.__beta = np.dot(pHT, t)
        return self.__beta

    def __build_seq_train_graph(self, x, t):
        # hidden layer output matrix
        H = activation(np.dot(x, self.__alpha) + self.__bias)
        HT = np.transpose(H)
        batch_size = x.shape[0]
        I = np.eye(batch_size)
        Hp = np.dot(H, self.__p)
        HpHT = np.dot(Hp, HT)
        temp = np.linalg.inv(I + HpHT)
        pHT = np.dot(self.__p, HT)
        self.__p -= np.dot(np.dot(pHT, temp), Hp)
        pHT = np.dot(self.__p, HT)
        Hbeta = np.dot(H, self.__beta)
        self.__beta += np.dot(pHT, t - Hbeta)
        return self.__beta
