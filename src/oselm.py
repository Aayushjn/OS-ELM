import numpy as np


def get_label_range(label):
    bottom = min(label)
    top = max(label)
    return top - bottom + 1, bottom


def normalize(data, label_index=-1):
    for ind in range(1, len(data[0])):
        if ind != label_index:
            maxi = np.max(data[:, ind:ind + 1])
            mini = np.min(data[:, ind:ind + 1])
            for i in range(len(data)):
                data[i][ind] = zero_one(data[i][ind], mini, maxi)

    return data


def sig(test_data, input_weights, bias):
    v = test_data * input_weights.T
    bias_1 = np.ones((len(test_data), 1)) * bias
    v = v + bias_1
    H = 1. / (1 + np.exp(-v, dtype="float64"))
    return H


def zero_one(x, minimum, maximum):
    return (x - minimum) / (maximum - minimum)


class OnlineSequentialExtremeLearningMachine:
    def __init__(self, n_hidden_nodes, n_input_nodes):
        self.n_hidden_nodes = n_hidden_nodes
        self.n_input_nodes = n_input_nodes
        # shape of input_weights = (n_hidden_nodes, n_input_nodes)
        self.input_weights = np.mat(np.random.rand(self.n_hidden_nodes, self.n_input_nodes) * 2 - 1)
        # shape of bias = (1, n_hidden_nodes)
        self.bias = np.mat(np.random.rand(1, self.n_hidden_nodes))
        self.P = None
        self.beta = None
        self.range = None

    def fit_init(self, data, label_index=0):
        label = []
        matrix = []
        # normalize data between 0 and 1
        data = normalize(data, label_index)
        np.random.shuffle(data)
        for row in data:
            temp = []
            label.append(int(row[label_index]))
            for index, item in enumerate(row):
                if index != label_index:
                    temp.append(item)
            matrix.append(temp)

        self.range = get_label_range(label)
        p0 = np.mat(matrix)
        T0 = np.zeros((len(matrix), self.range[0]))
        for index, item in enumerate(label):
            T0[index][item - self.range[1]] = 1
        # 0th target attribute matrix
        T0 = T0 * 2 - 1
        # 0th hidden layer output matrix
        H0 = sig(p0, self.input_weights, self.bias)
        self.P = (H0.T * H0).I
        self.beta = self.P * H0.T * T0
        self.error_calc(data)

        return self

    def fit_train(self, data, label_index=0):
        data = normalize(data, label_index)
        for row in data:
            # kth target attribute matrix
            Tk = np.zeros((1, self.range[0]))
            b = int(row[0])
            Tk[0][b - self.range[1]] = 1
            Tk = Tk * 2 - 1
            matrix = []
            for index, item in enumerate(row):
                if index != label_index:
                    matrix.append(item)
            Pk = np.mat(matrix)
            # kth hidden layer output data
            Hk = sig(Pk, self.input_weights, self.bias)
            self.P = self.P - self.P * Hk.T * (np.eye(1, 1) + Hk * self.P * Hk.T).I * Hk * self.P
            self.beta = self.beta + self.P * Hk.T * (Tk - Hk * self.beta)
        self.error_calc(data)

        return self

    def predict(self, data):
        res = []
        data = normalize(data)
        for row in data:
            matrix = []
            for item in row:
                matrix.append(item)
            p = np.mat(matrix)
            HTrain = sig(p, self.input_weights, self.bias)
            Y = HTrain * self.beta
            res.append(np.argmax(Y) + self.range[1])

        return res

    def error_calc(self, data, label_index=0):
        correct = 0
        summation = 0
        for row in data:
            matrix = []
            for index, item in enumerate(row):
                if index != label_index:
                    matrix.append(item)
            p = np.mat(matrix)
            HTrain = sig(p, self.input_weights, self.bias)
            Y = HTrain * self.beta
            if np.argmax(Y) + self.range[1] == int(row[label_index]):
                correct += 1
            summation += 1
        print("Accuracy：{:.5f}%".format((correct / summation) * 100))
        return correct / summation
