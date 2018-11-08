import numpy as np
from oselm import OnlineSequentialExtremeLearningMachine


def main():
    data = np.loadtxt(open('../datasets/housing.csv', 'r'), delimiter=',', skiprows=1)

    split = int(0.8 * len(data))
    train = data[0:split]
    test = data[split:]

    network = OnlineSequentialExtremeLearningMachine(n_hidden_nodes=256, n_input_nodes=9)
    network = network.fit_init(data=train)
    network = network.fit_train(data=train)
    network.predict(data=test[:, 1:])
    network.error_calc(data=train)


if __name__ == '__main__':
    main()
