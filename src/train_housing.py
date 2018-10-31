import numpy as np
from elm import OS_ELM
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def soft_max(a):
    c = np.max(a, axis=-1).reshape(-1, 1)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=-1).reshape(-1, 1)
    return exp_a / sum_exp_a


def main():
    n_input_nodes = 9
    n_hidden_nodes = 16
    n_output_nodes = 5

    os_elm = OS_ELM(n_input_nodes=n_input_nodes,
                    n_hidden_nodes=n_hidden_nodes,
                    n_output_nodes=n_output_nodes)

    data = read_csv('../datasets/housing.csv')
    data['ocean_proximity'] = LabelEncoder().fit_transform(data['ocean_proximity'].astype('str'))
    x = data.iloc[:, :data.shape[1] - 1]
    t = data.iloc[:, data.shape[1] - 1]

    normalize_q = max(x.values)

    for val in x:
        x[val] /= normalize_q

    x_train, x_test, t_train, t_test = train_test_split(x, t, train_size=0.8, test_size=0.2)

    border = int(1.5 * n_hidden_nodes)
    x_train_init = x_train.values[:border]
    x_train_seq = x_train.values[border:]
    t_train_init = t_train.values[:border]
    t_train_seq = t_train.values[border:]

    progress_bar = tqdm(total=len(x_train), desc='initial training phase')
    os_elm.init_train(x_train_init, t_train_init)
    progress_bar.update(len(x_train_init))

    progress_bar.set_description('sequential training phase')
    batch_size = 64
    for i in range(0, len(x_train_seq), batch_size):
        x_batch = x_train_seq[i: i + batch_size]
        t_batch = t_train_seq[i: i + batch_size]
        os_elm.seq_train(x_batch, t_batch)
        progress_bar.update(len(x_batch))
    progress_bar.close()

    n = 10
    x = x_test.values[:n]
    t = t_test.values[:n]

    y = os_elm.predict(x)
    y = soft_max(y)

    for i in range(n):
        max_ind = np.argmax(y[i])
        print('======== sample index {} ========'.format(i))
        print('estimated answer: class {}'.format(max_ind))
        print('estimated probability: {}'.format(y[max_ind, i]))
        print('true answer: class {}'.format(t[i]))

    [loss, accuracy] = os_elm.evaluate(x_test, t_test, metrics=['loss', 'accuracy'])
    print('val_loss: {}, val_accuracy: {}'.format(loss, accuracy))


if __name__ == '__main__':
    main()
