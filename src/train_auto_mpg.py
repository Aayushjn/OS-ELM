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
    n_input_nodes = 8
    n_hidden_nodes = 128
    n_output_nodes = 305

    os_elm = OS_ELM(n_input_nodes=n_input_nodes,
                    n_hidden_nodes=n_hidden_nodes,
                    n_output_nodes=n_output_nodes)

    # Load data
    data = read_csv('../datasets/auto-mpg.csv')
    # Encode classes to 0 - n_classes
    data['car name'] = LabelEncoder().fit_transform(data['car name'].astype('str'))
    # Split data to inputs and class labels
    x = data.iloc[:, :data.shape[1] - 1]
    t = data.iloc[:, data.shape[1] - 1]

    # Data normalization
    normalize_q = x.values.max()

    for val in x:
        x[val] /= normalize_q

    x_train, x_test, t_train, t_test = train_test_split(x, t, train_size=0.8, test_size=0.2)

    # Divide the dataset into two parts:-
    #   (1) for the initial training phase
    #   (2) for the sequential training phase
    # NOTE: The number of training samples for the initial training phase
    # must be much greater than the number of the model's hidden nodes.
    # Here we assign int(1.5 * n_hidden_nodes) training samples
    # for the initial training phase.
    border = int(1.5 * n_hidden_nodes)
    x_train_init = x_train.values[:border]
    x_train_seq = x_train.values[border:]
    t_train_init = t_train.values[:border]
    t_train_seq = t_train.values[border:]

    # ========== Initial training phase ==========
    progress_bar = tqdm(total=len(x_train), desc='initial training phase')
    os_elm.init_train(x_train_init, t_train_init)
    progress_bar.update(len(x_train_init))

    # ========== Sequential training phase ==========
    progress_bar.set_description('sequential training phase')
    batch_size = 64
    for i in range(0, len(x_train_seq), batch_size):
        x_batch = x_train_seq[i: i + batch_size]
        t_batch = t_train_seq[i: i + batch_size]
        os_elm.seq_train(x_batch, t_batch)
        progress_bar.update(len(x_batch))
    progress_bar.close()

    # Sample 'n' samples from the x_test
    n = len(x_test.values)
    x = x_test.values[:n]
    t = t_test.values[:n]

    y = os_elm.predict(x)
    y = soft_max(y)

    # Check the answers
    for i in range(n):
        max_ind = np.argmax(y.flatten()[i])
        print('======== sample index {} ========'.format(i))
        print('estimated answer: class {}'.format(max_ind))
        print('estimated probability: {}'.format(y[max_ind, i]))
        print('true answer: class {}'.format(t[i]))

    # Evaluate 'loss' and 'accuracy' metrics for the model
    [loss, accuracy] = os_elm.evaluate(x_test.values, t_test.values, metrics=['loss', 'accuracy'])
    print('\nval_loss: {}, val_accuracy: {:.3f}%'.format(loss, accuracy * 100))


if __name__ == '__main__':
    main()
