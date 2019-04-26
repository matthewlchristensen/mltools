import numpy as np
from numpy import genfromtxt
from math import exp


def sigmoid(x):
    if x >= 0:
        z = exp(-x)
        return 1/(1+z)
    else:
        z = exp(x)
        return z/(1+z)


def dsig(x):
    """
    Wrapper for derivative of sigmoid
    :param x:
    :return:
    """
    return sigmoid(x) * (1-sigmoid(x))


def read_data(filepath):
    data = genfromtxt(filepath, delimiter=',')

    data = np.insert(data, -1, 1, axis=1)
    data[:, -1] -= .5
    data[:, -1] *= 2

    return data, data.shape[1] - 1


class NN:
    def __init__(self, input_data, epochs, nodes_per_layer, attr_num, layers, init_rate, g, init_scheme="zeros"):

        # initialize caches, structures, etc.
        self.weights = []
        self.nodes = []
        self.dloss_dnode_cache = []
        self.dloss_dweights = []

        self.num_inputs = attr_num
        self.nodes_per_layer = nodes_per_layer
        self.num_layers = layers

        self.initialize_weight(attr_num, layers, nodes_per_layer, init_scheme)
        trate = self.learn_rate(init_rate, g)

        # train data
        for t in range(epochs):
            next_rate = next(trate)
            np.random.shuffle(input_data)
            for sample in input_data:
                self.reset_nodes_n_caches(self.num_inputs, self.num_layers, self.nodes_per_layer)
                prediction = self.get_predict(sample)
                self.backpropagate(sample, prediction)

                for layer in reversed(range(len(self.weights))):
                    for dest in range(len(self.weights[layer])):
                        for src in range(len(self.weights[layer][dest])):
                            self.weights[layer][dest][src] -= next_rate * self.dloss_dweights[layer][dest][src]

    def learn_rate(self, init, g):
        yield init
        iter = 0
        while True:
            yield init / (1 + (init/g))
            iter += 1

    def reset_nodes_n_caches(self, num_inputs, num_layers, nodes_per_layer):
        self.nodes = [np.full(num_inputs, np.nan)]
        self.dloss_dnode_cache = [np.full(num_inputs, np.nan)]
        self.dloss_dweights = [np.full((nodes_per_layer - 1, num_inputs), np.nan)]
        for layer in range(1, num_layers):
            self.nodes.append(np.full(nodes_per_layer, np.nan))
            self.nodes[-1][-1] = 1
            self.dloss_dnode_cache.append(np.full(nodes_per_layer, np.nan))
            self.dloss_dweights.append(np.full((nodes_per_layer - 1, nodes_per_layer), np.nan))
        self.dloss_dweights[-1] = np.full((1, nodes_per_layer), np.nan)
        # print("{}, {}, {}".format(len(self.nodes), len(self.dloss_dnode_cache), len(self.dloss_dweights)))

    def initialize_weight(self, attr_num, layers, nodes_per_layer, scheme="zeros"):
        if scheme == "zeros":
            self.weights.append(np.zeros((nodes_per_layer - 1, attr_num)))
            for layer in range(1, layers - 1):
                self.weights.append(np.zeros((nodes_per_layer - 1, nodes_per_layer)))
            self.weights.append(np.zeros((1, nodes_per_layer)))
        elif scheme == "gaussian":
            self.weights.append(np.random.normal(size=(nodes_per_layer - 1, attr_num)))
            for layer in range(1, layers - 1):
                self.weights.append(np.random.normal(size=(nodes_per_layer - 1, nodes_per_layer)))
            self.weights.append(np.random.normal(size=(1, nodes_per_layer)))
        # print("Length of weights: {}".format(len(self.weights)))

    def dnextnode_dcurrnode(self, row, dest, src):
        return dsig(np.dot(self.nodes[row][dest], self.weights[row][dest]))*self.weights[row][dest][src]

    def dnextnode_dweight(self, row, dest, src):
        return dsig(np.dot(self.nodes[row][dest], self.weights[row][dest]) * self.nodes[row][src])

    def dloss_dnode(self, row, index, sample, prediction):
        # if we're at the top layer
        if row == self.num_layers - 1:
            self.dloss_dnode_cache[row][index] = (prediction - sample[-1]) * self.weights[row][0][index]
            return self.dloss_dnode_cache[row][index]
        elif not np.isnan(self.dloss_dnode_cache[row][index]):
            return self.dloss_dnode_cache[row][index]
        else:
            total = 0
            for i in range(len(self.nodes[row + 1]) - 1):
                total += self.dloss_dnode(row+1, i, sample, prediction) * \
                         dsig(np.dot(self.nodes[row], self.weights[row][i])) * self.weights[row][i][index]

            self.dloss_dnode_cache[row][index] = total
            return total

    def dloss_dweight(self, row, dest, src, sample, prediction):
        if row == self.num_layers - 1:
            self.dloss_dweights[row][dest][src] = (prediction - sample[-1]) * self.nodes[row][src]
            pass

        elif np.isnan(self.dloss_dweights[row][dest][src]):
            self.dloss_dweights[row][dest][src] = self.dloss_dnode(row + 1, dest, sample, prediction) \
                                                  * dsig(np.dot(self.nodes[row], self.weights[row][dest])) \
                                                  * self.nodes[row][src]
        # return self.dloss_dweights[row][dest][src]

    def backpropagate(self, sample, prediction):
        for layer in reversed(range(len(self.weights))):
            for dest in range(len(self.weights[layer])):
                for src in range(len(self.weights[layer][dest])):
                    self.dloss_dweight(layer, dest, src, sample, prediction)

    def predict(self, example):
        self.reset_nodes_n_caches(self.num_inputs, self.num_layers, self.nodes_per_layer)
        prediction = self.get_predict(example)
        return -1 if prediction <= 0 else 1

    def test(self, test_data):
        error = 0
        for to_test in test_data:
            prediction = self.predict(to_test)
            if prediction * to_test[-1] < 0:
                error += 1
        return error/len(test_data)

    def get_predict(self, sample):
        self.nodes[0] = sample[:-1]
        top_layer = self.num_layers - 1
        for neuron in range(len(self.nodes[top_layer])):
            if np.isnan(self.nodes[top_layer][neuron]):
                self.fill_neuron(top_layer, neuron)

        return np.dot(self.nodes[top_layer], self.weights[top_layer][0])

    def fill_neuron(self, layer, index):
        """
        Ensures that the neuron at the provided layer and index is filled in the cache
        :param layer:
        :param index:
        :return:
        """
        for i in range(len(self.nodes[layer - 1])):
            if np.isnan(self.nodes[layer - 1][i]):
                self.fill_neuron(layer - 1, i)
        self.nodes[layer][index] = sigmoid(np.dot(self.nodes[layer - 1], self.weights[layer - 1][index]))


if __name__ == "__main__":
    train_data, attr_num = read_data("bank-note/train.csv")
    test_data, _ = read_data("bank-note/test.csv")
    # use 1 and .01 for init and g
    print("nodes    train_g test_g  train_0 test_0")
    for k in [5, 10, 25, 50, 100]:
        nn_gauss = NN(train_data, 10, k, attr_num, 3, .1, .01, init_scheme="gaussian")
        test_err_gauss = nn_gauss.test(test_data)
        train_err_gauss = nn_gauss.test(train_data)
        nn_z = NN(train_data, 10, k, attr_num, 3, .1, .01, init_scheme="zeros")
        test_err_z = nn_z.test(test_data)
        train_err_z = nn_z.test(train_data)
        print("{:<4d}&{:7.4f}&{:7.4f}&{:7.4f}&{:7.4f}\\\\\\hline"
              .format(k, train_err_gauss, test_err_gauss, train_err_z, test_err_z))
    pass
