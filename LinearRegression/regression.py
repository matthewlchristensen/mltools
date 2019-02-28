from DecisionTree.dtree import *
from math import sqrt
from random import shuffle

INIT_RADIUS = 1
THRESHOLD = 1e-8


class BatchReg:
    def __init__(self, training_data, r=1.0):
        vector_size = len(training_data[0].attributes) + 1
        self.weight_vector = [0] * vector_size
        self.err_progression = list()
        radius = r
        for i in range(1000):
            gradient = [0] * vector_size
            total_err = 0
            for datum in training_data:
                err = self.error(datum)
                for j in range(vector_size-1):
                    gradient[j] -= err*float(datum.attributes[j])
                gradient[-1] -= err
                total_err += err*err
            delta = 0
            total_err /= 2
            self.err_progression.append(total_err)
            for j in range(vector_size):
                grad = gradient[j]
                self.weight_vector[j] -= grad*radius
                delta += grad*grad
            if sqrt(delta) < THRESHOLD:
                return

    def guess(self, example : Example):
        total = self.weight_vector[-1]
        for i in range(len(self.weight_vector) - 1):
            total += self.weight_vector[i] * float(example.attributes[i])
        return total

    def error(self, example: Example):
        return float(example.label) - self.guess(example)

    def test(self, test_set):
        total = 0
        for test in test_set:
            err = self.error(test)
            total += err*err
        total /= 2
        return total


class StochasticReg:
    def __init__(self, training_data, epochs=100, r=1.0):
        vector_size = len(training_data[0].attributes) + 1
        self.weight_vector = [0] * vector_size
        radius = r
        self.costs = list()
        for epoch in range(epochs):
            shuffle(training_data)
            for datum in training_data:
                gradient = [0] * vector_size
                err = self.error(datum)
                for j in range(vector_size-1):
                    gradient[j] = radius*err*float(datum.attributes[j])
                gradient[-1] = radius*err
                total_cost = self.test(training_data)
                self.costs.append(total_cost)
                delta = 0
                for j in range(vector_size):
                    grad = gradient[j]
                    self.weight_vector[j] += grad
                    delta += grad*grad
                if sqrt(delta) < THRESHOLD:
                    return

    def guess(self, example : Example):
        total = self.weight_vector[-1]
        for i in range(len(self.weight_vector) - 1):
            total += self.weight_vector[i] * float(example.attributes[i])
        return total

    def error(self, example: Example):
        return float(example.label) - self.guess(example)


    def test(self, test_set):
        total = 0
        for test in test_set:
            err = self.error(test)
            total += err*err
        total /= 2
        return total


if __name__ == "__main__":
    training_examples_concrete = load_data("concrete/train.csv")
    testing_examples_concrete = load_data("concrete/test.csv")

    batcher = BatchReg(training_examples_concrete, .01)
    print(batcher.weight_vector)
    print("{:<10s}{:>10s}".format("update", "error"))
    for i in range(0, len(batcher.err_progression), 1):
        print("{:<10d}{:10.4f}".format(i, batcher.err_progression[i]))
    print("\nFinal Testing Error")
    print(batcher.test(testing_examples_concrete))

    print("Testing stoch")
    print("{:<10s}{:>10s}{:>10s}{:>10s}".format("update", ".01", ".001", ".0001"))
    stoch1 = StochasticReg(training_examples_concrete, 1000, .01)
    stoch2 = StochasticReg(training_examples_concrete, 1000, .001)
    stoch3 = StochasticReg(training_examples_concrete, 1000, .0001)
    for i in range(0, len(stoch1.costs), 10):
        print("{:<10d}{:10.4f}{:10.4f}{:10.4f}".format(i, stoch1.costs[i], stoch2.costs[i], stoch3.costs[i]))
    print(".01 error: " + str(stoch1.test(testing_examples_concrete)))
    print(".001 error: " + str(stoch2.test(testing_examples_concrete)))
    print(".0001 error: ", stoch3.test(testing_examples_concrete))

    print("Weights for .01, .001, .0001 respectively")
    for i in range(len(stoch1.weight_vector)):
        print("{:<10d}{:10.4f}{:10.4f}{:10.4f}".format(i, stoch1.weight_vector[i], stoch2.weight_vector[i],
                                                       stoch3.weight_vector[i]))

