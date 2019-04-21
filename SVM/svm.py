from random import shuffle
from random import choice
import numpy
from numpy import array as vec
from scipy.optimize import NonlinearConstraint
from scipy.optimize import Bounds
from scipy.optimize import minimize
from math import exp


class PointVal:
    def __init__(self, vals: list):
        self.label = 2*(vals[-1] - .5)
        self.vals = numpy.zeros(len(vals))
        for i in range(len(vals) - 1):
            self.vals[i] = vals[i]
        self.vals[-1] = 1


class SVM:
    def __init__(self, train_data: list, attr_len,  epochs: int, rate, C):
        self.weight = numpy.zeros(attr_len)

        for t in range(epochs):
            rt = next(rate)
            shuffle(train_data)
            for point in train_data:
                if point.label * numpy.dot(self.weight, point.vals) < 1:
                    self.weight[-1] = 0
                    self.weight = (1 - rt)*self.weight + rt * C * len(train_data) * point.label * point.vals
                else:
                    self.weight[:-1] = (1 - rt) * self.weight[:-1]

    def check(self, example):
        return example.label * numpy.dot(self.weight, example.vals) > 0

    def test(self, testing_set):
        wrong = 0
        total = len(testing_set)
        for i in testing_set:
            if not self.check(i):
                wrong += 1
        return wrong / total


def read_data(filename):
    result = list()
    with open(filename, 'r') as dataset:
        for line in dataset:
            raw_data = line.strip().split(',')
            num_data = list()
            for i in raw_data:
                num_data.append(float(i))
            result.append(PointVal(num_data))
    attribute_length = len(result[0].vals)
    return result, attribute_length


class DualSVM:
    def __init__(self, train_data, C, threshold, kernal=numpy.dot):
        self.kernal = kernal
        self.y = numpy.array(list(map(lambda a: a.label, train_data)))
        self.x = numpy.array(list(map(lambda ex: ex.vals[:-1], train_data)))
        y = self.y
        x = self.x

        bounds = Bounds(numpy.full(len(self.y), 0), numpy.full(len(self.y), C))
        eq_const = {
            'type': 'eq',
            'fun': lambda a: numpy.dot(a, self.y)
        }

        def funky_boy(al):
            total_sum = 0
            for i in range(len(al)):
                inner_sum = 0
                for j in range(len(al)):
                    inner_sum += al[j]*y[j] * kernal(x[i], x[j])
                total_sum += inner_sum * al[i] * y[i]
            return total_sum - numpy.sum(al)

        self.params = minimize(funky_boy, numpy.zeros(len(self.y)),
                               jac=False, constraints=[eq_const], bounds=bounds).x

        self.bias = self.recover_bias(threshold)

        self.support_num = len(list(filter(lambda a: a > threshold, self.params)))
        self.support_indices = self.find_supports(threshold)

    def find_supports(self, threshold):
        support_list = list()
        for i in range(len(self.params)):
            if self.params[i] > threshold:
                support_list.append(i)
        return support_list

    def recover_bias(self, threshold):
        b_sum = 0
        b_count = 0
        for i in range(len(self.params)):
            if self.params[i] > threshold:
                b_count += 1
                inner_sum = 0
                for j in range(len(self.params)):
                    inner_sum += self.params[j] * self.y[j] * self.kernal(self.x[j], self.x[i])
                b_sum += self.y[i] - inner_sum
        return b_sum/b_count

    def recover_weights(self):
        weight = numpy.zeros(len(self.x[0]))
        for i in range(len(self.params)):
           weight += self.params[i] * self.y[i] * self.x[i]
        return weight

    def predict(self, example):
        inner_sum = 0
        for j in range(len(self.params)):
            inner_sum += self.params[j] * self.y[j] * self.kernal(self.x[j], example.vals[:-1])
        return self.bias + inner_sum

    def test(self, test_data):
        wrong = 0
        total = len(test_data)
        for i in test_data:
            if i.label * self.predict(i) <= 0:
                wrong += 1
        return wrong / total


def gaussian_k(x1, x2, c):
    return exp(-numpy.dot(x1 - x2, x1 - x2)/c)


if __name__ == "__main__":
    train_set, attrs = read_data("bank-note/train.csv")

    Cs = [1/873, 10/873, 50/873, 100/873, 300/873, 500/873, 700/873]

    def part1rate(initial, raterate):
        initi = initial
        count = 0
        param = raterate
        while True:
            count += 1
            initi = initi / (1 + initi*count/param)
            yield initi

    def part2rate(initial):
        count = 0
        initi = initial
        while True:
            count += 1
            initi = initial / (1 + count)
            yield initi

    rate = part1rate(.1, .1)
    rate2 = part2rate(.05)

    print("Testing primal SVM with different rate functions-----")
    print("{:10}{:8}{:8}".format("c", "train", "test"))

    test_set, _ = read_data("bank-note/test.csv")
    for c in Cs:
        sivum = SVM(train_set, attrs, 100, rate, c)
        sivum2 = SVM(train_set, attrs, 100, rate2, c)
        # weightstring = ""
        # for i in range(len(sivum.weight)):
        #     weightstring += "{:8.4f}&".format(sivum.weight[i] - sivum2.weight[i])
        # print("{:<10.4}&{}\\\\\\hline".format(c, weightstring))
        print("{:<10.4}{:8.4f}{:8.4f}\\\\\\hline".format(c, sivum.test(train_set) - sivum2.test(train_set),
                                                           sivum.test(test_set) - sivum2.test(test_set)))
    num_samples = 50

    print("Testing linear dual svm with different c's")
    print("{:8.4}{:32}{:8}{:8}{:8}".format("c", "weight","bias", "train", "test"))

    for c in [100/873, 500/873, 700/873]:
        sivumdual = DualSVM(train_set[:num_samples], c, 1e-10)
        weight = sivumdual.recover_weights()
        train_err = sivumdual.test(train_set)
        test_err = sivumdual.test(test_set)
        weightstring = ""
        for i in range(len(weight)):
            weightstring += "{:8.4f}".format(weight[i])
        print("{:8.4f}{}{:8.4f}{:8.4f}{:8.4f}\\\\\\hline"
              .format(c, weightstring, sivumdual.bias, train_err, test_err))

    print("testing nonline svm----------------------")
    print("{:8}{:16}{:16}{:16}".format("g", "c = 100/873", "c = 500/873", "c = 700/873"))
    for g in [.01,.1,.5,1,2,5,10,100]:
        gdata = ""
        for c in [100 / 873, 500 / 873, 700 / 873]:
            sivumdual = DualSVM(train_set[:num_samples], c, 1e-10, lambda a, b: gaussian_k(a, b, g))
            train_err = sivumdual.test(train_set)
            test_err = sivumdual.test(test_set)
            gdata += ("{:8.4f}{:8.4f}".format(train_err,test_err))
        gline = "{:<8.4f}{}\\\\\\hline".format(g, gdata)
        print(gline)
    #



    print("testing nonline svm repeat supports----------------------")
    old_supports = list()
    for g in [.01,.1,.5,1,2,5,10,100]:
        gdata = ""
        c = 500/873
        sivumdual = DualSVM(train_set[:num_samples], c, 1e-10, lambda a, b: gaussian_k(a, b, g))
        new_supports = sivumdual.support_indices
        count = 0
        for support in old_supports:
            if support in new_supports:
                count += 1
        print(g, count)
        old_supports = new_supports



