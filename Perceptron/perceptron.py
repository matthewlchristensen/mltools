from random import shuffle


class PointVal:
    def __init__(self, vals: list):
        self.label = vals[-1]
        self.vals = vals[:-1]
        self.vals.append(1)


class Perceptron:
    def __init__(self, train_data: list, attr_len, epochs: int, rate: float,  mode="default"):
        self.weight = [0]*attr_len
        self.avgw = [0]*attr_len
        self.voters = list()
        self.vote_weights = [0]
        curr_index = 0
        self.mode = mode

        for t in range(epochs):
            shuffle(train_data)
            for i in train_data:
                label_index = (i.label - .5) * 2
                if label_index * self.predict(i, override=True) <= 0:
                    for j in range(attr_len):
                        self.weight[j] += rate * label_index * i.vals[j]

                    if mode == "vote":
                        self.vote_weights.append(1)
                        curr_index += 1
                        self.voters.append(self.weight.copy())
                elif mode == "vote":
                    self.vote_weights[curr_index] += 1

                if mode == "avg":
                    for j in range(attr_len):
                        self.avgw[j] += self.weight[j]

    def predict(self, example, override=False):
        if self.mode == "vote" and not override:
            gran_total = 0
            attrs = len(example.vals)
            for k in range(len(self.voters)):
                subtotal = 0
                voter = self.voters[k]
                for j in range(attrs):
                    subtotal += voter[j] * example.vals[j]
                gran_total += (1 if subtotal > 0 else -1) * self.vote_weights[k]
            return gran_total

        total = 0
        judge = self.avgw if self.mode == "avg" and not override else self.weight
        for i in range(len(judge)):
            total += judge[i] * example.vals[i]
        return total

    def test(self, testing_set):
        wrong = 0
        total = len(testing_set)
        for i in testing_set:
            if (i.label - 1/2)*2 * self.predict(i) <= 0:
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


if __name__ == "__main__":
    train_set, attrs = read_data("bank-note/train.csv")
    test_set, _ = read_data("bank-note/test.csv")
    perc = Perceptron(train_set, attrs, 10, .1)
    perca = Perceptron(train_set, attrs, 10, .1, "avg")
    percv = Perceptron(train_set, attrs, 10, .1, "vote")
    print("Vectors for voted perceptron:\n----------------------")
    print("{:7s}{:7s}{:7s}{:7s}{:7s}{:7s}".format("weights", "", "", "", "", "counts"))
    weight_format = "{:7.3f}{:7.3f}{:7.3f}{:7.3f}{:7.3f}{:7d}"
    for i in range(len(percv.voters)):
        voter = percv.voters[i]
        print(weight_format.format(voter[0], voter[1], voter[2], voter[3], voter[4], percv.vote_weights[i]))
    print("Standard final weight:\n", perc.weight, "\nStandard Test Error:\n", perc.test(test_set))
    print("Voted most recent\n", percv.weight, "\nVoted Test Error:\n", percv.test(test_set))
    print("Averaged Weight:", perca.weight, "\nAveraged Error:\n", perca.test(test_set))


