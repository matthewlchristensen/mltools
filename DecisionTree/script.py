import statistics
from collections import Counter
import math


class Example:
    def __init__(self, read_data, given_label):
        self.attributes = read_data
        self.label = given_label


def is_uniform(data_set: list):
    first_label = data_set[0].label
    for example in data_set[1:]:
        if not first_label == example.label:
            return False
    return True


class DecisionTree:                                        # possible_values, num_indices, [unknown_mode = false],
    def __init__(self, training_data, max_depth, strategy, possible_values, num_indices,
                 most_common="Nothing", depth=0, unknown_mode = False):
        """
        Takes in a list of examples and constructs a decision tree out of it.
        :param training_data:
        """

        if not training_data:
            self.prediction = most_common
            self.is_leaf = True
            return
        if is_uniform(training_data):
            self.prediction = training_data[0].label
            self.is_leaf = True
            return
        if depth == max_depth:
            self.prediction = most_common_label(training_data)
            self.is_leaf = True
            return

        self.is_leaf = False
        self.data = training_data
        self.children = dict()
        self.decision_attribute = self.find_decision_attribute(strategy, possible_values, num_indices)
        self.is_numeric = self.decision_attribute in num_indices
        if self.is_numeric:
            self.median = possible_values[self.decision_attribute]
        self.common_label = most_common_label(training_data)
        if self.decision_attribute not in num_indices:
            subsets = partition_categorical_data(training_data, self.decision_attribute)
            for value in subsets.keys():
                self.children[value] = DecisionTree(subsets[value], max_depth, strategy,
                                                    possible_values, num_indices, self.common_label, depth + 1)
        else:
            subsets = partition_numerical_data(training_data, self.decision_attribute,
                                               possible_values[self.decision_attribute])
            self.children["above"] = DecisionTree(subsets["above"], max_depth, strategy,
                                                  possible_values, num_indices, self.common_label, depth+1)
            self.children["below"] = DecisionTree(subsets["below"], max_depth, strategy,
                                                  possible_values, num_indices, self.common_label, depth+1)

    def test(self, test_data):
        total = len(test_data)
        incorrect = 0
        for example in test_data:
            if not self.predict(example) == example.label:
                incorrect += 1
        return incorrect/total

    def predict(self, example):
        if not self.is_leaf:
            value = example.attributes[self.decision_attribute]
            if not self.is_numeric:
                if value not in self.children:
                    return self.common_label
                selected_child = self.children[value]
            else:
                if float(value) > self.median:
                    selected_child = self.children["above"]
                else:
                    selected_child = self.children["below"]
            return selected_child.predict(example)
        else:
            return self.prediction

    def find_decision_attribute(self, strategy, possible_values, num_indices):
        best_so_far = 0
        index_of_best_so_far = 0
        for i in range(len(possible_values)):
            heuristic = strategy(i, self.data, possible_values, i in num_indices)
            # heuristic = information_gain(i, self.data, possible_values, i in num_indices, strategy)
            if heuristic > best_so_far:
                best_so_far = heuristic
                index_of_best_so_far = i
        return index_of_best_so_far


def entropy(subset):
    total_entropy = 0
    label_counts = dict()
    for individual in subset:
        if individual.label in label_counts:
            label_counts[individual.label] += 1
        else:
            label_counts[individual.label] = 1
    total_count = len(subset)
    for count in label_counts.values():
        proportion = count/total_count
        total_entropy -= proportion*math.log(proportion)
    return total_entropy


def max_error(subset):
    label_set = list()
    for individual in subset:
        label_set.append(individual.label)
    label_counter = Counter(label_list)
    most_common_count = label_counter.most_common(1)[0][1]
    total_count = len(subset)
    negatives = total_count - most_common_count
    return negatives/total_count


def gini(subset):
    total_gini = 1
    label_counts = dict()
    for individual in subset:
        if individual.label in label_counts:
            label_counts[individual.label] += 1
        else:
            label_counts[individual.label] = 1
    total_count = len(subset)
    for count in label_counts.values():
        proportion = count/total_count
        total_gini -= proportion*proportion
    return total_gini


def information_gain(attribute, example_list: list, possible_values, is_numeric):
# def information_gain(attribute, example_list: list, possible_values, is_numeric, strategy):
    total_gain = entropy(example_list)
    # total_gain = strategy(example_list)
    if not is_numeric:
        partition = partition_categorical_data(example_list, attribute)
        for value in partition.keys():
            subset = partition[value]
            proportion = len(subset)/len(example_list)
            total_gain -= proportion*entropy(subset)
            # total_gain -= proportion*strategy(subset)
    else:
        partition = partition_numerical_data(example_list, attribute, possible_values[attribute])
        subset_above = partition["above"]
        proportion_above = len(subset_above)/len(example_list)
        subset_below = partition["below"]
        proportion_below = len(subset_below)/len(example_list)
        total_gain -= proportion_above * entropy(subset_above) + proportion_below * entropy(subset_below)
        # total_gain -= proportion_above * strategy(subset_above) + proportion_below * strategy(subset_below)
    return total_gain


def partition_categorical_data(example_list: list, attribute):
    partition = dict()
    for example in example_list:
        value = example.attributes[attribute]
        if value in partition:
            partition[value].append(example)
        else:
            partition[value] = list()
            partition[value].append(example)
    return partition


def partition_numerical_data(example_list: list, attribute, median):
    partition = dict()
    partition["above"] = list()
    partition["below"] = list()
    for example in example_list:
        value = float(example.attributes[attribute])
        if value > median:
            partition["above"].append(example)
        else:
            partition["below"].append(example)
    return partition


def most_common_label(example_list: list):
    label_list = list()
    for example in example_list:
        label_list.append(example.label)
    counter = Counter(label_list)
    return counter.most_common(1)[0][0]


def attr_values(example_list: list):
    """
    Runs through a list of examples and returns a list of sets of values for each attribute index.
    If the attribute at the index is numeric, returns the median instead.
    :param example_list:
    :return: the attribute_value sets, and the indices of numerical attributes.j
    """
    attribute_count = len(example_list[0].attributes)
    values_list = list(range(attribute_count))
    numerical_indices = list(range(attribute_count))
    for i in range(attribute_count):
        for example in example_list:
            if not is_num(example.attributes[i]):
                numerical_indices.remove(i)
                break
    for i in range(attribute_count):
        if i not in numerical_indices:
            values_list[i] = set()
            for example in example_list:
                values_list[i].add(example.attributes[i])
        else:
            num_list = list()
            for example in example_list:
                num_list.append(float(example.attributes[i]))
            values_list[i] = statistics.median(num_list)
    return values_list, numerical_indices


def is_num(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    examples = list()
    print("hello world")
    with open("bank/train.csv", 'r') as dataset:
        for line in dataset:
            data = line.strip().split(',')
            label = data[-1]
            attr = data[:-1]
            examples.append(Example(attr, label))
    tests = list()
    with open("bank/test.csv", 'r') as test_set:
        for line in test_set:
            data = line.strip().split(',')
            label = data[-1]
            attr = data[:-1]
            tests.append(Example(attr, label))
    values, nums = attr_values(examples)
    labels = most_common_label(examples)
    print(values, nums)
    print(labels)

    label_list = list()
    for example in examples:
        label_list.append(example.label)
    counter = Counter(label_list)
    print(counter.most_common(1)[0][1])
    # tree = DecisionTree(examples, 3, information_gain, values, nums)
    # tree = DecisionTree(examples, 3, entropy, values, nums)
    # tree = DecisionTree(examples, 3, max_error, values, nums)
    # tree = DecisionTree(examples, 3, gini, values, nums)
    # print(tree.test(examples))
    # print(tree.test(tests))
