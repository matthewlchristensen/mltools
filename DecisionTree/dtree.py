import statistics
from collections import Counter
import math
from random import sample


class Example:
    def __init__(self, read_data, given_label, weight=1):
        self.attributes = read_data
        self.label = given_label
        self.weight = weight


def is_uniform(data_set: list):
    first_label = data_set[0].label
    for example in data_set[1:]:
        if not first_label == example.label:
            return False
    return True


class DecisionTree:                                        # possible_values, num_indices, [unknown_mode = false],
    def __init__(self, training_data, max_depth, strategy, possible_values, num_indices,
                 most_common="Nothing", depth=0, use_random_subset=False, random_subset_size=0):
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
            self.prediction = most_common_label_weighted(training_data)
            self.is_leaf = True
            return

        self.is_leaf = False
        self.children = dict()
        self.decision_attribute = DecisionTree.find_decision_attribute(training_data, strategy,
                                                                       possible_values, num_indices,
                                                                       use_random_subset, random_subset_size)
        self.is_numeric = self.decision_attribute in num_indices
        if self.is_numeric:
            self.median = possible_values[self.decision_attribute]
        self.common_label = most_common_label_weighted(training_data)
        if self.decision_attribute not in num_indices:
            subsets = partition_categorical_data(training_data, self.decision_attribute)
            for value in subsets.keys():
                self.children[value] = DecisionTree(subsets[value], max_depth, strategy,
                                                    possible_values, num_indices, self.common_label, depth + 1,
                                                    use_random_subset, random_subset_size)
        else:
            subsets = partition_numerical_data(training_data, self.decision_attribute,
                                               possible_values[self.decision_attribute])
            self.children["above"] = DecisionTree(subsets["above"], max_depth, strategy,
                                                  possible_values, num_indices, self.common_label, depth+1,
                                                  use_random_subset, random_subset_size)
            self.children["below"] = DecisionTree(subsets["below"], max_depth, strategy,
                                                  possible_values, num_indices, self.common_label, depth+1,
                                                  use_random_subset, random_subset_size)

    def test(self, test_data):
        """
        :param test_data:
        :return: weighted error proportion
        """
        total = len(test_data)
        total_weight = 0
        incorrect = 0
        for example in test_data:
            if not self.predict(example) == example.label:
                incorrect += example.weight
            total_weight += example.weight
        return incorrect/total_weight

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

    @staticmethod
    def find_decision_attribute(training_data, strategy, possible_values, num_indices,
                                use_random_subset=False, random_subset_size=0):
        best_so_far = 0
        index_of_best_so_far = 0
        attr_range = range(len(possible_values))
        if use_random_subset:
            attr_range = sample(attr_range, random_subset_size)
        for i in attr_range:
            heuristic = information_gain(i, training_data, possible_values, i in num_indices, strategy)
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


def entropy_weighted(subset):
    total_entropy = 0
    label_counts = dict()
    total_count = 0
    for individual in subset:
        if individual.label in label_counts:
            label_counts[individual.label] += individual.weight
        else:
            label_counts[individual.label] = individual.weight
        total_count += individual.weight
    for count in label_counts.values():
        proportion = count/total_count
        total_entropy -= proportion*math.log(proportion)
    return total_entropy


def max_error(subset):
    if not subset:
        return 0
    label_set = list()
    for individual in subset:
        label_set.append(individual.label)
    label_counter = Counter(label_set)
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


def information_gain(attribute, example_list: list, possible_values, is_numeric, strategy):
    total_gain = strategy(example_list)
    if not is_numeric:
        partition = partition_categorical_data(example_list, attribute)
        for value in partition.keys():
            subset = partition[value]
            proportion = len(subset)/len(example_list)
            total_gain -= proportion*strategy(subset)
    else:
        partition = partition_numerical_data(example_list, attribute, possible_values[attribute])
        subset_above = partition["above"]
        proportion_above = len(subset_above)/len(example_list)
        subset_below = partition["below"]
        proportion_below = len(subset_below)/len(example_list)
        total_gain -= proportion_above * strategy(subset_above) + proportion_below * strategy(subset_below)
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


def most_common_label_weighted(example_list: list):
    label_list = dict()
    biggest_amount = 0
    biggest_label = ""
    for example in example_list:
        label = example.label
        if label in label_list:
            label_list[label] += example.weight
        else:
            label_list[label] = example.weight
        if label_list[label] > biggest_amount:
            biggest_label = label
            biggest_amount = label_list[label]
    return biggest_label


def most_common_attribute(example_list: list, attr):
    attr_list = list()
    for sample in example_list:
        value = sample.attributes[attr]
        if not value == "unknown":
            attr_list.append(sample.attributes[attr])
    count = Counter(attr_list)
    return count.most_common(1)[0][0]


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


def load_data(file, unknown_replace=False):
    result = list()
    with open(file, 'r') as dataset:
        for line in dataset:
            data = line.strip().split(',')
            label = data[-1]
            attr = data[:-1]
            result.append(Example(attr, label))
    if not unknown_replace:
        return result
    most_commons = list()
    attribute_length = len(result[0].attributes)
    for i in range(attribute_length):
        most_commons.append(most_common_attribute(result, i))
    for sample in result:
        for i in range(attribute_length):
            value = sample.attributes[i]
            if value == "unknown":
                sample.attributes[i] = most_commons[i]
    return result


if __name__ == "__main__":
    examples = list()
    print("Hello Human\nI am a learning robot. Behold my training prowess:\n")

    # initialize training and testing data
    training_examples_car = load_data("car/train.csv")
    training_examples_bank = load_data("bank/train.csv")
    training_examples_bank_unknown = load_data("bank/train.csv", True)

    testing_examples_car = load_data("car/test.csv")
    testing_examples_bank = load_data("bank/test.csv")
    testing_examples_bank_unknown = load_data("bank/test.csv", True)

    car_val, car_num = attr_values(training_examples_car)
    bank_val, bank_num = attr_values(training_examples_bank)
    bank_val_u, bank_num_u = attr_values(training_examples_bank_unknown)

    # initialize string formatting options
    cat_format_string = "{:7s}{:>10s}{:>10s}{:>10s}\n"
    num_format_string = "{:<7d}{:10.3f}{:10.3f}{:10.3f}\n"
    cat_format_string_tex = "{:s}&{:s}&{:s}&{:s}\\\\\\hline\\hline\n"
    num_format_string_tex = "{:d}&{:.3f}&{:.3f}&{:.3f}\\\\\\hline\n"

    # Begin part 2 tests:----------------------------------------------------------------------------
    car_train_err_results = cat_format_string.format("strat", "entropy", "gin", "max")
    car_test_err_results = cat_format_string.format("strat", "entropy", "gin", "max")
    car_train_err_results_tex = cat_format_string_tex.format("strat", "entropy", "gin", "max")
    car_test_err_results_tex = cat_format_string_tex.format("strat", "entropy", "gin", "max")

    for i in range(1, 7):
        tree_ent = DecisionTree(training_examples_car, i, entropy_weighted, car_val, car_num)
        tree_gin = DecisionTree(training_examples_car, i, gini, car_val, car_num)
        tree_max = DecisionTree(training_examples_car, i, max_error, car_val, car_num)

        ent_train_res = tree_ent.test(training_examples_car)
        gin_train_res = tree_gin.test(training_examples_car)
        max_train_res = tree_max.test(training_examples_car)
        ent_test_res = tree_ent.test(testing_examples_car)
        gin_test_res = tree_gin.test(testing_examples_car)
        max_test_res = tree_max.test(testing_examples_car)

        car_train_err_results += num_format_string.format(i, ent_train_res, gin_train_res, max_train_res)
        car_test_err_results += num_format_string.format(i, ent_test_res, gin_test_res, max_test_res)
        car_train_err_results_tex += num_format_string_tex.format(i, ent_train_res, gin_train_res, max_train_res)
        car_test_err_results_tex += num_format_string_tex.format(i, ent_test_res, gin_test_res, max_test_res)

    print("Part 2(b):\ncar training dataset, varying strategy (col) and depth(row):")
    print(car_train_err_results)
    # print(car_train_err_results_tex)
    print("car testing dataset:")
    print(car_test_err_results)
    # print(car_test_err_results_tex)

    # Part 3(a) tests:------------------------------------------------------------------------------------

    print("Calculating Trees for bank data (this may take a minute)\n")

    bank_train_err_results = cat_format_string.format("strat", "entropy", "gin", "max")
    bank_test_err_results = cat_format_string.format("strat", "entropy", "gin", "max")
    bank_train_err_results_tex = cat_format_string_tex.format("strat", "entropy", "gin", "max")
    bank_test_err_results_tex = cat_format_string_tex.format("strat", "entropy", "gin", "max")

    for i in range(1, 17):
        tree_ent = DecisionTree(training_examples_bank, i, entropy, bank_val, bank_num)
        tree_gin = DecisionTree(training_examples_bank, i, gini, bank_val, bank_num)
        tree_max = DecisionTree(training_examples_bank, i, max_error, bank_val, bank_num)

        ent_train_res = tree_ent.test(training_examples_bank)
        gin_train_res = tree_gin.test(training_examples_bank)
        max_train_res = tree_max.test(training_examples_bank)
        ent_test_res = tree_ent.test(testing_examples_bank)
        gin_test_res = tree_gin.test(testing_examples_bank)
        max_test_res = tree_max.test(testing_examples_bank)

        bank_train_err_results += num_format_string.format(i, ent_train_res, gin_train_res, max_train_res)
        bank_test_err_results += num_format_string.format(i, ent_test_res, gin_test_res, max_test_res)
        bank_train_err_results_tex += num_format_string_tex.format(i, ent_train_res, gin_train_res, max_train_res)
        bank_test_err_results_tex += num_format_string_tex.format(i, ent_test_res, gin_test_res, max_test_res)

    print("Part 3(b):\nbank training dataset:")
    print(bank_train_err_results)
    # print(bank_train_err_results_tex)
    print("bank testing dataset:")
    print(bank_test_err_results)
    # print(bank_test_err_results_tex)

    # Part 3(b):----------------------------------------------------------------------------------

    print("Calculating Trees for bank data, replace unknown (this may take a minute)\n")

    bank_train_err_results_unk = cat_format_string.format("strat", "entropy", "gin", "max")
    bank_test_err_results_unk = cat_format_string.format("strat", "entropy", "gin", "max")
    bank_train_err_results_unk_tex = cat_format_string_tex.format("strat", "entropy", "gin", "max")
    bank_test_err_results_unk_tex = cat_format_string_tex.format("strat", "entropy", "gin", "max")

    for i in range(1, 17):
        tree_ent = DecisionTree(training_examples_bank_unknown, i, entropy, bank_val, bank_num)
        tree_gin = DecisionTree(training_examples_bank_unknown, i, gini, bank_val, bank_num)
        tree_max = DecisionTree(training_examples_bank_unknown, i, max_error, bank_val, bank_num)

        ent_train_res = tree_ent.test(training_examples_bank)
        gin_train_res = tree_gin.test(training_examples_bank)
        max_train_res = tree_max.test(training_examples_bank)
        ent_test_res = tree_ent.test(testing_examples_bank)
        gin_test_res = tree_gin.test(testing_examples_bank)
        max_test_res = tree_max.test(testing_examples_bank)

        bank_train_err_results_unk += num_format_string.format(i, ent_train_res, gin_train_res, max_train_res)
        bank_test_err_results_unk += num_format_string.format(i, ent_test_res, gin_test_res, max_test_res)
        bank_train_err_results_unk_tex += num_format_string_tex.format(i, ent_train_res, gin_train_res, max_train_res)
        bank_test_err_results_unk_tex += num_format_string_tex.format(i, ent_test_res, gin_test_res, max_test_res)

    print("Part 3(c):\nbank training dataset, replace unknown:")
    print(bank_train_err_results_unk)
    # print(bank_train_err_results_unk_tex)
    print("bank testing dataset, replace unknown:")
    print(bank_test_err_results_unk)
    # print(bank_test_err_results_unk_tex)

    print("---------TESTS COMPLETE--------------")
