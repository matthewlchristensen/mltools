from DecisionTree.dtree import *
from math import log, exp
from random import choice


SWARM_DEPTH = 1


class BoardMember:
    def __init__(self, tree, shares):
        self.tree = tree
        self.shares = shares

    def vote(self, example):
        answer = self.tree.predict(example)
        return answer, self.shares


class ADABoostLearner:
    def __init__(self, training_set, stump_num, attr_vals, num_data):
        self.the_board = list()
        for t in range(stump_num):
            tree_t = DecisionTree(training_set, SWARM_DEPTH, entropy_weighted, attr_vals, num_data)
            error_t = tree_t.test(training_set)
            a_t = 1/2*log((1-error_t)/error_t)
            z_t = 0
            for example in training_set:
                example.weight *= exp(-1 * a_t) if (tree_t.predict(example) == example.label) else exp(a_t)
                z_t += example.weight
            for example in training_set:
                example.weight /= z_t
            self.the_board.append(BoardMember(tree_t, a_t))

    def evaluate_example(self, datum, num_board_members):
        result = dict()
        plurality_decision = ""
        decision_votes = 0
        for member in self.the_board[:num_board_members]:
            prediction, weight = member.vote(datum)
            if prediction in result:
                result[prediction] += weight
            else:
                result[prediction] = weight
            if result[prediction] > decision_votes:
                decision_votes = result[prediction]
                plurality_decision = prediction
        return plurality_decision

    def test(self, test_data, num_board_members):
        mistakes = 0
        total = len(test_data)
        for datum in test_data:
            if self.evaluate_example(datum, num_board_members) != datum.label:
                mistakes += 1
        return mistakes / total


class BagBooster:
    def __init__(self, training_set, bag_size, subset_size, attr_vals, num_data):
        self.the_bag = list()
        for t in range(bag_size):
            subset = [choice(training_set) for _ in range(subset_size)]
            tree_t = DecisionTree(subset, len(attr_vals), entropy_weighted, attr_vals, num_data)
            self.the_bag.append(tree_t)

    def evaluate(self, example, bag_count):
        result = dict()
        plurality_decision = ""
        decision_votes = 0
        for stump in self.the_bag[:bag_count]:
            prediction = stump.predict(example)
            if prediction in result:
                result[prediction] += 1
            else:
                result[prediction] = 1
            if result[prediction] > decision_votes:
                decision_votes = result[prediction]
                plurality_decision = prediction
        return plurality_decision

    def test(self, testing_set, bag_count):
        mistakes = 0
        total = len(testing_set)
        for datum in testing_set:
            if self.evaluate(datum, bag_count) != datum.label:
                mistakes += 1
        return mistakes / total


class ForestBooster:
    def __init__(self, training_set, num_trees, subset_size, attr_vals, num_data):
        self.the_bag = list()
        for t in range(num_trees):
            tree_t = DecisionTree(training_set, len(attr_vals), entropy_weighted, attr_vals, num_data,
                                  use_random_subset=True, random_subset_size=subset_size)
            self.the_bag.append(tree_t)

    def evaluate(self, example, bag_count):
        result = dict()
        plurality_decision = ""
        decision_votes = 0
        for stump in self.the_bag[:bag_count]:
            prediction = stump.predict(example)
            if prediction in result:
                result[prediction] += 1
            else:
                result[prediction] = 1
            if result[prediction] > decision_votes:
                decision_votes = result[prediction]
                plurality_decision = prediction
        return plurality_decision

    def test(self, testing_set, bag_count):
        mistakes = 0
        total = len(testing_set)
        for datum in testing_set:
            if self.evaluate(datum, bag_count) != datum.label:
                mistakes += 1
        return mistakes / total


if __name__ == "__main__":
    training_examples_bank = load_data("bank/train.csv")
    training_examples_bank2 = load_data("bank/train.csv")
    testing_examples_bank = load_data("bank/test.csv")
    train_size = len(training_examples_bank)
    for example in training_examples_bank:
        example.weight = 1/train_size
    bank_vals, bank_nums = attr_values(training_examples_bank)
    print("Training ADABoost. Strap in, this will take a while\n")
    booster = ADABoostLearner(training_examples_bank, 100, bank_vals, bank_nums)

    ada_print = "{:<7d}{:10.3f}{:10.3f}{:10.3f}{:10.3f}"

    for i in range(1, 100, 20):
        train_error = booster.test(training_examples_bank2, i)
        test_error = booster.test(testing_examples_bank, i)
        member_train_error = booster.the_board[i].tree.test(training_examples_bank2)
        member_test_error = booster.the_board[i].tree.test(testing_examples_bank)
        print(ada_print.format(i, train_error, test_error, member_train_error, member_test_error))

    bag_print = "{:<7d}{:10.3f}{:10.3f}"

    print("Training BagBoost. Please hold.")
    bag_boosted = BagBooster(training_examples_bank2, 100, 1000, bank_vals, bank_nums)
    for i in range(1, 100, 20):
        bag_train_error = bag_boosted.test(training_examples_bank2, i)
        bag_test_error = bag_boosted.test(testing_examples_bank, i)
        print(bag_print.format(i, bag_train_error, bag_test_error))

    forest_print = "{:<7d}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}{:10.3f}"

    print("Training the Forests. This one is another doozy")
    forest2 = ForestBooster(training_examples_bank2, 100, 2, bank_vals, bank_nums)
    forest4 = ForestBooster(training_examples_bank2, 100, 4, bank_vals, bank_nums)
    forest6 = ForestBooster(training_examples_bank2, 100, 6, bank_vals, bank_nums)
    for i in range(1, 100, 20):
        forest2_train_error = forest2.test(training_examples_bank2, i)
        forest2_test_error = forest2.test(testing_examples_bank, i)
        forest4_train_error = forest4.test(training_examples_bank2, i)
        forest4_test_error = forest4.test(testing_examples_bank, i)
        forest6_train_error = forest6.test(training_examples_bank2, i)
        forest6_test_error = forest6.test(testing_examples_bank, i)
        print(forest_print.format(i, forest2_train_error, forest2_test_error, forest4_train_error, forest4_test_error,
                                  forest6_train_error, forest6_test_error))

