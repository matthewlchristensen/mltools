from LinearRegression.regression import *
from EnsembleLearning.ensemble import *
from random import sample

DO_EVERYTHING = True
training_examples_bank = load_data("EnsembleLearning/bank/train.csv")
training_examples_bank2 = load_data("EnsembleLearning/bank/train.csv")
testing_examples_bank = load_data("EnsembleLearning/bank/test.csv")
train_size = len(training_examples_bank)
for example in training_examples_bank:
    example.weight = 1 / train_size
bank_vals, bank_nums = attr_values(training_examples_bank)


print("Training ADABoost. Strap in, this will take a while\n")
booster = ADABoostLearner(training_examples_bank, 100, bank_vals, bank_nums)

ada_header = "{:<7s}{:>11s}{:>11s}{:>11s}{:>11s}"
ada_print = "{:<7d}{:11.3f}{:11.3f}{:11.3f}{:11.3f}"
print(ada_header.format("index", "train err", "test err", "tree train", "tree test"))

for i in range(1, 100, 5):
    train_error = booster.test(training_examples_bank2, i)
    test_error = booster.test(testing_examples_bank, i)
    member_train_error = booster.the_board[i].tree.test(training_examples_bank2)
    member_test_error = booster.the_board[i].tree.test(testing_examples_bank)
    print(ada_print.format(i, train_error, test_error, member_train_error, member_test_error))

bag_header = "{:<7s}{:>11s}{:>11s}"
bag_print = "{:<7d}{:11.3f}{:11.3f}"

print("Training BagBoost. Please hold.")
bag_boosted = BagBooster(training_examples_bank2, 100, 1000, bank_vals, bank_nums)
print(bag_header.format("index", "train err", "test err"))
for i in range(1, 100, 5):
    bag_train_error = bag_boosted.test(training_examples_bank2, i)
    bag_test_error = bag_boosted.test(testing_examples_bank, i)
    print(bag_print.format(i, bag_train_error, bag_test_error))

forest_header = "{:<7s}{:>22s}{:>22s}{:>22s}"
forest_print = "{:<7d}{:11.3f}{:11.3f}{:11.3f}{:11.3f}{:11.3f}{:11.3f}"

print("Training the Forests. This one is another doozy")
forest2 = ForestBooster(training_examples_bank2, 100, 2, bank_vals, bank_nums)
print("1 forest down")
forest4 = ForestBooster(training_examples_bank2, 100, 4, bank_vals, bank_nums)
print("2 forests down")
forest6 = ForestBooster(training_examples_bank2, 100, 6, bank_vals, bank_nums)
print(forest_header.format("index", "2 train and test err", "4 train and test err", "6 train and test err"))
for i in range(1, 100, 5):
    forest2_train_error = forest2.test(training_examples_bank2, i)
    forest2_test_error = forest2.test(testing_examples_bank, i)
    forest4_train_error = forest4.test(training_examples_bank2, i)
    forest4_test_error = forest4.test(testing_examples_bank, i)
    forest6_train_error = forest6.test(training_examples_bank2, i)
    forest6_test_error = forest6.test(testing_examples_bank, i)
    print(forest_print.format(i, forest2_train_error, forest2_test_error, forest4_train_error, forest4_test_error,
                              forest6_train_error, forest6_test_error))




#Bag/forest/tree comparison ----------------------------------------------------------------------------------

bags = list()
forests = list()
trees = list()
rnd_trees = list()
NUM_TREES = 100

for i in range(100):
    print(".", end="", flush=True)
    subset = sample(training_examples_bank2, 1000)
    bag_i = BagBooster(subset, NUM_TREES, 100, bank_vals, bank_nums)
    forest_i = ForestBooster(subset, NUM_TREES, 4, bank_vals, bank_nums)
    tree_i = bag_i.the_bag[0]
    rnd_tree_i = forest_i.the_bag[0]
    predictor_results_bag = list()
    predictor_results_tree = list()
    predictor_results_forest = list()
    predictor_results_rnd_tree = list()
    for example in testing_examples_bank:
        predictor_results_bag.append(1 if bag_i.evaluate(example, NUM_TREES) == "yes" else 0)
        predictor_results_forest.append(1 if forest_i.evaluate(example, NUM_TREES) == "yes" else 0)
        predictor_results_tree.append(1 if tree_i.predict(example) == "yes" else 0)
        predictor_results_rnd_tree.append(1 if rnd_tree_i.predict(example) == "yes" else 0)
    bags.append(predictor_results_bag)
    forests.append(predictor_results_forest)
    trees.append(predictor_results_tree)
    rnd_trees.append(predictor_results_rnd_tree)

tree_bias = 0
bags_bias = 0
forests_bias = 0
rnd_tree_bias = 0
tree_var = 0
bags_var = 0
forests_var = 0
rnd_tree_var = 0

for j in range(len(testing_examples_bank)):
    avg_bags = 0
    avg_tree = 0
    avg_forests = 0
    avg_rnd_tree = 0
    example = testing_examples_bank[j]

    for i in range(100):
        forest = forests[i]
        bag = bags[i]
        tree = trees[i]
        rnd_tree = rnd_trees[i]
        avg_tree += tree[j]
        avg_bags += bag[j]
        avg_forests += forest[j]
        avg_rnd_tree += rnd_tree[j]

    avg_bags /= 100
    avg_tree /= 100
    avg_forests /= 100
    avg_rnd_tree /= 100

    label_val = 1 if example.label == "yes" else 0

    tree_bias += math.pow(avg_tree - label_val, 2)
    bags_bias += math.pow(avg_bags - label_val, 2)
    forests_bias += math.pow(avg_forests - label_val, 2)
    rnd_tree_bias += math.pow(avg_rnd_tree - label_val, 2)

    var_bags_i = 0
    var_tree_i = 0
    var_rnd_tree_i = 0
    var_forests_i = 0

    for i in range(100):
        forest = forests[i]
        bag = bags[i]
        tree = trees[i]
        rnd_tree = rnd_trees[i]
        var_tree_i += math.pow(avg_tree - tree[j], 2)
        var_bags_i += math.pow(avg_bags - bag[j], 2)
        var_forests_i += math.pow(avg_forests - forest[j], 2)
        var_rnd_tree_i += math.pow(avg_rnd_tree - rnd_tree[j], 2)
    var_bags_i /= (100-1)
    var_tree_i /= (100-1)
    var_rnd_tree_i /= (100-1)
    var_forests_i /= (100-1)

    tree_var += var_tree_i
    bags_var += var_bags_i
    forests_var += var_forests_i
    rnd_tree_var += var_rnd_tree_i

n = len(training_examples_bank2)

tree_bias /= n
bags_bias /= n
forests_bias /= n
rnd_tree_bias /= n
tree_var /= n
bags_var /= n
forests_var /= n
rnd_tree_var /= n

print("\nFinal Results")
print("Type   bias      variance  error")
results_format = "{:7s}{:10.4f}{:10.4f}{:10.4f}"
print(results_format.format("tree", tree_bias, tree_var, tree_bias + tree_var))
print(results_format.format("bag", bags_bias, bags_var, bags_bias + bags_var))
print(results_format.format("rnd_tree", rnd_tree_bias, rnd_tree_var, rnd_tree_bias + rnd_tree_var))
print(results_format.format("forests", forests_bias, forests_var, forests_bias + forests_var))


#Regression ----------------------------------------------------------------------------------

training_examples_concrete = load_data("LinearRegression/concrete/train.csv")
testing_examples_concrete = load_data("LinearRegression/concrete/test.csv")

batcher = BatchReg(training_examples_concrete, .01)
print("Batch final weights: ", batcher.weight_vector)
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

