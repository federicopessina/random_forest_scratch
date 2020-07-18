import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool
from operator import itemgetter
import matplotlib.pyplot as plt
import itertools as it
from random import sample
import time
np.seterr(divide='ignore', invalid='ignore')  # ignore Runtime Warning about divide
plt.style.use('seaborn-whitegrid')

#Utils

def one_hot_encoder(labels):
    target = []
    for label in labels:
        target.append(np.array([np.float64(1) if i==label else np.float64(0) for i in range(10)]))
    return np.array(target)    

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred)/len(y_test)

def cross_val(dataset,k):
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    n_fold = int(len(dataset)/k)
    folds = []
    for progress in range(n_fold,len(dataset)+n_fold,n_fold):
        folds.append(dataset[-n_fold:])
        dataset.drop(range(len(dataset)-progress))
    return folds

def nested_cross_val(dataset,k,j,hp):
    ds = cross_val(dataset,k)
    acc_ext = []
    all_hp = sorted(hp)
    combinations = it.product(*(hp[name] for name in all_hp))
    all_hp_list = list(combinations)
    for i in range(k):
        test_sub_cross = ds[i]
        train_sub_cross = pd.concat([x for f,x in enumerate(ds) if f!=i], axis=0)
        sub_cross = cross_val(train_sub_cross,j)
        acc_int = []
        for a in range(j):
            test = sub_cross[a]
            train = pd.concat([x for f,x in enumerate(sub_cross) if f!=a], axis=0)
            data_train = train.values
            data_test = test.values
            x_train, y_train, x_test, y_test = data_train[:, 1:], data_train[:, 0], data_test[:, 1:], data_test[:, 0]
            for hyperp in all_hp_list:
                rf = RandomForest(n_classifiers=30,subsample_size=hyperp[2],min_child_in_leaf=hyperp[0],min_to_split=hyperp[1])
                rf.fit(x_train, y_train)
                y_pred = rf.predict(x_test)
                acc = accuracy(y_test, y_pred) 
                acc_int.append((acc,hyperp))
                #print(f"------ hps: {hyperp} -- Nfold_int_{a} : {acc} ")
        best_hp_acc_int = max(acc_int,key=itemgetter(0))
        #print(f"---- best internal {i} fold acc: {best_hp_acc_int[0]} -- hyperparams {best_hp_acc_int[1]}")
        data_train = train_sub_cross.values
        data_test = test_sub_cross.values
        x_train, y_train, x_test, y_test = data_train[:, 1:], data_train[:, 0], data_test[:, 1:], data_test[:, 0]
        rf = RandomForest(n_classifiers=30,subsample_size=best_hp_acc_int[1][2],min_child_in_leaf=best_hp_acc_int[1][0],min_to_split=best_hp_acc_int[1][1])  # optimal 100 trees
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        acc = accuracy(y_test, y_pred) 
        acc_ext.append((acc,best_hp_acc_int[1]))
        #print(f"-- Nfold_ext_{i} : {acc} ")
    best_hp_acc_ext = max(acc_ext,key=itemgetter(0))
    print(f"best external fold acc: {best_hp_acc_ext[0]} -- hyperparams {best_hp_acc_ext[1]}")
    return best_hp_acc_ext[1]


class TreeNode:
    def __init__(self, n_features, min_child_in_leaf, min_to_split, max_depth):
        self.n_features = n_features
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.split_gini = 1
        self.label = None
        self.max_depth = max_depth
        self.min_child_in_leaf = min_child_in_leaf
        self.min_to_split = min_to_split

    def is_leaf(self):
        return self.label is not None

    """ use 2d array (matrix) to compute gini index. Numerical feature values only """
    def gini(self, f, y, target):
        trans = f.reshape(len(f), -1)  # transpose 1d np array
        a = np.concatenate((trans, target), axis=1)  # vertical concatenation
        a = a[a[:, 0].argsort()]  # sort by column 0, feature values
        sort = a[:, 0]
        split = (sort[0:-1] + sort[1:]) / 2  # compute possible split values

        left, right = np.array([split]), np.array([split])
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        # count occurrence of labels for each possible split value
        for i in range(n_classes):
            temp = a[:, -n_classes + i].cumsum()[:-1]
            left = np.vstack((left, temp))  # horizontal concatenation
            right = np.vstack((right, counts[i] - temp))

        sum_1 = left[1:, :].sum(axis=0)  # sum occurrence of labels
        sum_2 = right[1:, :].sum(axis=0)
        n = len(split)
        gini_t1, gini_t2 = [1] * n, [1] * n
        # calculate left and right gini
        for i in range(n_classes):
            gini_t1 -= (left[i + 1, :] / sum_1) ** 2
            gini_t2 -= (right[i + 1, :] / sum_2) ** 2
        s = sum(counts)
        g = gini_t1 * sum_1 / s + gini_t2 * sum_2 / s
        g = list(g)
        min_g = min(g)
        split_value = split[g.index(min_g)]
        return split_value, min_g

    def split_feature_value(self, x, y, target):
        # compute gini index of every column
        n = x.shape[1]  # number of x columns
        sub_features = sample(range(n), self.n_features)  # feature sub-space
        # list of (split_value, split_gini) tuples
        value_g = [self.gini(x[:, i], y, target) for i in sub_features]
        result = min(value_g, key=lambda t: t[1])  # (value, gini) tuple with min gini
        feature = sub_features[value_g.index(result)]  # feature with min gini
        return feature, result[0], result[1]  # split feature, value, gini

    # recursively grow the tree
    def attempt_split(self, x, y, target,depth=0):
        c = Counter(y)
        majority = c.most_common()[0]  # majority class and count
        label, count = majority[0], majority[1]
        if len(y) < self.min_to_split or len(c) == 1 or count/len(y) > 0.9 or depth > self.max_depth:  # stop criterion
            self.label = label  # set leaf
            return
        # split feature, value, gini
        feature, value, split_gini = self.split_feature_value(x, y, target)
        # stop split when gini decrease smaller than some threshold
        if self.split_gini - split_gini < 0.01:  # stop criterion
            self.label = label  # set leaf
            return
        index1 = x[:, feature] <= value
        index2 = x[:, feature] > value
        x1, y1, x2, y2 = x[index1], y[index1], x[index2], y[index2]
        target1, target2 = target[index1], target[index2]
        if len(y2) < self.min_child_in_leaf or len(y1) < self.min_child_in_leaf or len(y2) == 0 or len(y1) == 0:  # stop split
            self.label = label  # set leaf
            return
        # splitting procedure
        depth+=1
        self.split_feature = feature
        self.split_value = value
        self.split_gini = split_gini
        self.left_child, self.right_child = TreeNode(self.n_features, self.min_child_in_leaf, self.min_to_split, self.max_depth), TreeNode(self.n_features, self.min_child_in_leaf, self.min_to_split, self.max_depth)
        self.left_child.split_gini, self.right_child.split_gini = split_gini, split_gini
        self.left_child.attempt_split(x1, y1, target1,depth)
        self.right_child.attempt_split(x2, y2, target2,depth)


    # trace down the tree for each data instance, for prediction
    def sort(self, x):  # x is 1d array
        if self.label is not None:
            return self.label
        if x[self.split_feature] <= self.split_value:
            return self.left_child.sort(x)
        else:
            return self.right_child.sort(x)


class ClassifierTree:
    def __init__(self, n_features,min_child_in_leaf, min_to_split,max_depth):
        self.root = TreeNode(n_features,min_child_in_leaf, min_to_split,max_depth)

    def train(self, x, y):
        # one hot encoded target is for gini index calculation
        labels = y.reshape(len(y), -1) # transpose 1d np array
        target = one_hot_encoder(labels) 
        self.root.attempt_split(x, y, target)

    def classify(self, x):  # x is 2d array
        return [self.root.sort(x[i]) for i in range(x.shape[0])]


class RandomForest:
    def __init__(self, n_classifiers=30,subsample_size=1000,n_features=25,min_child_in_leaf=0,min_to_split=2,max_depth=10):
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.subsample_size = subsample_size
        self.n_features = n_features
        self.min_child_in_leaf = min_child_in_leaf
        self.min_to_split = min_to_split
        self.max_depth = max_depth
        self.x = None
        self.y = None

    def build_tree(self, tree):
        ids = np.random.choice(len(self.y), self.subsample_size)
        tree.train(self.x[ids], self.y[ids])
        return tree  # return tree for multiprocessing pool

    def fit(self, x, y):
        self.x, self.y = x, y
        #n_select_features = int(np.sqrt(x.shape[1]))  # number of features
        for _ in range(self.n_classifiers):
            tree = ClassifierTree(self.n_features,self.min_child_in_leaf,self.min_to_split,self.max_depth)
            self.classifiers.append(tree)
        # multiprocessing pool
        pool = Pool()
        self.classifiers = pool.map(self.build_tree, self.classifiers)
        pool.close()
        pool.join()

    def predict(self, x_test):  # ensemble
        pred = [tree.classify(x_test) for tree in self.classifiers]
        pred = np.array(pred)
        result = [Counter(pred[:, i]).most_common()[0][0] for i in range(pred.shape[1])]
        return result