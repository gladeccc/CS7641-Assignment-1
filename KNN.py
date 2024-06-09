from sklearn import svm
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import time

import warnings

warnings.filterwarnings('ignore')

# KNN
def plot_n_neighbors(X_train, y_train, X_test, y_test, X_less, y_less, p, weights, algorithm, Range, dataset):
    print(plot_n_neighbors)

    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        print(epoch)
        KNN = KNeighborsClassifier(n_neighbors=Range[epoch], p=p, weights=weights, algorithm=algorithm)
        KNN.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, KNN.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, KNN.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, KNN.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, KNN.predict(X_less), average="weighted")
    print("std")
    print("1.1, 2.2")
    print(np.std(score_training))
    print(np.std(score_validation))
    print("1.2")
    print(np.std(score_test))
    print("2.1")
    print(np.std(score_less))
    print("difference")
    print("1.4, 2.4")
    print(np.abs(np.mean(score_training - score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_validation, label='Validation Score')
        plt.plot(Range, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_test, label='Test Score')
        plt.plot(Range, score_less, label='Score for Imbalanced Dataset')
    plt.legend()
    plt.title("N Neighbors " + dataset + " (KNN)")
    plt.xlabel("N Neighbors")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('KNN/' + dataset + '_KNN_validation_curve_1.png')
    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("N Neighbors " + dataset + " (KNN)")
        plt.xlabel("N Neighbors")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('KNN/' + dataset + '_KNN_validation_curve_1_2.png')


def plot_power(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, weights, algorithm, Range, dataset):
    print(plot_power)
    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        KNN = KNeighborsClassifier(n_neighbors=neighbors, p=Range[epoch], weights=weights, algorithm=algorithm)
        KNN.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, KNN.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, KNN.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, KNN.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, KNN.predict(X_less), average="weighted")
    print("std")
    print("1.1, 2.2")
    print(np.std(score_training))
    print(np.std(score_validation))
    print("1.2")
    print(np.std(score_test))
    print("2.1")
    print(np.std(score_less))
    print("difference")
    print("1.4, 2.4")
    print(np.abs(np.mean(score_training - score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_validation, label='Validation Score')
        plt.plot(Range, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_test, label='Test Score')
        plt.plot(Range, score_less, label='Score for Imbalanced Dataset')
    plt.legend()
    plt.title("Power " + dataset + " (KNN)")
    plt.xlabel("Power")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('KNN/' + dataset + '_KNN_validation_curve_2.png')
    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Power " + dataset + " (KNN)")
        plt.xlabel("Power")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('KNN/' + dataset + '_KNN_validation_curve_2_2.png')


def best_params_KNN(X_train, y_train, y_test, X_test, X_less, y_less, classifier_knn):
    print(best_params_KNN)
    # param_grid = {'n_neighbors': [8], 'p': [1], "weights": ['uniform'], "algorithm": ['auto']}
    param_grid = {'n_neighbors': np.arange(0,10,2), 'p': [1, 2, 3], "weights": ['distance', 'uniform'],
                  "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute']}
    classifier_knn_best = GridSearchCV(classifier_knn, param_grid=param_grid, cv=4)

    start_time = time.time()
    classifier_knn_best.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time - start_time
    print("Best params for k-NN:", classifier_knn_best.best_params_)

    start_time = time.time()
    classifier_accuracy = f1_score(y_test, classifier_knn_best.predict(X_test),
                                           average="weighted")
    end_time = time.time()
    time_infer = end_time - start_time
    classifier_accuracy2 = f1_score(y_less, classifier_knn_best.predict(X_less),
                                           average="weighted")
    print("Accuracy for best k-NN:", classifier_accuracy)
    print(confusion_matrix(y_test, classifier_knn_best.predict(X_test)))
    return time_train, time_infer, classifier_accuracy, classifier_accuracy2,\
           classifier_knn_best.best_params_['n_neighbors'], classifier_knn_best.best_params_['p'], \
           classifier_knn_best.best_params_['weights'], classifier_knn_best.best_params_['algorithm']


def learning_curve_knn(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, weights, algorithm, dataset):
    print(learning_curve_knn)
    KNN = KNeighborsClassifier(n_neighbors=neighbors, p=p, weights=weights, algorithm=algorithm)

    epochs = 10
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)

    for epoch in range(epochs):
        X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train,
                                                                                  test_size=np.linspace(0.1, 0.99, 10)[
                                                                                      epoch],
                                                                                  random_state=9)
        KNN.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, KNN.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, KNN.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, KNN.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, KNN.predict(X_less), average="weighted")
    print("std")
    print("1.1, 2.2")
    print(np.std(score_training))
    print(np.std(score_validation))
    print("1.2")
    print(np.std(score_test))
    print("2.1")
    print(np.std(score_less))
    print("difference")
    print("1.4, 2.4")
    print(np.abs(np.mean(score_training - score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_training, label='Train Score')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_validation, label='Validation Score')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_training, label='Train Score')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_less, label='Score for Imbalanced Dataset')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_test, label='Test Score')
    plt.legend()
    plt.title("Learning Curve " + dataset + " (KNN)")
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("F1 Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()
    plt.savefig('KNN/' + dataset + '_KNN_learning_curve.png')

    if dataset == 'Credit':
        plt.figure()
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_test, label='Test Score')
        plt.legend()
        plt.title("Learning Curve " + dataset + " (KNN)")
        plt.xlabel("Percentage of Training Examples")
        plt.ylabel("F1 Score")
        plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
        plt.grid()
        plt.savefig('KNN/' + dataset + '_KNN_learning_curve_2.png')

def plot_weight(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, algorithm, Range, dataset):
    print(plot_weight)
    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        KNN = KNeighborsClassifier(n_neighbors=neighbors, p=p, weights=Range[epoch], algorithm=algorithm)
        KNN.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, KNN.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, KNN.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, KNN.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, KNN.predict(X_less), average="weighted")
    print("std")
    print("1.1, 2.2")
    print(np.std(score_training))
    print(np.std(score_validation))
    print("1.2")
    print(np.std(score_test))
    print("2.1")
    print(np.std(score_less))
    print("difference")
    print("1.4, 2.4")
    print(np.abs(np.mean(score_training - score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_validation, label='Validation Score')
        plt.plot(Range, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_test, label='Test Score')
        plt.plot(Range, score_less, label='Score for Imbalanced Dataset')
    plt.legend()
    plt.title("Weights " + dataset + " (KNN)")
    plt.xlabel("Weights")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('KNN/' + dataset + '_KNN_validation_curve_3.png')
    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Weights " + dataset + " (KNN)")
        plt.xlabel("Weights")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('KNN/' + dataset + '_KNN_validation_curve_3_2.png')

def plot_algo(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, weigth, Range, dataset):
    print(plot_algo)
    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        KNN = KNeighborsClassifier(n_neighbors=neighbors, p=p, weights=weigth, algorithm=Range[epoch])
        KNN.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, KNN.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, KNN.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, KNN.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, KNN.predict(X_less), average="weighted")
    print("std")
    print("1.1, 2.2")
    print(np.std(score_training))
    print(np.std(score_validation))
    print("1.2")
    print(np.std(score_test))
    print("2.1")
    print(np.std(score_less))
    print("difference")
    print("1.4, 2.4")
    print(np.abs(np.mean(score_training - score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_validation, label='Validation Score')
        plt.plot(Range, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.plot(Range, score_training, label='Train Score')
        plt.plot(Range, score_test, label='Test Score')
        plt.plot(Range, score_less, label='Score for Imbalanced Dataset')
    plt.legend()
    plt.title("Algo " + dataset + " (KNN)")
    plt.xlabel("Algo")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('KNN/' + dataset + '_KNN_validation_curve_4.png')
    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Algo " + dataset + " (KNN)")
        plt.xlabel("Algo")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('KNN/' + dataset + '_KNN_validation_curve_4_2.png')