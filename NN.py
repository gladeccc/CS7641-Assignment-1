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


# Neural Network
def plot_alpha(X_train, y_train, X_test, y_test, X_less, y_less, learning_rate_init, max_iter, layer_size, Range,
               dataset):
    print(plot_alpha)
    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        classifier_neural_network1 = MLPClassifier(hidden_layer_sizes=(layer_size,), random_state=9,
                                                   max_iter=max_iter,
                                                   alpha=Range[epoch],
                                                   learning_rate_init=learning_rate_init)
        classifier_neural_network1.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, classifier_neural_network1.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, classifier_neural_network1.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, classifier_neural_network1.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, classifier_neural_network1.predict(X_less), average="weighted")
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
    print(np.abs(np.mean(score_training-score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.semilogx(Range, score_training, label='Train Score')
        plt.semilogx(Range, score_validation, label='Validation Score')
        plt.semilogx(Range, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.semilogx(Range, score_training, label='Train Score')
        plt.semilogx(Range, score_test, label='Test Score')
        plt.semilogx(Range, score_less, label='Score for Imbalanced Dataset')
    plt.legend()
    plt.title("Alpha " + dataset+ " (Neural Network)" )
    plt.xlabel("Alpha")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('NN/' + dataset + '_neural_network_validation_curve_1.png')

    if dataset == 'Credit':
        plt.figure()
        plt.semilogx(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Alpha " + dataset + " (Neural Network)")
        plt.xlabel("Alpha")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('NN/' + dataset + '_neural_network_validation_curve_1_2.png')


def plot_learning_rate_NN(X_train, y_train, X_test, y_test, X_less, y_less, alpha, max_iter, layer_size, Range,
                          dataset):
    print(plot_learning_rate_NN)
    # train_scores, test_scores = validation_curve(classifier_neural_network, X_train, y_train,
    #                                              param_name="learning_rate_init", param_range=np.logspace(-5, 0, 15),
    #                                              cv=4,scoring='f1')
    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        classifier_neural_network1 = MLPClassifier(hidden_layer_sizes=(layer_size,), random_state=9,
                                                   max_iter=max_iter,
                                                   alpha=alpha,
                                                   learning_rate_init=Range[epoch])
        classifier_neural_network1.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, classifier_neural_network1.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, classifier_neural_network1.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, classifier_neural_network1.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, classifier_neural_network1.predict(X_less), average="weighted")
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
    print(np.abs(np.mean(score_training-score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.semilogx(Range, score_training, label='Train Score')
        plt.semilogx(Range, score_validation, label='Validation Score')
        plt.semilogx(Range, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.semilogx(Range, score_training, label='Train Score')
        plt.semilogx(Range, score_test, label='Test Score')
        plt.semilogx(Range, score_less, label='Score for Imbalanced Dataset')
    plt.legend()
    plt.title("Learning Rate " + dataset+" (Neural Network)")
    plt.xlabel("Learning Rate")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('NN/' + dataset + '_neural_network_validation_curve_2.png')

    if dataset == 'Credit':
        plt.figure()
        plt.semilogx(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Learning Rate " + dataset+" (Neural Network)")
        plt.xlabel("Learning Rate")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('NN/' + dataset + '_neural_network_validation_curve_2_2.png')


def best_params_NN_ML(X_train, y_train, y_test, X_test, X_less, y_less, classifier_neural_network):
    print(best_params_NN_ML)
    param_grid = {'alpha': np.logspace(-5, -2, 5), 'learning_rate_init': np.logspace(-3, -1, 5)}
    classifier_neural_network_best = GridSearchCV(classifier_neural_network, param_grid=param_grid, cv=4)
    param_grid2 = {'hidden_layer_sizes': np.arange(5,20,5), 'max_iter': [220]}
    classifier_neural_network_best2 = GridSearchCV(classifier_neural_network, param_grid=param_grid2, cv=4)
    start_time = time.time()
    classifier_neural_network_best.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time - start_time
    print("Best params for neural network:", classifier_neural_network_best.best_params_)
    classifier_neural_network_best2.fit(X_train, y_train)
    print("Best params for neural network:", classifier_neural_network_best2.best_params_)
    start_time = time.time()
    classifier_accuracy = f1_score(y_test, classifier_neural_network_best.predict(X_test), average="weighted")
    end_time = time.time()
    time_infer = end_time - start_time
    print("F1 for best neural network:", classifier_accuracy)
    classifier_accuracy2 = f1_score(y_less, classifier_neural_network_best.predict(X_less), average="weighted")
    print(confusion_matrix(y_test, classifier_neural_network_best.predict(X_test)))
    return time_train, time_infer, classifier_accuracy, classifier_accuracy2,\
           classifier_neural_network_best.best_params_['learning_rate_init'], \
           classifier_neural_network_best.best_params_['alpha'], \
           classifier_neural_network_best2.best_params_['hidden_layer_sizes'], \
           classifier_neural_network_best2.best_params_['max_iter']


def learning_curve_NN(X_train, y_train, X_test, y_test, X_less, y_less, learning_rate_init, alpha, layer_size, max_iter,
                      dataset):
    print(learning_curve_NN)
    # classifier_neural_network_learning = MLPClassifier(hidden_layer_sizes=(layer_size,), random_state=9,
    #                                                    max_iter=max_iter,
    #                                                    learning_rate_init=learning_rate_init,
    #                                                    alpha=alpha)
    # _, train_scores, test_scores = learning_curve(classifier_neural_network_learning, X_train, y_train,
    #                                               train_sizes=np.linspace(0.1, 1.0, 10), cv=4,scoring='f1')
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
        classifier_neural_network1 = MLPClassifier(hidden_layer_sizes=(layer_size,), random_state=9,
                                                   max_iter=max_iter,
                                                   alpha=alpha,
                                                   learning_rate_init=learning_rate_init)
        classifier_neural_network1.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, classifier_neural_network1.predict(X_train_train), average="weighted")
        score_validation[epoch] = f1_score(y_train_val, classifier_neural_network1.predict(X_train_val), average="weighted")
        score_test[epoch] = f1_score(y_test, classifier_neural_network1.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, classifier_neural_network1.predict(X_less), average="weighted")
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
    print(np.abs(np.mean(score_training-score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))
    plt.figure()
    if dataset == 'Credit':
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_training, label='Train Score')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_validation, label='Validation Score')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_less, label='Score for Dataset with Less Default')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_test, label='Test Score')
    if dataset == 'Music':
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_training, label='Train Score')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_less, label='Score for Imbalanced Dataset')
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_test, label='Test Score')
    plt.legend()
    plt.title("Learning Curve " + dataset+" (Neural Network)")
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("F1 Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()
    plt.savefig('NN/' + dataset + '_neural_network_learning_curve.png')


def loss_curve_NN(X_train, y_train, X_test, y_test, X_less, y_less, alpha, learning_rate_init, layer_size, Range,
                  dataset):
    print(loss_curve_NN)
    epochs = len(Range)
    loss_training = np.zeros(epochs)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        classifier_neural_network1 = MLPClassifier(hidden_layer_sizes=(layer_size,), random_state=9,
                                                   max_iter=Range[epoch],
                                                   alpha=alpha, learning_rate_init=learning_rate_init)
        classifier_neural_network1.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, classifier_neural_network1.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, classifier_neural_network1.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, classifier_neural_network1.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, classifier_neural_network1.predict(X_less), average="weighted")
        loss_training[epoch] = classifier_neural_network1.loss_

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
    print(np.abs(np.mean(score_training-score_validation)))
    print("1.3")
    print(np.abs(np.mean(score_training - score_test)))
    print("2.3")
    print(np.abs(np.mean(score_training - score_less)))

    plt.figure()
    plt.plot(Range, loss_training, label='Train Loss')
    plt.legend()
    plt.title("Loss Curve (Neural Network) " + dataset)
    plt.xlabel("Maximum Iteration")
    plt.ylabel("Training Loss")
    plt.grid()
    plt.savefig('NN/' + dataset + '_neural_network_loss_curve.png')

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
    plt.title("Score Curve " + dataset+" (Neural Network)")
    plt.xlabel("Maximum Iteration")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('NN/' + dataset + '_neural_network_score_curve.png')

    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Score Curve " + dataset+" (Neural Network)")
        plt.xlabel("Maximum Iteration")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('NN/' + dataset + '_neural_network_score_curve_2.png')


def plot_hidden_layer(X_train, y_train, X_test, y_test, X_less, y_less, alpha, learning_rate_init, max_iter, Range,
                      dataset):
    print(plot_hidden_layer)
    # train_scores, test_scores = validation_curve(classifier_neural_network, X_train, y_train,
    #                                              param_name="hidden_layer_sizes", param_range=np.arange(2, 31, 2), cv=4,scoring='f1')
    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        classifier_neural_network1 = MLPClassifier(hidden_layer_sizes=(Range[epoch],), random_state=9,
                                                   max_iter=max_iter,
                                                   alpha=alpha,
                                                   learning_rate_init=learning_rate_init)
        classifier_neural_network1.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, classifier_neural_network1.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, classifier_neural_network1.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, classifier_neural_network1.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, classifier_neural_network1.predict(X_less), average="weighted")
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
    print(np.abs(np.mean(score_training-score_validation)))
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
    plt.title("Hidden Layer " + dataset+" (Neural Network)")
    plt.xlabel("Hidden Layer")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('NN/' + dataset + '_neural_network_validation_curve_3.png')

    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Hidden Layer " + dataset+" (Neural Network)")
        plt.xlabel("Hidden Layer")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('NN/' + dataset + '_neural_network_validation_curve_3_2.png')
