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

def plot_c(X_train, y_train, X_test, y_test, X_less, y_less, kernel, Range, dataset):
    print(plot_c)
    # train_scores, test_scores = validation_curve(classifier_svm, X_train, y_train, param_name="C",
    #                                              param_range=np.logspace(-3, 3, 10), cv=4)

    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        print(epoch)
        SVM = svm.SVC(random_state=9,C=Range[epoch],kernel=kernel)
        SVM.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, SVM.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, SVM.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, SVM.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, SVM.predict(X_less), average="weighted")
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
        plt.semilogx(Range, score_training, label='Train Score')
        plt.semilogx(Range, score_validation, label='Validation Score')
        plt.semilogx(Range, score_less, label='Score for Dataset with Less Default')
    if dataset == 'Music':
        plt.semilogx(Range, score_training, label='Train Score')
        plt.semilogx(Range, score_test, label='Test Score')
        plt.semilogx(Range, score_less, label='Score for Imbalanced Dataset')
    plt.legend()
    plt.title("C " + dataset + " (SVM)")
    plt.xlabel("C")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('SVM/' + dataset + '_SVM_validation_curve_1.png')
    if dataset == 'Credit':
        plt.figure()
        plt.semilogx(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("C " + dataset + " (SVM)")
        plt.xlabel("C")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('SVM/' + dataset + '_SVM_validation_curve_1_2.png')


def plot_kernel_type(X_train, y_train, X_test, y_test, X_less, y_less, C, Range, dataset):
    print(plot_kernel_type)
    epochs = len(Range)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        print(epoch)
        SVM = svm.SVC(random_state=9, kernel=Range[epoch], C=C)
        SVM.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, SVM.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, SVM.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, SVM.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, SVM.predict(X_less), average="weighted")
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
    plt.title("Kernel " + dataset + " (SVM)")
    plt.xlabel("kernel")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('SVM/' + dataset + '_SVM_validation_curve_2.png')
    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Kernel " + dataset + " (SVM)")
        plt.xlabel("kernel")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('SVM/' + dataset + '_SVM_validation_curve_2_2.png')


def best_params_SVM(X_train, y_train,y_test,X_test,  X_less, y_less,classifier_svm):
    print(best_params_SVM)
    # param_grid = {'C': [2.15], 'kernel': [ 'rbf']}
    param_grid = {'C': np.logspace(-3, 3, 10), 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    classifier_svm_best = GridSearchCV(classifier_svm, param_grid=param_grid, cv=4)

    start_time = time.time()
    classifier_svm_best.fit(X_train, y_train)
    end_time = time.time()
    time_train = end_time - start_time
    print("Best params for SVM:", classifier_svm_best.best_params_)

    start_time = time.time()
    classifier_accuracy = f1_score(y_test, classifier_svm_best.predict(X_test),average='weighted')
    end_time = time.time()
    time_infer = end_time - start_time
    print("Accuracy for best SVM:", classifier_accuracy)
    classifier_accuracy2 = f1_score(y_less, classifier_svm_best.predict(X_less), average='weighted')
    return time_train, time_infer, classifier_accuracy, classifier_accuracy2,\
           classifier_svm_best.best_params_['C'], classifier_svm_best.best_params_['kernel']


def learning_curve_svm(X_train, y_train, X_test, y_test, X_less, y_less, C, kernel, dataset):
    print(learning_curve_svm)
    SVM = svm.SVC(random_state=9, kernel=kernel, C=C)

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
        SVM.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, SVM.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, SVM.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, SVM.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, SVM.predict(X_less), average="weighted")
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
    plt.title("Learning Curve " + dataset + " (SVM)")
    plt.xlabel("Percentage of Training Examples")
    plt.ylabel("F1 Score")
    plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
    plt.grid()
    plt.savefig('SVM/' + dataset + '_SVM_learning_curve.png')

    if dataset == 'Credit':
        plt.figure()
        plt.plot(np.linspace(0.99, 0.1, 10) * 100, score_test, label='Test Score')
        plt.legend()
        plt.title("Learning Curve " + dataset + " (SVM)")
        plt.xlabel("Percentage of Training Examples")
        plt.ylabel("F1 Score")
        plt.xticks(np.linspace(0.1, 1.0, 10) * 100)
        plt.grid()
        plt.savefig('SVM/' + dataset + '_SVM_learning_curve_2.png')

def loss_curve_SVM(X_train, y_train, X_test, y_test, X_less, y_less, C, kernel, Range, dataset):
    print(loss_curve_SVM)
    epochs = len(Range)
    loss_training = np.zeros(epochs)
    score_training = np.zeros(epochs)
    score_test = np.zeros(epochs)
    score_validation = np.zeros(epochs)
    score_less = np.zeros(epochs)
    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
                                                                              random_state=9)

    for epoch in range(epochs):
        print(epoch)
        SVM = svm.SVC(random_state=9, kernel=kernel, C=C,max_iter=Range[epoch])
        SVM.fit(X_train_train, y_train_train)
        score_training[epoch] = f1_score(y_train_train, SVM.predict(X_train_train),
                                         average="weighted")
        score_validation[epoch] = f1_score(y_train_val, SVM.predict(X_train_val),
                                           average="weighted")
        score_test[epoch] = f1_score(y_test, SVM.predict(X_test), average="weighted")
        score_less[epoch] = f1_score(y_less, SVM.predict(X_less), average="weighted")

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
    plt.title("Score Curve " + dataset+" (SVM)")
    plt.xlabel("Maximum Iteration")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('SVM/' + dataset + '_SVM_score_curve.png')

    if dataset == 'Credit':
        plt.figure()
        plt.plot(Range, score_test, label='Test Score')
        plt.legend()
        plt.title("Score Curve " + dataset+" (SVM)")
        plt.xlabel("Maximum Iteration")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig('SVM/' + dataset + '_SVM_score_curve_2.png')