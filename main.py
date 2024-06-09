from NN import plot_alpha
from NN import plot_learning_rate_NN
from NN import best_params_NN_ML
from NN import learning_curve_NN
from NN import loss_curve_NN
from NN import plot_hidden_layer
from sklearn.neural_network import MLPClassifier
from KNN import plot_n_neighbors
from KNN import plot_power
from KNN import best_params_KNN
from KNN import learning_curve_knn
from KNN import plot_weight
from KNN import plot_algo
from SVM import plot_c
from SVM import plot_kernel_type
from SVM import best_params_SVM
from SVM import learning_curve_svm
from SVM import loss_curve_SVM

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Processing import data_processing_credit
from Processing import data_processing_music
from Processing import plot_counts
from sklearn import svm
from Comparison import time_train_comparison
from Comparison import time_infer_comparison
from Comparison import accuracy_comparison
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time

import warnings

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    data = pd.read_csv('credit_risk_dataset.csv')

    data_processed, default_case, data_less_default = data_processing_credit(data, .6)
    default_case_add = default_case.drop(default_case.sample(frac=.45, random_state=1).index)

    # data_processed=data_processed.drop(data_processed.sample(frac=.9, random_state=1).index)
    # default_case_add = default_case_add.drop(default_case_add.sample(frac=.9, random_state=1).index)
    # print(data_processed['loan_status'].value_counts())

    y = data_processed.loan_status
    y = y.values
    y_less = data_less_default.loan_status
    y_less = y_less.values
    print("Number of samples:", y.size)
    print("Percentage of default:", y[y == 1].size / y.size * 100, "%")
    # # split into train and test
    X = data_processed.drop(['loan_status'], axis=1)
    X = preprocessing.scale(X)
    X_less = data_less_default.drop(['loan_status'], axis=1)
    X_less = preprocessing.scale(X_less)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

    X_test = np.append(default_case_add.drop(['loan_status'], axis=1).values, X_test, axis=0)
    y_test = np.append(default_case_add.loan_status, y_test)
    print('percentage:', np.unique(y_test, return_counts=True))
    a = pd.DataFrame(y_test)
    a.columns = ['loan_status']
    plot_counts(a, "loan_status")
    #


    # # Neural Network
    # classifier_accuracy = np.zeros(3)
    # classifier_accuracy_2 = np.zeros(3)
    # time_train = np.zeros(3)
    # time_infer = np.zeros(3)
    # classifier_neural_network = MLPClassifier(hidden_layer_sizes=(4, 4), random_state=9)
    # X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
    #                                                                           random_state=9)
    # time_train[0], time_infer[0], classifier_accuracy[0], classifier_accuracy_2[0], learning_rate_init, alpha,layer_size,max_iter = best_params_NN_ML(X_train_train, y_train_train,y_train_val,X_train_val,X_test,y_test,
    #                                                                                                     classifier_neural_network)
    # np.save("time_train.npy",time_train)
    # np.save("time_infer.npy",time_infer)
    # np.save("classifier_accuracy.npy",classifier_accuracy)
    # np.save("classifier_accuracy_2.npy",classifier_accuracy_2)

    # learning_rate_init, alpha,layer_size,max_iter = [0.01,1.6e-06,15,220]
    # plot_alpha(X_train, y_train, X_test, y_test, X_less, y_less,learning_rate_init,max_iter,layer_size,np.logspace(-7,0,20),'Credit')
    # plot_learning_rate_NN(X_train, y_train, X_test, y_test, X_less, y_less,alpha,max_iter,layer_size,np.logspace(-5,0,10),'Credit')
    # plot_hidden_layer(X_train, y_train, X_test, y_test, X_less, y_less,alpha, learning_rate_init,max_iter,np.arange(2,31,2),'Credit')
    # learning_curve_NN(X_train, y_train, X_test, y_test, X_less, y_less,learning_rate_init, alpha,layer_size,max_iter,'Credit')
    # loss_curve_NN(X_train, y_train, X_test, y_test, X_less, y_less,alpha, learning_rate_init,layer_size,np.arange(5, 51 * 5, 5),'Credit')


    # # # KNN
    # time_train = np.load("time_train.npy")
    # time_infer = np.load("time_infer.npy")
    # classifier_accuracy = np.load("classifier_accuracy.npy")
    # classifier_accuracy_2 = np.load("classifier_accuracy_2.npy")
    # classifier_knn = KNeighborsClassifier()
    # X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
    #                                                                           random_state=9)
    # time_train[1], time_infer[1], classifier_accuracy[1],classifier_accuracy_2[1], n_neighbors, p, weights, algorithm = best_params_KNN(X_train_train, y_train_train,y_train_val,X_train_val,X_test,y_test,
    #                                                                                                            classifier_knn)
    # np.save("time_train.npy",time_train)
    # np.save("time_infer.npy",time_infer)
    # np.save("classifier_accuracy.npy",classifier_accuracy)
    # np.save("classifier_accuracy_2.npy",classifier_accuracy_2)


    # neighbors, p,weights,algorithm = [8, 1, 'distance', 'auto']
    # plot_n_neighbors(X_train, y_train, X_test, y_test, X_less, y_less, p, weights, algorithm, np.arange(1, 50, 2),
    #                  'Credit')
    # plot_power(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, weights, algorithm, [1, 2, 3], 'Credit')
    # learning_curve_knn(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, weights, algorithm, 'Credit')
    # plot_weight(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, algorithm, ['distance', 'uniform'],
    #             'Credit')
    # plot_algo(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, weights,
    #           ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #           'Credit')

    # # # SVM
    # time_train = np.load("time_train.npy")
    # time_infer = np.load("time_infer.npy")
    # classifier_accuracy = np.load("classifier_accuracy.npy")
    # classifier_accuracy_2 = np.load("classifier_accuracy_2.npy")
    # SVM = svm.SVC(random_state=9)
    # X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(X_train, y_train, test_size=0.3,
    #                                                                           random_state=9)
    # time_train[2], time_infer[2], classifier_accuracy[2],classifier_accuracy_2[2], C, kernel = best_params_SVM(X_train_train, y_train_train,y_train_val,X_train_val,X_test,y_test, SVM)
    # np.save("time_train.npy",time_train)
    # np.save("time_infer.npy",time_infer)
    # np.save("classifier_accuracy.npy",classifier_accuracy)
    # np.save("classifier_accuracy_2.npy",classifier_accuracy_2)

    C, kernel =[2.15,'rbf']
    # plot_c(X_train, y_train, X_test, y_test, X_less, y_less, kernel, np.logspace(-3, 3, 10), 'Credit')
    # plot_kernel_type(X_train, y_train, X_test, y_test, X_less, y_less, C, ['linear', 'rbf', 'poly', 'sigmoid'], 'Credit')
    # learning_curve_svm(X_train, y_train, X_test, y_test, X_less, y_less, C, kernel, 'Credit')
    loss_curve_SVM(X_train, y_train, X_test, y_test, X_less, y_less, C, kernel, np.arange(1,200,10),'Credit')





###################################################################################################################################################################################################
    # Dataset 2 Music Genre
    data = pd.read_csv('music_genre.csv')


    data_processed, data_less = data_processing_music(data)
    plot_counts(data_less, "music_genre")
    # data_processed = data_processed.drop(data_processed.sample(frac=.9, random_state=1).index)
    # data_less = data_less.drop(data_less.sample(frac=.9, random_state=1).index)
    # print(data_processed['music_genre'].value_counts())
    # print(data_less['music_genre'].value_counts())

    y = data_processed.music_genre
    y = y.values
    y_less = data_less.music_genre
    y_less = y_less.values
    X = data_processed.drop(['music_genre'], axis=1)
    X = X.apply(pd.to_numeric)
    X = preprocessing.scale(X)
    X_less = data_less.drop(['music_genre'], axis=1)
    X_less = X_less.apply(pd.to_numeric)
    X_less = preprocessing.scale(X_less)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
    plot_counts(data_processed, "music_genre")




    # # Neural Network
    # classifier_accuracy = np.zeros(3)
    # classifier_accuracy_2 = np.zeros(3)
    # time_train = np.zeros(3)
    # time_infer = np.zeros(3)
    # classifier_neural_network = MLPClassifier(hidden_layer_sizes=(4, 4), random_state=9, max_iter=200)
    # time_train[0], time_infer[0], classifier_accuracy[0], classifier_accuracy_2[0],learning_rate_init, alpha,layer_size,max_iter = best_params_NN_ML(X_train,
    #                                                                                                     y_train, y_test,
    #                                                                                                     X_test,X_less,y_less,
    #                                                                                                     classifier_neural_network)
    # np.save("time_train2.npy",time_train)
    # np.save("time_infer2.npy",time_infer)
    # np.save("classifier_accuracy2.npy",classifier_accuracy)
    # np.save("classifier_accuracy2_2.npy", classifier_accuracy_2)


    # learning_rate_init, alpha, layer_size, max_iter = [0.02154, 0.000129, 22, 240]
    # plot_alpha(X_train, y_train, X_test, y_test, X_less, y_less,learning_rate_init,max_iter,layer_size,np.logspace(-7,0,20),'Music')
    # plot_learning_rate_NN(X_train, y_train, X_test, y_test, X_less, y_less,alpha,max_iter,layer_size,np.logspace(-5,0,10),'Music')
    # plot_hidden_layer(X_train, y_train, X_test, y_test, X_less, y_less,alpha, learning_rate_init,max_iter,np.arange(2,31,2),'Music')
    # learning_curve_NN(X_train, y_train, X_test, y_test, X_less, y_less,learning_rate_init, alpha,layer_size,max_iter,'Music')
    # loss_curve_NN(X_train, y_train, X_test, y_test, X_less, y_less,alpha, learning_rate_init,layer_size,np.arange(5, 51 * 5, 5),'Music')


    # # # KNN
    # time_train = np.load("time_train2.npy")
    # time_infer = np.load("time_infer2.npy")
    # classifier_accuracy = np.load("classifier_accuracy2.npy")
    # classifier_accuracy_2 = np.load("classifier_accuracy2_2.npy")
    # classifier_knn = KNeighborsClassifier()
    # time_train[1], time_infer[1], classifier_accuracy[1], classifier_accuracy_2[1],n_neighbors, p, weights, algorithm = best_params_KNN(X_train,
    #                                                                                                            y_train,
    #                                                                                                            y_test,
    #                                                                                                            X_test,X_less,y_less,
    #                                                                                                            classifier_knn)
    # np.save("time_train2.npy",time_train)
    # np.save("time_infer2.npy",time_infer)
    # np.save("classifier_accuracy2.npy",classifier_accuracy)
    # np.save("classifier_accuracy2_2.npy", classifier_accuracy_2)


    # neighbors, p,weights,algorithm = [46, 1, 'uniform', 'auto']
    # plot_n_neighbors(X_train, y_train, X_test, y_test, X_less, y_less,p,weights,algorithm, np.arange(1, 50,5), 'Music')
    # plot_power(X_train, y_train, X_test, y_test, X_less, y_less,neighbors,weights,algorithm, [1,2,3], 'Music')
    # learning_curve_knn(X_train, y_train, X_test, y_test, X_less, y_less,neighbors, p,weights,algorithm, 'Music')
    # plot_weight(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, algorithm, ['distance', 'uniform'], 'Music')
    # plot_algo(X_train, y_train, X_test, y_test, X_less, y_less, neighbors, p, weights, ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #             'Music')

    # # # SVM
    # time_train = np.load("time_train2.npy")
    # time_infer = np.load("time_infer2.npy")
    # classifier_accuracy = np.load("classifier_accuracy2.npy")
    # classifier_accuracy_2 = np.load("classifier_accuracy2_2.npy")
    # SVM = svm.SVC(random_state=9)
    # time_train[2], time_infer[2], classifier_accuracy[2], classifier_accuracy_2[2],C, kernel = best_params_SVM(X_train, y_train,y_test,X_test,X_less,y_less, SVM)
    # np.save("time_train.npy2",time_train)
    # np.save("time_infer.npy2",time_infer)
    # np.save("classifier_accuracy2.npy",classifier_accuracy)
    # np.save("classifier_accuracy2_2.npy",classifier_accuracy_2)

    # C, kernel =[2.15,'rbf']
    loss_curve_SVM(X_train, y_train, X_test, y_test, X_less, y_less, C, kernel, np.arange(1,200,10), 'Music')
    # plot_c(X_train, y_train, X_test, y_test, X_less, y_less, kernel, np.logspace(-3, 3, 10), 'Music')
    # plot_kernel_type(X_train, y_train, X_test, y_test, X_less, y_less, C, ['linear', 'rbf', 'poly', 'sigmoid'], 'Music')
    # learning_curve_svm(X_train, y_train, X_test, y_test, X_less, y_less, C, kernel, 'Music')

    # comparison
    time_train = np.load("time_train.npy")[0:3]
    time_infer = np.load("time_infer.npy")[0:3]
    classifier_accuracy = np.load("classifier_accuracy.npy")[0:3]
    classifier_accuracy_2 = np.load("classifier_accuracy_2.npy")[0:3]
    time_train2 = np.load("time_train2.npy")[0:3]
    time_infer2 = np.load("time_infer2.npy")[0:3]
    classifier_accuracy2 = np.load("classifier_accuracy2.npy")[0:3]
    classifier_accuracy2_2 = np.load("classifier_accuracy2_2.npy")[0:3]
    time_train2[2], time_infer2[2]=[487.86176,0.48399]
    time_train_comparison(time_train,'Credit')
    time_train_comparison(time_train2, 'Music')
    time_infer_comparison(time_infer,'Credit')
    time_infer_comparison(time_infer2, 'Music')
    accuracy_comparison(classifier_accuracy,classifier_accuracy_2, 'Credit',"1")
    accuracy_comparison(classifier_accuracy2, classifier_accuracy2_2,'Music',"2")



