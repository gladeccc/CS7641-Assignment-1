import matplotlib.pyplot as plt
import numpy as np
def time_train_comparison(time_train,dataset):
    Range=['Neural Network',"KNN","SVM"]
    plt.figure()
    plt.bar(Range, time_train, label='train time')
    plt.legend()
    plt.title("Training Time with Model, "+dataset)
    plt.xlabel("Model Type")
    plt.ylabel("Time")
    plt.grid()
    plt.savefig('time_train_comparison'+dataset+'.png')

def time_infer_comparison(time_infer,dataset):
    Range=['Neural Network',"KNN","SVM"]
    plt.figure()
    plt.bar(Range, time_infer, label='infer time')
    plt.legend()
    plt.title("Inferring Time with Model, "+dataset)
    plt.xlabel("Model Type")
    plt.ylabel("Time")
    plt.grid()
    plt.savefig('time_infer_comparison '+dataset+'.png')

def accuracy_comparison(classifier_accuracy,classifier_accuracy2,dataset,num):
    Range=['Neural Network',"KNN","SVM"]
    plt.figure()
    plt.bar(Range, classifier_accuracy, label="Same Distribution")
    plt.bar(Range, classifier_accuracy2, label="Different Distribution")
    plt.legend(loc=4)
    plt.title("F1 Score with Model, "+dataset)
    plt.xlabel("Model Type")
    plt.ylabel("F1 Score")
    plt.grid()
    plt.savefig('accuracy_comparison '+dataset+num+'.png')