# imports
import pandas

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

# vars
selected_features_path =\
    'MFDCA-DATA/FraudedFeatureSelectedOutputs/output0.csv'

# Magic


def classify(features_number=220, method=1, nu=1/10, gamma=0.0001):
    global number_of_features
    number_of_features = features_number
    methods = {1: one_class_svm}
    return methods[method](nu, gamma)


def one_class_svm(n, g):
    data_set = pandas.read_csv(selected_features_path)
    data_set.pop(data_set.columns[0])

    # class distribution
    print(data_set.groupby('Class').size())

    # Split-out validation dataset
    array = data_set.values
    X = array[:, 0:number_of_features]
    Y = array[:, number_of_features]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = \
        X[0:50], X[50:], Y[0:50], Y[50:]
    #    model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('OneClassSVM', OneClassSVM()))
    # evaluate each model in turn
    results = []
    names = []
    #for name, model in models:
    #    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    #    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    #    results.append(cv_results)
    #    names.append(name)
    #    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    #    print(msg)

    model = OneClassSVM(nu=n, kernel='rbf', gamma=g)
    model.fit(X_train)
#    print('\n')
#    print(model)
#    print('\n')

    preds = model.predict(X_validation)
    correct_preds = []
    for pred in preds:
        if pred == -1:
            correct_preds.append(1)
        else:
            correct_preds.append(0)
    targs = Y_validation
    print('\n')

    correct_targs = []
    for targ in targs:
        correct_targs.append(targ)
#    print(correct_targs)
#    print(correct_preds)

    print("accuracy: ", metrics.accuracy_score(correct_targs, correct_preds))
#    print("precision: ", metrics.precision_score(correct_targs, correct_preds, average=None))
#    print("recall: ", metrics.recall_score(correct_targs, correct_preds, average=None))
#    print("f1: ", metrics.f1_score(correct_targs, correct_preds, average=None))
    # print("area under curve (auc): ", metrics.roc_auc_score(correct_targs, preds))

    res = metrics.accuracy_score(correct_targs, correct_preds)
#    print(type(np.float64(res).item()))
    fres = np.float64(res).item()
    return fres
