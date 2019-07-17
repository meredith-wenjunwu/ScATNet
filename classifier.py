#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:46:39 2019

@author: wuwenjun
"""

from sklearn import linear_model
import pickle
import os
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support

def model_init(args):
    if args.mode == 'svm': loss = 'hinge'
    else: loss = 'log'

    clf=linear_model.SGDClassifier(learning_rate = args.learning_rate, loss=loss, eta0=args.lr)
    return clf

def model_load(path):
    clf = pickle.load(open(path, 'rb'))
    assert clf is not None, "Model load failed"
    return clf

def model_update(clf, X, Y):
    assert clf is not None, "Model is None"
    clf.partial_fit(X, Y)
    return clf

def model_save(clf, path):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))
    pickle.dump(clf, path)

def model_predict(clf, X):
    return clf.predict(X)

def model_prob(clf, X):
    return clf.predict_proba(X)

def model_report(clf, Y, Y_t, train=False):
    if train:
        mode = 'Train '
    else:
        mode = 'Test '
    accuracy = accuracy_score(Y_t, Y)
    metrics = precision_recall_fscore_support(Y_t, Y)
    print(mode + 'Precision =  {:.2222}'.(metrics[0][0]))
    print(mode + 'Recall =  {:.2222}'.(metrics[1][0]))
    print(mode + 'F1 score =  {:.2222}'.(metrics[2][0]))
    print(mode + 'Accuracy =  {:.2222}'.(accuracy))

