#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Nick Knowles (knowlen@wwu.edu)
    May 2017
    
    A callable script for most modern machine learning models. Used for rapid baseline testing. 
    Specify a model, train/dev data paths, and then the hyperparameters you wish to set. 
    
    
    Currently supports: 
        Classification:
          -K Nearest Neighbor (KNN; Classify based on the Euclidean distance of neighboring datapoints)
          -Support Vector Machine (SVM; project the data onto a kernel function in high dimensional space, 
                                   try to separate with linear max-margin hyperplane.)
          -Random Forest (collection of decision trees, with bagged features & dataset bootstrapping).
          -Boosting (a forest sequentially built using info from gradients of the past trees). 

"""

import argparse
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

def return_parser():
    parser = argparse.ArgumentParser("Baseline models.")
    parser.add_argument('model', type=str, help='[random_forest, knn, svm, boost].')
    
    # data files
    parser.add_argument('train_feat', type=str, help='Features/inputs for the development set')
    parser.add_argument('train_target', type=str, help='Targets/labels for training set.')
    parser.add_argument('dev_feat', type=str, help='Features/inputs for the development/test set.')
    parser.add_argument('dev_target', type=str, help='Targets/labels for development/test set.')
 
    parser.add_argument('-mode', type=str, default='R', help='Targets/labels for development/test set.')
 

 

    """  
                         ** Support Vector Machine **
    
    Notes: look into vs one vs rest. http://scikit-learn.org/stable/modules/svm.html
    sklearn.svm.SVC
    SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, 
        tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
        decision_function_shape=None, random_state=None)

    """
    parser.add_argument("-C", type=float, default=1.0, help="Penalty param of the error term.")
    parser.add_argument("-kernel", type=str, default="rbf", help="[linear, poly, rbf, sigmoid].")
    parser.add_argument("-degree", type=int, default=3, help="Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.")
    parser.add_argument("-gamma", type=str, default="auto", help="kernel coefficient (if not linear).")
    parser.add_argument("-decision_shape", type=str, default="ovr", help="[ovo, ovr] one vs one, one vs rest.")



    """
                         ** Boosting Trees **

    sklearn.ensemble.GradientBoostingClassifier:
    GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, 
			       criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, 
                               init=None, random_state=None, max_features=None, verbose=0, 
                               max_leaf_nodes=None, warm_start=False, presort='auto')
    """
    parser.add_argument("-learn_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("-num_trees", type=int, default=100, help="Number of trees to boost.")
    parser.add_argument("-max_depth", type=int, default=None, help="Maximum tree depth.")
    parser.add_argument("-subsample", type=float, default=1.0, help="Number of samples to use for fitting each tree.")
    parser.add_argument("-num_feats", type=int, default=-1, help="Number of features to bag.")




    """
                         ** KNN **

    sklearn.neighbors.KNeighborsClassifier:
    KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                         metric='minkowski', metric_params=None, n_jobs=1, **kwargs)
    """    
    parser.add_argument('-learnrate', type=float, default=0.001,help='Step size for gradient descent.')
    parser.add_argument("-k", type=int, default=5, help="number of neighbores used.")
    parser.add_argument('-weights', type=str, default='distance', help='[distance, uniform] Weight function used in predictions.')
    parser.add_argument('-alg', type=str, default='auto', help='[ball_tree, kd_tree, brute]')
    parser.add_argument('-p', type=int, default=2, help='Power param for Minkowski metric, 2 = euclidian distance.')
    # n_jobs also included 


    """
                         ** Random Forest **

    sklearn.ensemble.RandomForestClassifier:
    RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, 
			   min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
			   max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, 
                           oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, 
                           class_weight=None) 
    """
    parser.add_argument('-criterion', type=str, default="entropy", help='[gini, entropy] Use purity or information gain when calculating splits.')
    parser.add_argument('-n_jobs', type=int, default=-1, help='Number of jobs to dispatch for paralell training.')
    #parser.add_argument('-num_feats', type=int, default=2, help='Number of features to bag.')
    #parser.add_argument('-max_depth', type=int, default=8, help='Maximum depth for the tree.')
    #parser.add_argument('-num_trees', type=int, default=10, help= 'Number of trees to use.') 
   

    return parser


def get_model(model, mode='R', n_trees=1, criterion='entropy', n_feats='auto', depth=8, n_jobs=-1, C=1.0, kernel='rbf', 
              deg=3, gamma='auto', dec_shape='ovr', learn_rate=0.001, subsample=1.0, n_neighbors=5, 
              weights='distance', alg='auto', p=2):
    # TREES
    if model == 'random_forest':
        if mode == 'R':
            m = RandomForestRegressor(n_estimators=n_trees, criterion='mse', max_depth=depth,
                                      max_features=n_feats, n_jobs=n_jobs, verbose=1)
        else:
            m = RandomForestClassifier(n_estimators=n_trees, criterion=criterion, max_depth=depth,
                                       max_features=n_feats, n_jobs=n_jobs, verbose=1)
    elif model == 'boost':
        if mode == 'R':
            m = GradientBoostingRegressor(learning_rate=learn_rate, n_estimators=n_trees, subsample=subsample, 
                                           max_depth=depth, max_features=n_feats) 
        else:    
            m = GradientBoostingClassifier(learning_rate=learn_rate, n_estimators=n_trees, subsample=subsample, 
                                           max_depth=depth, max_features=n_feats) 
    
    elif model == 'extra':
        if mode == 'R':
            m = ExtraTreesRegressor(n_estimators=n_trees, criterion='mse', max_depth=depth, 
                                    max_features=n_feats, bootstrap=True, n_jobs=n_jobs)
        else:
            m = ExtraTreesClassifier(n_estimators=n_trees, criterion=criterion, max_depth=depth, 
                                    max_features=n_feats, bootstrap=True, n_jobs=n_jobs)
        
    elif model == 'knn':
        m = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=alg, p=p, n_jobs=n_jobs)
    elif model == 'svm':
        m = SVC()
    return m



if __name__ == '__main__':
    args = return_parser().parse_args()     
    # data 
    train_feats =  np.loadtxt(args.train_feat)
    train_targets =  np.loadtxt(args.train_target)
    dev_feats =  np.loadtxt(args.dev_feat)
    dev_targets =  np.loadtxt(args.dev_target)
    
    # if input vector is scalar, reshape to matrix. 
    #if len(train_feats.shape) < 2:
    #       train_feats = train_feats.reshape([-1, 1])
    #       dev_feats = dev_feats.reshape([-1,1])

    args = return_parser().parse_args()
    if args.num_feats == -1:
        num_feats = 'auto'
    else:
        num_feats = args.num_feats


    mod = get_model(args.model, mode=args.mode, n_trees=args.num_trees, criterion=args.criterion, n_feats=num_feats, depth=args.max_depth, n_jobs=-1, C=args.C, kernel=args.kernel,
                    deg=args.degree, gamma=args.gamma, dec_shape=args.decision_shape, learn_rate=args.learn_rate, subsample=args.subsample, n_neighbors=args.k,
                    weights=args.weights, alg=args.alg, p=args.p)
    mod.fit(train_feats, train_targets)
    print mod
    
    if args.mode == 'R':
        # for regression, give Loss 
        train_preds =  np.rint(mod.predict(train_feats))
        dev_preds =  np.rint(mod.predict(dev_feats))
        train_score = mean_squared_error(train_targets, train_preds)
        dev_score = mean_squared_error(dev_targets, dev_preds)

    else:    
        # for classification, just calculate accuracy
        train_score =  mod.score(y=train_targets, X=train_feats)
        dev_score =  mod.score(y=dev_targets, X=dev_feats)
#       train_preds =  np.rint(mod.predict(train_feats))
#       dev_preds =  np.rint(mod.predict(dev_feats))
#       train_score = mean_squared_error(train_targets, train_preds)
#       dev_score = mean_squared_error(dev_targets, dev_preds)



    print "train: ", train_score, "  dev: ", dev_score

