#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy
from typing import Tuple
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .classifer_base import Classifier, Samples
from .builder import CLASSIFIERS


HIDDEN_LAYER_SIZES = (10,)
ACTIVATION = 'relu'  # （'relu'、'tanh'、'logistic'）
LEARNING_RATE = 'adaptive'   # （'constant'、'adaptive'、'invscaling'）
MAX_ITER = 5000


@CLASSIFIERS("BPNN")
class ClassifierBPNN(Classifier):

    def __init__(self, data_path: str, run_time: int = 1, split_ratio: float = 0.8, **kargs: dict) -> None:
        super().__init__(data_path, run_time, split_ratio, **kargs)

        hidden_layer_sizes = kargs.get("hidden_layer_sizes", HIDDEN_LAYER_SIZES)
        activation = kargs.get("activation", ACTIVATION)
        learning_rate = kargs.get("learning_rate", LEARNING_RATE)
        max_iter = kargs.get("max_iter", MAX_ITER)


        bpnn_regressor = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=10,
            solver="adam",
            early_stopping=True,
        )

        self.model = bpnn_regressor

    def search_best_params(self, train_sample: Samples, val_sample: Samples = None) -> Tuple[dict, dict]:
        """ do search """
        cross_fold = max(1, self.kfold)
        model = MLPRegressor(random_state=10, max_iter=MAX_ITER, early_stopping=True,)
                             #hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                             #activation="relu")
        param_grid = {
            'hidden_layer_sizes': [(200,), (200, 10), (10, )],
            "activation": ["relu", 'logistic', 'tanh'],
            'solver': ["adam", "sgd"],
            'learning_rate': ["constant", "adaptive"],
        }

        scoreing_method = self.get_scoring_method_scikit()


        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoreing_method,
                                   cv=cross_fold, n_jobs=-1, verbose=4)


        search_train, search_val = copy.deepcopy(train_sample), copy.deepcopy(val_sample)
        search_train, search_val = self.normalize_data(search_train, search_val, "MinMaxScaler")
        grid_search.fit(search_train.x, search_train.gt)


        self.model = grid_search.best_estimator_


        model_score = self.train_and_test(train_sample, val_sample)

        return grid_search.best_params_, model_score