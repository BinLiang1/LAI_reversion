#!/usr/bin/python
# -*- coding : utf-8 -*-

import copy
from typing import Tuple
import numpy as np
import math

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from .classifer_base import Classifier, Samples
from .builder import CLASSIFIERS



N_ESTIMATORS = 100
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1


@CLASSIFIERS("RF")
class ClassifierRF(Classifier):

    def __init__(self, data_path: str, run_time: int = 1, split_ratio: float = 0.8, **kargs: dict) -> None:
        super().__init__(data_path, run_time, split_ratio, **kargs)
        n_estimators = kargs.get("n_estimators", N_ESTIMATORS)
        max_depth = kargs.get("max_depth", MAX_DEPTH)
        min_samples_split = kargs.get("min_samples_split", MIN_SAMPLES_SPLIT)
        min_samples_leaf = kargs.get("min_samples_leaf", MIN_SAMPLES_LEAF)


        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=10,
        )

        self.model = rf_regressor

    def search_best_params(self, train_sample: Samples, val_sample: Samples = None) -> Tuple[dict, dict]:
        """ do search """
        cross_fold = max(1, self.kfold)
        max_sample = int(math.floor(len(train_sample.gt) * (1.0 - 1.0 / cross_fold)))
        model = RandomForestRegressor(random_state=10)
        param_grid = {
            'n_estimators': range(5, 500, 5),
            'min_samples_split': range(2, max_sample, 1),
            'min_samples_leaf': range(1, max_sample, 1)
        }

        scoreing_method = self.get_scoring_method_scikit()


        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoreing_method,
                                   cv=cross_fold, n_jobs=-1, verbose=4)


        search_train, search_val = copy.deepcopy(train_sample), copy.deepcopy(val_sample)
        search_train, search_val = self.normalize_data(search_train, search_val)
        grid_search.fit(search_train.x, search_train.gt)


        self.model = grid_search.best_estimator_


        model_score = self.train_and_test(train_sample, val_sample)

        return grid_search.best_params_, model_score