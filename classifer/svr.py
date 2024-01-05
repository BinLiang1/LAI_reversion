import copy
import math
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from .classifer_base import Classifier, Samples
from .builder import CLASSIFIERS


SVR_C = 1.0
SVR_KERNEL = "rbf"
SVR_EPSION = 0.2718005081632653
SVR_GAMMA = "auto"


@CLASSIFIERS("SVR")
class ClassifierSVR(Classifier):

    def __init__(self, data_path: str, run_time: int = 1, split_ratio: float = 0.8, **kargs: dict) -> None:
        super().__init__(data_path, run_time, split_ratio, kargs=kargs)

        svr_c = kargs.get("C", SVR_C)
        svr_kernel = kargs.get("kernel", SVR_KERNEL)
        svr_epsion = kargs.get("epsion", SVR_EPSION)
        svr_gamma = kargs.get("gamma", SVR_GAMMA)


        svr_regressor = SVR(kernel=svr_kernel,
                            gamma=svr_gamma,
                            epsilon=svr_epsion,
                            C=svr_c,)

        self.model = svr_regressor

    def search_best_params(self, train_sample: Samples, val_sample: Samples = None) -> Tuple[dict, dict]:
        """ do search """
        cross_fold = max(1, self.kfold)
        model = SVR(epsilon=SVR_EPSION)
        param_grid = {
            'kernel': ["linear", "poly", "rbf", "sigmoid"],  # kernel
            'C': np.linspace(0.0001, 100, 1000),
            'gamma': ['scale', 'auto']  #  "poly", "rbf", "sigmoid"
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
