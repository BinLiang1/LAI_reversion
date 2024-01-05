#!/usr/bin/python
# -*- coding : utf-8 -*-

import os
from abc import abstractmethod
from dataclasses import dataclass,field
import tqdm
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from utils.logging_utils import config_logging


DEFAULT_SPLIT_SEED = 10


@dataclass
class Samples:
    x: np.ndarray = field(default_factory=np.zeros((1,0)))
    gt: np.ndarray = field(default_factory=np.zeros((1,0)))

    @property
    def num(self) -> int:
        return len(self.x)

    def split(self, split_index: list):
        if len(split_index) >= self.num:
            raise ValueError("split index too long")
        x_ = [self.x[_] for _ in split_index]
        gt_ = [self.gt[_] for _ in split_index]
        return Samples(np.asarray(x_), np.asarray(gt_))


class Classifier():

    def __init__(self, data_path: str, run_time: int = 1, split_ratio: float = 0.8, **kargs: dict) -> None:
        self.data_path = data_path
        self.split_ratio = split_ratio
        self.run_time = run_time
        self.kargs = kargs

        self.val_metric_lists = ['RMSE', 'R2', 'MAE', "model_train", "model_val"]
        self.search_metric = self.kargs.get("search_metric", "R2")
        if self.search_metric not in self.val_metric_lists:
            raise ValueError(f"non-valid search metric: {self.search_metric}")
        self.search_type = self.kargs.get("search_type", "median")
        self.kfold = self.kargs.get("kfold", 5)  # kfold used when do best param search

        self.model = None

        # 读取数据
        self.ori_sample_data = self.read_data(data_path)

    @staticmethod
    def read_data(data_path: str) -> Samples:

        if not os.path.exists(data_path):
            raise FileExistsError(f"{data_path} not found")


        data = pd.read_csv(data_path)
        df = pd.DataFrame(data, columns=data.columns)
        df.sort_values(by=data.columns[-1], axis=0, inplace=True)
        sample_x = df.iloc[:, :-1].values
        sample_gt = df.iloc[:, -1].values
        return Samples(sample_x, sample_gt)

    @staticmethod
    def random_split_scikit(sample: Samples, ratio: float = 0.8, seed: int = None):

        sample_num = sample.num
        if sample_num <= 10:
            raise ValueError("data too small")
        if ratio < 0.1 or ratio > 0.99:
            raise ValueError("strange split ratio")

        train_x, val_x, train_gt, val_gt = train_test_split(sample.x, sample.gt, train_size=ratio, random_state=seed)
        return Samples(train_x, train_gt), Samples(val_x, val_gt)

    @staticmethod
    def plot_results(data: pd.DataFrame, save_path: str):

        plt.figure(figsize=(12, 6))

        for index in range(1, len(data.columns)):
            x = list(range(len(data)))
            y = list(data.iloc[:, index].T.values)
            if len(x) == 1:
                plt.scatter(x, y)
            else:
                plt.plot(x, y)
        plt.autoscale(enable=True, axis="y")
        title = os.path.splitext(os.path.split(save_path)[1])[0]
        plt.title(title)
        plt.xlabel("iteration")
        plt.ylabel("val")
        plt.xticks(list(range(len(data))))
        plt.legend(data.columns[1:])
        plt.savefig(save_path)

    @staticmethod
    def calculate_score(val_pred, val_gt):

        rmse = np.sqrt(mean_squared_error(val_gt, val_pred))
        r2 = r2_score(val_gt, val_pred)
        mae = mean_absolute_error(val_gt, val_pred)
        return {
            'RMSE': rmse,
            'R2': r2,
            'MAE': mae,
        }

    @staticmethod
    def normalize_data(train_sample: Samples, val_sample: Samples = None, method: str = "StandardScaler"):
        method_dict = {
            "StandardScaler": StandardScaler,
            "Normalizer": Normalizer,
            "MinMaxScaler": MinMaxScaler,
        }

        scaler = method_dict.get(method, StandardScaler)()
        train_sample.x = scaler.fit_transform(train_sample.x)
        if val_sample is not None:
            val_sample.x = scaler.transform(val_sample.x)
        return train_sample, val_sample

    @abstractmethod
    def search_best_params(self, train_sample: Samples, val_sample: Samples):
        """ search best params """
        pass

    def get_scoring_method_scikit(self):
        method_dict = {
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
            "MAE": "neg_mean_absolute_error",
        }
        return method_dict.get(self.search_metric, "neg_mean_squared_error")

    def train_and_test(self, train_sample: Samples, val_sample: Samples):
        """ train model and do test """
        if self.model is None:
            raise ValueError("model not defined")


        train_sample, val_sample = self.normalize_data(train_sample, val_sample)


        self.model.fit(train_sample.x, train_sample.gt)


        y_pred = self.model.predict(val_sample.x)

        # pred score
        pred_score = self.calculate_score(y_pred, val_sample.gt)

        # model score
        model_score = {
            "model_train": self.model.score(train_sample.x, train_sample.gt),
            "model_val": self.model.score(val_sample.x, val_sample.gt),
        }

        score_result = {score: 0.0 for score in self.val_metric_lists}
        score_result.update(pred_score)
        score_result.update(model_score)

        return score_result

    def statistic_multi_run(self, results: pd.DataFrame, save_path: str):
        """ statistic multi """
        statistic_dict = {
            "mean": results.mean().iloc[1::],
            "median": results.median().iloc[1::],
            "std": results.std().iloc[1::],
            "var": results.var().iloc[1::],
            "min": results.min().iloc[1::],
            "max": results.max().iloc[1::],
        }

        statistic_pd: pd.DataFrame = pd.concat([_ for _ in statistic_dict.values()], axis=1)
        statistic_pd.columns = [_ for _ in statistic_dict.keys()]
        statistic_pd.to_csv(save_path)

    def run_search(self, save_dir: str = None):

        train, val = self.random_split_scikit(self.ori_sample_data, self.split_ratio, seed=DEFAULT_SPLIT_SEED)

        best_params, model_score = self.search_best_params(train, val)
        model_params = self.model.get_params()

        if save_dir is None:
            logger = config_logging()
            logger.info(f"best search params: {best_params}")
            logger.info(f"best model total params: {model_params}")
            logger.info(f"best model score: {model_score}")
            return

        os.makedirs(save_dir, exist_ok=True)
        ori_tiltle = os.path.splitext(os.path.split(self.data_path)[1])[0]
        save_title = f"params_{ori_tiltle}_{type(self).__name__}_By_{self.split_ratio:.2f}".lower()

        with open(os.path.join(save_dir, f"{save_title}.json"), "w+") as fp:
            json_str = json.dumps({
                "best_search_params": best_params,
                "best_model_total_params": model_params,
                "best_model_score": model_score
            }, indent=4)
            fp.write(json_str)

    def run(self, save_dir: str = None):
        """ actually run """

        heads = ['Iteration']
        heads.extend(self.val_metric_lists)
        results = pd.DataFrame(columns=heads)
        search_index = results.columns.get_loc(self.search_metric)


        seed_split = None if self.run_time > 1 else DEFAULT_SPLIT_SEED
        pbar = tqdm.tqdm(range(self.run_time), leave=False)
        for iter in pbar:
            pbar.set_description(f"iter: {iter}")

            train, val = self.random_split_scikit(self.ori_sample_data, self.split_ratio, seed=seed_split)

            result = dict()
            result.update({"Iteration": iter + 1})

            model_score = self.train_and_test(train_sample=train, val_sample=val)

            result.update(model_score)

            # to pd frame
            results.loc[len(results)] = result

        statistic_dict = {
            "mean": results.mean().iloc[search_index],
            "median": results.median().iloc[search_index],
            "std": results.std().iloc[search_index],
            "max": results.max().iloc[search_index],
        }
        return_vals = list()
        for type_ in self.search_type:
            return_vals.append(statistic_dict.get(type_))
        return_vals.append(statistic_dict.get("std"))

        if save_dir is None:
            return return_vals

        os.makedirs(save_dir, exist_ok=True)
        ori_tiltle = os.path.splitext(os.path.split(self.data_path)[1])[0]
        save_title = f"{ori_tiltle}_{type(self).__name__}_Runtime{self.run_time}_By_{self.split_ratio:.2f}".lower()


        results.to_csv(os.path.join(save_dir, f"{save_title}.csv"))


        self.plot_results(results, os.path.join(save_dir, f"{save_title}.png"))


        self.statistic_multi_run(results, os.path.join(save_dir, f"{save_title}_statistic.csv"))

        return return_vals
