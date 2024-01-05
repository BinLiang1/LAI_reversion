#!/usr/bin/python
# -*- coding : utf-8 -*-

import argparse
import os
import numpy as np
import copy
from matplotlib import pyplot as plt
import tqdm

from classifer import CLASSIFIERS

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data-path", type=str, help="the data file path",
                        default="./data/1BAN.csv")
    parser.add_argument("-o", "--output-dir", type=str, help="the output path",
                        default="./output")
    parser.add_argument("--split-ratio", type=float,
                        help="split ratio of train and val, if -1, then split from [0.1 - 0.9]",
                        default=0.7)
    parser.add_argument("--run-time", type=int, help="times to run classifier",
                        default=100)
    parser.add_argument("-c", "--classifer", type=str, help="the classifer to use",
                        choices=["RF", "SVR", "BPNN"], default="RF")
    parser.add_argument("--search-metric", type=str, help="the metric index need to search",
                        choices=["R2", "RMSE", "MAE"], default="R2")
    parser.add_argument("--search-type", nargs="+", type=str, help="the metric index need to search",
                        default=["mean", "median", "max"])
    parser.add_argument("--search-best", action="store_true", help="do best param search")
    return parser

def search_best_split_ratio(args):
    split_ratios = np.arange(0.25, 0.9, 0.05)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    metric_of_splits = list()

    search_types = list()
    for type_ in args.search_type:
        if type_ in ["mean", "median", "max"]:
            search_types.append(type_)
    args.search_type = search_types

    pbar = tqdm.tqdm(split_ratios, leave=True)
    for split_ratio in pbar:
        pbar.set_description(f"split_ratio: {split_ratio:.2f}")
        args_ = copy.deepcopy(args)
        args_.split_ratio = split_ratio
        classifer = CLASSIFIERS[args.classifer](**vars(args_))
        metric_of_splits.append(classifer.run(args_.output_dir))


    def plot_figure(x, y, tag, filepath):
        title = os.path.splitext(os.path.split(filepath)[1])[0]

        plt.figure(figsize=(8, 6))
        fig, ax1 = plt.subplots()
        ax1.plot(x, y[0, :], color="blue")
        ax1.set_xlabel("split_ratio")
        ax1.set_ylabel(tag)
        ax2 = ax1.twinx()
        ax2.plot(x, y[1, :], color="green")
        ax2.set_ylabel("std")
        fig.legend([tag, "std"], loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax1.transAxes)

        plt.autoscale(enable=True, axis="both")
        plt.title(title)
        plt.xticks(x)
        plt.savefig(save_fig_path)

    metric_of_splits = np.asarray(metric_of_splits)
    for idx, type_ in enumerate(search_types):
        save_fig_path = os.path.join(output_dir,
                                    f"{args.search_metric}_of_split_ratio_of_{args.classifer}_{type_}.jpg").lower()
        y = np.asarray([metric_of_splits[:, idx], metric_of_splits[:, -1]])
        plot_figure(split_ratios, y, type_, save_fig_path)


def main():
    args = arg_parser().parse_args()
    if args.split_ratio == -1.0:
        search_best_split_ratio(args)
    else:
        classifer = CLASSIFIERS[args.classifer](**vars(args))
        if args.search_best:
            classifer.run_search(args.output_dir)
        else:
            classifer.run(args.output_dir)


if __name__ == "__main__":
    main()