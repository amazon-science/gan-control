# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import matplotlib.pyplot as plt


def plot_error_bar(xs, ys, xlabel='layer #', ylabel='abs mean', legends=None, save_path=None, title=None, mean_abs=False):
    markers = ['^', 'o', 's', '+', 'x', '>', '<', '*']
    for i, (x, y, marker) in enumerate(zip(xs, ys, markers)):
        x_means = [np.array(xi).mean() for xi in x]
        x_std = [np.array(xi).std() if len(xi) > 1 else 0 for xi in x]

        if mean_abs:
            y_means = [np.abs(np.array(yi)).mean() for yi in y]
        else:
            y_means = [np.array(yi).mean() for yi in y]
        y_std = [np.array(yi).std() if len(yi) > 1 else 0 for yi in y]

        plt.errorbar(x_means, y_means, xerr=x_std, yerr=y_std, linestyle='None', marker='^')


    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if legends is not None:
        plt.legend(legends)
    if title is not None:
        plt.title(title)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)