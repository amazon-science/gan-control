# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

v1_dark_blue = '#012875'
v1_dark_cyan = '#2BAEB3'
v1_dark_orange = '#F9B72A'
v1_dark_pink = '#FFCFCB'
v1_dark_red = '#FF6E3C'

v1_light_blue = '#425BDD'
v1_light_cyan = '#6FD9DD'
v1_light_orange = '#F9DB73'
v1_light_pink = '#FBE2DF'
v1_light_red = '#F8905C'

COLOR_PALETTE_DARK_V1 = [v1_dark_blue, v1_dark_cyan, v1_dark_orange, v1_dark_pink, v1_dark_red, '#F06E3C', '#EF003C', '#4F6E3C', '#206E3C', '#1F003C', '#0F6E3C']
COLOR_PALETTE_LIGHT_V1 = [v1_light_blue, v1_light_cyan, v1_light_orange, v1_light_pink, v1_light_red, '#F06E3C', '#EF003C', '#4F6E3C', '#206E3C', '#1F003C', '#0F6E3C']
TITLE_FONT = {"family" : "sans", "size" : 16, "color" : "black", "weight" : "roman"}


def plot_graph(arrays, x_arrays=None, labels=None, title= None, xlabel=None, ylabel=None, use_spine=True,
               annotate_min_and_last=False, use_colors=True, save_path=None, figsize=(6, 4), ncol=None):
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    color = None
    for i, array in enumerate(arrays):
        label = None if labels is None else labels[i]
        if use_colors:
            color = COLOR_PALETTE_DARK_V1[i]
        if x_arrays is None:
            x_array = range(len(array))
        else:
            x_array = x_arrays[i]
        plt.plot(x_array, array, label=label, color=color)
    if annotate_min_and_last:
        for i, array in enumerate(arrays):
            if x_arrays is None:
                x_array = range(len(array))
            else:
                x_array = x_arrays[i]
            min_val = np.array(array).min()
            min_arg = np.array(array).argmin()
            last_val = array[-1]
            label = "{:.2f}".format(min_val)
            plt.annotate(label, (x_array[min_arg], min_val), textcoords="offset points", xytext=(0, 10), ha='center')
            label = "{:.2f}".format(last_val)
            plt.annotate(label, (x_array[-1], last_val), textcoords="offset points", xytext=(0, 10), ha='center')
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontdict=TITLE_FONT)
    ax.grid(False)
    if labels is not None:
        ncol = len(arrays) if ncol is None else ncol
        if title is None:
            plt.legend(bbox_to_anchor=(0, 1.01, 1, 0.2), loc="lower left", mode='expand', ncol=ncol)
        else:
            plt.legend(loc="upper right", ncol=ncol)
    if not use_spine:
        sns.despine(left=True, bottom=True)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig='all')


def plot_hist(arrays, labels=None, title= None, xlabel=None, ylabel=None, bins=100, plt_range=None, alpha=0.5,
              figsize=(6, 4), show_kde=True, show_percentiles=True, percentiles=(0.1,0.2,0.5,0.8,0.9), use_colors=True,
              use_spine=True, ncol=1, save_path=None, min_lim=None, max_lim=None):
    plt.clf()
    if plt_range is None:
        all_arr = np.concatenate(arrays)
        if min_lim is None:
            min_lim = - 1000000.
        if max_lim is None:
            max_lim = 1000000.
        plt_range = (max(min_lim, all_arr.min() - 0.1), min(all_arr.max() + 0.1, max_lim))
    fig, ax = plt.subplots(figsize=figsize)
    color = None
    color_kde = None
    percentiles_color = None

    # plot hist
    for i, array in enumerate(arrays):
        label = None if labels is None else labels[i]
        if use_colors:
            color = COLOR_PALETTE_LIGHT_V1[i]
        series = pd.Series(array)
        series.plot(kind="hist", density=True, bins=bins, range=plt_range, alpha=alpha, histtype="stepfilled",
                    label=label, color=color)
    if show_kde:
        for i, array in enumerate(arrays):
            if use_colors:
                color_kde = COLOR_PALETTE_DARK_V1[i]
            series = pd.Series(array)
            series.plot(kind="kde", color=color_kde, label='_nolegend_')
    if show_percentiles:
        for i, array in enumerate(arrays):
            if use_colors:
                percentiles_color = COLOR_PALETTE_DARK_V1[i]
            series = pd.Series(array)
            # Calculate percentiles
            quants = []
            for j, percentile in enumerate(percentiles):
                # [quantile, opacity, length]
                quants.append([series.quantile(percentile), 0.85, percentile])
                ax.axvline(quants[-1][0], alpha=quants[-1][1], ymax=quants[-1][2], linestyle=":",
                           color=percentiles_color)
                ax.text(quants[-1][0], (quants[-1][2] + 0.01) * ax.get_ylim()[1], '%.3f' % (quants[-1][0]), size=10,
                        alpha=0.99, color=percentiles_color)
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontdict=TITLE_FONT)
    ax.set_xlim(plt_range[0], plt_range[1])
    ax.grid(False)
    if labels is not None:
        if title is None:
            plt.legend(bbox_to_anchor=(0, 1.01, 1, 0.2), loc="lower left", mode='expand', ncol=ncol)
        else:
            plt.legend(loc="upper right", ncol=ncol)
    if not use_spine:
        sns.despine(left=True, bottom=True)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig='all')


def plot_bar(arrays, xticklabels, labels=None, title=None, xlabel=None, ylabel=None, use_text=True,
             figsize=(6, 4), use_colors=True, use_spine=True, save_path=None, not_hist=False):
    plt.clf()
    label = None
    fig, ax = plt.subplots(figsize=figsize)
    num_of_labels = len(xticklabels) # len(np.unique(arrays[0]))
    num_of_arrays = len(arrays)
    x = np.arange(num_of_labels)
    width = 0.9 #/ num_of_arrays
    rects_list = []
    counts = []
    for array_num, array in enumerate(arrays):
        if use_colors:
            color = COLOR_PALETTE_LIGHT_V1[array_num]
        if labels is not None:
            label = labels[array_num]
        if not_hist:
            bars = array
        else:
            bars = np.histogram(array, bins=np.arange(-0.5, num_of_labels, 1))[0]
        if num_of_arrays > 1:
            if use_colors:
                rects_list.append(ax.bar(x - width / 2 + array_num * (width / num_of_arrays), bars, width/ num_of_arrays, label=label, color=color))
            else:
                rects_list.append(ax.bar(x - width / 2 + array_num * (width / num_of_arrays), bars, width/ num_of_arrays, label=label))
        else:
            rects_list.append(ax.bar(x, bars, width, label=label, color=color))
        counts.append(len(array))

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_ylabel(xlabel)
    if title is not None:
        ax.set_title(title)
    if xticklabels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
    if labels is not None:
        ax.legend()
    if use_text:
        for rects, count in zip(rects_list, counts):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('%.1f%s' % (100 * height / count, '%'),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
    if not use_spine:
        sns.despine(left=True, bottom=True)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig='all')

