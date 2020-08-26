import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
import csv
import statistics

def print_corelation_heatmap(df):
    df.corr()['price'].sort_values(ascending=False)
    correlation = df.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
    plt.show()

def print_dataset_info(df):
    print(df.head(n=10))
    df.info()
    np.round(df.describe())
    df.isnull().any()


def print_metrics(metrics, id, split):
    ind = np.arange(len(metrics))
    width = 0.35
    vals = np.array(list(metrics.values()))*100
    minimum = min(vals)
    with open(f'results\exp2_increasing\lab1_{id}.csv', 'w', newline='\n') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f, delimiter=',')
        for item in metrics:
            w.writerow([item, metrics[item]])
    s = 0;
    for i in range(len(metrics.values()) - 1):
        s += vals[i]
    standard_dev = statistics.stdev(vals[:len(vals)-1])
    print(f'{list(metrics.keys())[0]} - Avg r2 score: {s/(len(vals)-1)}')
    print(f'With std:{standard_dev}')
    plt.figure()
    plt.bar(ind, vals-minimum, width, color=['green', 'blue', 'yellow', 'cyan', 'red'])
    for i, v in zip(ind, vals-minimum):
        plt.text(i, v, round(vals[i], 2), horizontalalignment='center', verticalalignment='bottom')

    plt.ylabel('Difference in %')
    plt.title(f'avg:{s/(len(vals)-1)}, std:{standard_dev}')
    plt.xticks(ind, (metrics.keys()), rotation=15)
    plt.tight_layout()
    plt.savefig(f'figures/metric_{split.name}_{id}.png')
