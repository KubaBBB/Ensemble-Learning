import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def print_metrics(metrics):
    labels = ['MSE', 'R2 score', 'Median abs error']
    bar_width = 0.30

    for agent in metrics:
        metric_keys = list(metrics[agent].keys())

    for idx in range(len(metric_keys)):
        values = []
        for agent in metrics:
            values.append(metrics[agent][metric_keys[idx]])
        index = np.arange(len(values))
        bars = []
        for i in range(len(values)):
            bar = plt.bar(index[i] + bar_width, values[i], width=bar_width)
            bars += bar;
       # plt.xticks(index + bar_width, labels=metrics.keys())
        for rect in bars:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%.2f' % height, ha='center', va='bottom')
        plt.title(f'Metric: {labels[idx]}')
        plt.xlabel("Classifiers")
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.legend(metrics.keys())
        axes = plt.gca()
        axes.set_ylim([min(values)-0.1*min(values), max(values)+0.02*max(values)])
        plt.tight_layout()
        plt.savefig(f'Metric: {labels[idx]}.png')
        plt.show()
