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
    labels = ['MSE', 'R2 score']
    index = np.arange(len(metrics))
    bar_width = 0.30
    bar_height = 0.50
    counter = 0

    for metric in metrics:
        values = [float(item) for item in metric.values()]
        bars = []
        for i in range(len(labels)):
            bar = plt.bar(index[i] + bar_width, values[i], width=bar_width)
            bars += bar;
        plt.xticks(index + bar_width, labels=metric.keys())
        for rect in bars:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, '%.2f' % height, ha='center', va='bottom')
        plt.title(f'Metric: {labels[counter]}')
        plt.xlabel("Classifiers")
        axes = plt.gca()
        axes.set_ylim([min(values)-0.1*min(values), max(values)+0.02*max(values)])
        plt.tight_layout()
        plt.savefig(f'Metric: {labels[counter]}.png')
        plt.show()
        counter+=1

