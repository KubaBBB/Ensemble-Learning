import time
import pandas as pd
from osbrain import run_agent, run_nameserver
from models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MasterClassifier import MasterClassifier
from RegressionAgent import RegressionAgent
import DataVisualizer
from math import floor
import matplotlib.pyplot as plt

models_list = [[Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE],
                [Model.MLP, Model.MLP, Model.MLP, Model.MLP],
                [Model.LINEAR_REGRESSION, Model.LINEAR_REGRESSION, Model.LINEAR_REGRESSION, Model.LINEAR_REGRESSION],
                [Model.LOGISTIC_REGRESSION, Model.LOGISTIC_REGRESSION, Model.LOGISTIC_REGRESSION, Model.LOGISTIC_REGRESSION],
                [Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE],
                [Model.MLP, Model.MLP, Model.MLP, Model.MLP],
               ]

split_data = [False, False, True, True, True, True]

def divide_into_train_test(df):
    X = df.drop(['id', 'date', 'price', 'type'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df['type'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    df = pd.read_csv('./data_with_classes.csv')
    x_train, y_train, x_test, y_test = divide_into_train_test(df)

    x_train_list = list()
    y_train_list = list()
    for models, flag in zip(models_list, split_data):
        if flag:
            n_samples = len(x_train)
            x_train_list.append(x_train[:floor(0.25*n_samples)])
            x_train_list.append(x_train[floor(0.25*n_samples):floor(0.50*n_samples)])
            x_train_list.append(x_train[floor(0.50*n_samples):floor(0.75*n_samples)])
            x_train_list.append(x_train[floor(0.75*n_samples):])
            y_train_list.append(y_train[:floor(0.25*n_samples)])
            y_train_list.append(y_train[floor(0.25 * n_samples):floor(0.50 * n_samples)])
            y_train_list.append(y_train[floor(0.50 * n_samples):floor(0.75 * n_samples)])
            y_train_list.append(y_train[floor(0.75 * n_samples):])
        else:
            x_train_list = [x_train for _ in range(4)]
            y_train_list = [y_train for _ in range(4)]

        ns = run_nameserver()
        agents = [run_agent(f'ClassificationAgent{i}', base=RegressionAgent) for i in range(len(models))]

        for i in range(len(models)):
            agents[i].initialize_agent(models[i], x_train_list[i], y_train_list[i], x_test, y_test, str(i))
            agents[i].calculate()

        master_agent = run_agent('DecisionAgent', base=MasterClassifier)
        master_agent.define_addr_conn(agents)
        master_agent.set_true_labels(y_test)
        # Send messages
        for agent in agents:
            agent.send_full_message()
            time.sleep(1)
        time.sleep(3)
        master_agent.calculate_final_prediction()
        metrics = master_agent.get_metrics()
        DataVisualizer.print_metrics(metrics)

        master_agent.debug()
        ns.shutdown()

    plt.show()
