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
               [Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE],
               [Model.SVR, Model.SVR, Model.SVR, Model.SVR],
               [Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.K_NEIGHBORS],
               [Model.BAYESIAN_RIDGE, Model.BAYESIAN_RIDGE, Model.BAYESIAN_RIDGE, Model.BAYESIAN_RIDGE]
               ]

split_dataset = [False, True, True, True, True]
number_of_agents = 4

def divide_into_train_test(df):
    X = df.drop(['id', 'date', 'price'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    df = pd.read_csv('./housesalesprediction/kc_house_data.csv')
    x_train, y_train, x_test, y_test = divide_into_train_test(df)

    iterator = 0

    ns = run_nameserver()
    agents = [run_agent(f'RegressionAgent{i}', base=RegressionAgent) for i in range(number_of_agents)]

    master_agent = run_agent('DecisionAgent', base=MasterClassifier)
    master_agent.define_addr_conn(agents)

    for models, split in zip(models_list, split_dataset):
        number_of_agents = len(models)
        if split:
            divisions = [0, 0.25, 0.50, 0.75, 1.0]
            n_samples = len(x_train)
            x_train_list = [x_train[floor(divisions[i] * n_samples):floor(divisions[i + 1] * n_samples)] for i in
                            range(number_of_agents)]  # divide x_train for 4 equal parts
            y_train_list = [y_train[floor(divisions[i] * n_samples):floor(divisions[i + 1] * n_samples)] for i in
                            range(number_of_agents)]  # divide y_train for 4 equal parts
        else:
            x_train_list = [x_train for _ in range(number_of_agents)]
            y_train_list = [y_train for _ in range(number_of_agents)]

        for i in range(number_of_agents):
            agents[i].initialize_agent(models[i], x_train_list[i], y_train_list[i], x_test, y_test, str(i))
            agents[i].calculate()

        master_agent.set_true_labels(y_test)

        # Send messages
        for agent in agents:
            agent.send_full_message()
            time.sleep(1)
        time.sleep(3)
        master_agent.calculate_final_prediction()
        metrics = master_agent.get_metrics()
        DataVisualizer.print_metrics(metrics, iterator)

        master_agent.debug()
        iterator += 1
    ns.shutdown()
    #plt.show()
