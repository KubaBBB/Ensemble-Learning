from osbrain import run_agent
from osbrain import run_nameserver
import DataVisualizer
from MasterClassifier import MasterClassifier
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from RegressionAgent import RegressionAgent
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import json

num_of_agents = 2;
classifier_mapper = {'linear' : 'LinearRegressionAgent',
                      'decision' : 'DecisionTreeAgent',
                      'mlp' : 'MLPAgent',
                      'logistic' : 'LogisticRegressionAgent'
                     }

models = [
          LinearRegression(n_jobs=-1),
          DecisionTreeRegressor(),
          #LogisticRegression(n_jobs=-1),
          #MLPRegressor()
          ]


def split_dataframe(df):
    train_set, test_set = train_test_split(df, test_size=0.2)
    x_train = train_set.drop(['id', 'date', 'price'], axis=1)
    y_train = train_set[['price']]

    x_test = test_set.drop(['id', 'date', 'price'], axis=1)
    y_test = test_set[['price']]

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    if num_of_agents == len(models):
        # Dataset
        df = pd.read_csv('./housesalesprediction/kc_house_data.csv');
        #DataVisualizer.print_corelation_heatmap(df)

        x_train, y_train, x_test, y_test = split_dataframe(df)

        # System deployment
        ns = run_nameserver()

        # System deployment
        agents = []
        for i in range(num_of_agents):
            linear_agent = run_agent(f'Agent_classifier{i}', base=RegressionAgent)
            agents.append(linear_agent)

        for i in range(num_of_agents):
            agents[i].initialize_agent(models[i], x_train, y_train, x_test, y_test)
            agents[i].calculate()

        classifier = run_agent('Classifier', base=MasterClassifier)
        classifier.define_addr_conn(agents)

        # Send messages
        for agent in agents:
            time.sleep(1)
            agent.send_full_message()

        time.sleep(3)

        metrics = [classifier.get_metrics()]
        DataVisualizer.print_metrics(metrics[0])

        classifier.debug()

        ns.shutdown()
    else:
        print(f'len(models != num_of_agents')
