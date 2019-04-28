import time
import pandas as pd
from osbrain import run_agent, run_nameserver
from models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MasterClassifier import MasterClassifier
from RegressionAgent import RegressionAgent
import DataVisualizer

models = [Model.LINEAR_REGRESSION, Model.DECISION_TREE]


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

    ns = run_nameserver()
    #TODO better name needed ASAP xD
    agents = [run_agent(f'RegressionAgent{i}', base=RegressionAgent) for i in range(len(models))]

    for i in range(len(models)):
        agents[i].initialize_agent(models[i], x_train, y_train, x_test, y_test, str(i))
        agents[i].calculate()

    master_agent = run_agent('DecisionAgent', base=MasterClassifier)
    master_agent.define_addr_conn(agents)

    # Send messages
    for agent in agents:
        time.sleep(1)
        agent.send_full_message()

    time.sleep(3)

    metrics = [master_agent.get_metrics()]
    DataVisualizer.print_metrics(metrics[0])

    master_agent.debug()

    ns.shutdown()

