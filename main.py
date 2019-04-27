from osbrain import run_agent
from osbrain import run_nameserver
import DataVisualizer
from Ąnsą_learning import Ąnsą_learning
import time
import pandas as pd
from DecisionTreeAgent import DecisionTreeAgent
from LinearRegressionAgent import LinearRegressionAgent
from MLPAgent import MLPAgent
from LogisticRegressionAgent import LogisticRegressionAgent

def log_message(agent, message):
    agent.log_info('Received: %s' % message)

if __name__ == '__main__':
    # Dataset
    df = pd.read_csv('./housesalesprediction/kc_house_data.csv');
    #DataVisualizer.print_corelation_heatmap(df)

    # System deployment
    ns = run_nameserver()

    # System deployment
    linear_agents = []
    for i in range(2):
        linear_agent = run_agent(f'Linear_classifier{i}', base=LinearRegressionAgent)
        linear_agents.append(linear_agent)

    for agent in linear_agents:
        agent.initialize_model(df)
        agent.split_dataframe()
        agent.calculate()

    decision_agent = run_agent('Decision_classifier', base=DecisionTreeAgent)
    decision_agent.initialize_model(df)
    decision_agent.split_dataframe()
    #decision_agent.calculate()

    mlp_agent = run_agent('MLP_classifier', base=MLPAgent)
    mlp_agent.initialize_model(df)
    mlp_agent.split_dataframe()
    #mlp_agent.calculate()

    logistic_agent = run_agent('Logisitic_regression_classifier', base=LogisticRegressionAgent)
    logistic_agent.initialize_model(df)
    logistic_agent.split_dataframe()
    #logistic_agent.calculate()

    classifier = run_agent('Classifier', base=Ąnsą_learning)
    classifier.define_addr_conn(linear_agents)

    classifier.connect(decision_agent.addr('main'), handler=log_message)
    classifier.connect(mlp_agent.addr('main'), handler=log_message)
    classifier.connect(logistic_agent.addr('main'), handler=log_message)

    # Send messages
    for agent in linear_agents:
        time.sleep(1)
        agent.send_full_message()
        #agent.send_info()

    #logistic_agent.send_info()
    #mlp_agent.send_info()
    #decision_agent.send_info()

    metrics = [classifier.get_metrics()]
    DataVisualizer.print_metrics(metrics[0])

    classifier.debug()

    ns.shutdown()
