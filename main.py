from osbrain import run_agent
from osbrain import run_nameserver
import dataInfo
from Ąnsą_learning import Ąnsą_learning
import time
import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier
from LinearRegressionClassifier import LinearRegressionClassifier

def log_message(agent, message):
    agent.log_info('Received: %s' % message)

if __name__ == '__main__':
    # Dataset
    df = pd.read_csv('./housesalesprediction/kc_house_data.csv');
    #dataInfo.print_corelation_heatmap(df)

    # System deployment
    ns = run_nameserver()

    # System deployment
    linear_agents = []
    for i in range(4):
        linear_agent = run_agent(f'Linear_classifier{i}', base=LinearRegressionClassifier)
        linear_agents.append(linear_agent)

    for agent in linear_agents:
        agent.initialize_model(df)
        agent.split_dataframe()
        agent.calculate()

    decision_agent = run_agent('Decision_classifier', base=DecisionTreeClassifier)
    decision_agent.initialize_model(df)
    decision_agent.split_dataframe()
    decision_mse = decision_agent.calculate()

    classifier = run_agent('Classifier', base=Ąnsą_learning)
    classifier.define_addr_conn(linear_agents)

    classifier.connect(decision_agent.addr('main'), handler=log_message)

    # Send messages
    for agent in linear_agents:
        time.sleep(1)
        agent.send_info()
        #decision_agent.send_info()

    ns.shutdown()
