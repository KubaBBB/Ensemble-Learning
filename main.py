from osbrain import run_agent
from osbrain import run_nameserver
import dataInfo
import time
import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier
from LinearRegressionClassifier import LinearRegressionClassifier

def log_message(agent, message):
    agent.log_info('Received: %s' % message)

if __name__ == '__main__':
    # Dataset
    df = pd.read_csv('./housesalesprediction/kc_house_data.csv');
    dataInfo.print_corelation_heatmap(df)

    # System deployment
    ns = run_nameserver()

    # System deployment
    linear_agent = run_agent('Linear_classifier', base=LinearRegressionClassifier)
    linear_agent.initialize_model(df)
    linear_agent.split_dataframe()
    linear_mse = linear_agent.calculate()

    decision_agent = run_agent('Decision_classifier', base=DecisionTreeClassifier)
    decision_agent.initialize_model(df)
    decision_agent.split_dataframe()
    decision_mse = decision_agent.calculate()

    classifier = run_agent('Classifier')

    classifier.connect(linear_agent.addr('main'), handler=log_message)
    classifier.connect(decision_agent.addr('main'), handler=log_message)

    # Send messages
    for _ in range(1):
        time.sleep(1)
        linear_agent.send_info()
        decision_agent.send_info()

    ns.shutdown()
