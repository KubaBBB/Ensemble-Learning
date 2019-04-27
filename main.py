from osbrain import run_agent
from osbrain import run_nameserver
import dataInfo
import time
import pandas as pd
from DecisionTreeRegressorAgent import DecisionTreeRegressorAgent
from LinearRegressionAgent import LinearRegressionAgent

def log_message(agent, message):
    agent.log_info('Received: %s' % message)

if __name__ == '__main__':
    # Dataset
    df = pd.read_csv('./housesalesprediction/kc_house_data.csv');
    dataInfo.print_corelation_heatmap(df)

    # System deployment
    ns = run_nameserver()
    lrAgent = run_agent('lrAgent')
    dtrAgent = run_agent('dtrAgent')

    # System configuration
    #addr = dtrAgent.bind('PUSH', alias='main')
    #dtrAgent.connect(addr, handler=log_message)

    #Encaplsulate agents
    dtr = DecisionTreeRegressorAgent(df, agent=dtrAgent);
    dtr.split_dataframe()
    dtr_mse = dtr.calculate()

    lr = LinearRegressionAgent(df, agent=lrAgent);
    lr.split_dataframe()
    #lr.define_handler
    lr_mse = lr.calculate()

    # System deployment
    first_class = run_agent('First_classifier')
    second_class = run_agent('Second_classifier')
    classifier = run_agent('Classifier')

    # System configuration
    addr1 = first_class.bind('PUSH', alias='main1')
    addr2 = second_class.bind('PUSH', alias='main2')

    classifier.connect(addr1, handler=log_message)
    classifier.connect(addr2, handler=log_message)

    #classifier.connect(addr1, handler=log_message)

    # Send messages
    for _ in range(1):
        time.sleep(1)
        first_class.send('main1', f'LR MSE:{lr_mse}')
        second_class.send('main2', f'DTR MSE:{dtr_mse}')

    ns.shutdown()
