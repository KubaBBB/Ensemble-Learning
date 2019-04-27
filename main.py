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
    dtr.calculate()

    lr = LinearRegressionAgent(df, agent=lrAgent);
    lr.split_dataframe()
    #lr.define_handler
    lr.calculate()


    # System deployment
    alice = run_agent('Alice')
    bob = run_agent('Bob')

    # System configuration
    addr = alice.bind('PUSH', alias='main')
    bob.connect(addr, handler=log_message)

    # Send messages
    for _ in range(3):
        time.sleep(1)
        alice.send('main', 'Hello, Bob!')

    ns.shutdown()
