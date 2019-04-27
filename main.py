from osbrain import run_agent
from osbrain import run_nameserver
import dataInfo
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

    ns.shutdown()
