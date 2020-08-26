import time
import pandas as pd
from osbrain import run_agent, run_nameserver
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from MasterClassifier import MasterClassifier
from MasterClassifier import Ensemble
from RegressionAgent import RegressionAgent
import DataVisualizer
from math import floor
import matplotlib.pyplot as plt
from EnumStorage import LabelMapper
from EnumStorage import SplitDataset
from EnumStorage import Model


average = [Ensemble.ARITHMETIC]
# models_list = [[Model.K_NEIGHBORS, Model.BAYESIAN_RIDGE, Model.DECISION_TREE, Model.K_NEIGHBORS],
#                [Model.DECISION_TREE, Model.K_NEIGHBORS, Model.SVR, Model.DECISION_TREE],
#                [Model.BAYESIAN_RIDGE, Model.SVR, Model.BAYESIAN_RIDGE, Model.SVR],
#                [Model.SVR, Model.BAYESIAN_RIDGE, Model.DECISION_TREE, Model.K_NEIGHBORS],
#                [Model.K_NEIGHBORS, Model.DECISION_TREE, Model.DECISION_TREE, Model.K_NEIGHBORS],
#                ]


# models_list = [[Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE],
#                [Model.DECISION_TREE, Model.DECISION_TREE, Model.K_NEIGHBORS, Model.K_NEIGHBORS],
#                [Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.DECISION_TREE, Model.DECISION_TREE],
#                [Model.DECISION_TREE, Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.DECISION_TREE],
#                [Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.K_NEIGHBORS],
#                ]

models_list = [[Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE, Model.DECISION_TREE],
                [Model.BAYESIAN_RIDGE, Model.BAYESIAN_RIDGE, Model.BAYESIAN_RIDGE, Model.BAYESIAN_RIDGE],
                [Model.SVR, Model.SVR, Model.SVR, Model.SVR],
                [Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.K_NEIGHBORS, Model.K_NEIGHBORS],
                ]

labels_group = [
    [LabelMapper.bedrooms, LabelMapper.bathrooms, LabelMapper.sqft_living, LabelMapper.sqft_lot],
    [LabelMapper.sqft_basement, LabelMapper.sqft_above, LabelMapper.sqft_living15, LabelMapper.sqft_lot15],
    [LabelMapper.yr_built, LabelMapper.yr_renovated, LabelMapper.condition, LabelMapper.grade],
    [LabelMapper.zipcode, LabelMapper.long, LabelMapper.lat, LabelMapper.waterfront]
]

split_dataset = [
    SplitDataset.NONE,
    SplitDataset.BAGGING,
    SplitDataset.BAGGING,
    SplitDataset.BAGGING,
    #SplitDataset.AGENT,
    ]

number_of_agents = 4


def divide_into_train_test(df):
    X = df.drop(['id', 'date', 'price'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    return X_train, y_train, X_test, y_test

def divide_train_sets_for_agents(x_train_set, x_test_set):
    x_train_list = []
    x_test_list = []

    for i in range(number_of_agents):
        groups = [label.value[0] for label in labels_group[i]]
        groups.sort();

        extracted_train_data = x_train_set[:, groups]
        x_train_list.append(extracted_train_data);

        extracted_test_data = x_test_set[:, groups]
        x_test_list.append(extracted_test_data);

    return x_train_list, x_test_list

if __name__ == '__main__':
    df = pd.read_csv('./housesalesprediction/kc_house_data.csv')
    x_train, y_train, x_test, y_test = divide_into_train_test(df)

    iterator = 0

    #DataVisualizer.print_corelation_heatmap(df);

    ns = run_nameserver()
    agents = [run_agent(f'RegressionAgent{i}', base=RegressionAgent) for i in range(number_of_agents)]

    master_agent = run_agent('DecisionAgent', base=MasterClassifier)
    master_agent.define_addr_conn(agents)
    for avg in average:
        for models, split in zip(models_list, split_dataset):
            number_of_agents = len(models)
            if split == SplitDataset.BAGGING:
                divisions = [0, 0.25, 0.50, 0.75, 1.0]
                n_samples = len(x_train)
                x_train_list = [x_train[floor(divisions[i] * n_samples):floor(divisions[i + 1] * n_samples)] for i in
                                range(number_of_agents)]  # divide x_train for 4 equal parts
                y_train_list = [y_train[floor(divisions[i] * n_samples):floor(divisions[i + 1] * n_samples)] for i in
                                range(number_of_agents)]  # divide y_train for 4 equal parts
                x_test_list = [x_test for _ in range(number_of_agents)]
            elif split == SplitDataset.NONE:
                x_train_list = [x_train for _ in range(number_of_agents)]
                y_train_list = [y_train for _ in range(number_of_agents)]
                x_test_list = [x_test for _ in range(number_of_agents)]
            elif split == SplitDataset.AGENT:
                x_train_list, x_test_list = divide_train_sets_for_agents(x_train, x_test)
                y_train_list = [y_train for _ in range(number_of_agents)]

            for i in range(number_of_agents):
                agents[i].initialize_agent(models[i], x_train_list[i], y_train_list[i], x_test_list[i], y_test, str(i))
                agents[i].calculate()

            master_agent.set_true_labels(y_test)

            # Send messages
            for agent in agents:
                agent.send_full_message()
                time.sleep(2)
            time.sleep(3)
            master_agent.calculate_final_prediction(avg)
            metrics = master_agent.get_metrics()
            DataVisualizer.print_metrics(metrics, f'{avg.name}_{iterator}', split)

            master_agent.clean_cache()
            master_agent.debug()
            iterator += 1
            time.sleep(4)
    ns.shutdown()
    plt.show()
