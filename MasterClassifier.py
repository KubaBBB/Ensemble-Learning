from osbrain import Agent
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, accuracy_score
from EnumStorage import Ensemble


weight = [5, 3, 2, 1]

class MasterClassifier(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.metrics = dict()
        self.y_predicted = dict()
        self.calculated = dict()

    def define_addr_conn(self, agents):
        for agent in agents:
            self.connect(agent.addr('main'), handler=self.handle_full_message)

    def set_true_labels(self, y_true):
        self.y_true = y_true

    def handle_full_message(self, msg):
        agent_name = msg['name']
        self.metrics[agent_name] = msg['metrics']
        self.y_predicted[agent_name] = msg['y_predicted']
        self.log_info(f'Message from {agent_name} was handled')

    def get_metrics(self):
        return self.metrics

    def calculate_final_prediction(self, average):
        final_prediction1 = list()
        preds1 = list(self.y_predicted.values())
        for p1, p2, p3, p4 in zip(preds1[0], preds1[1], preds1[2], preds1[3]):
            final_prediction1.append(round((p1 + p2 + p3 + p4) / 4))
        self.calculated['ec_avg'] = r2_score(self.y_true, final_prediction1)

        sorted_agent = self.map_weight()
        final_prediction2 = list()
        final_prediction3 = list()

        preds2 = [self.y_predicted[agent] for agent in sorted_agent ]
        for p1, p2, p3, p4 in zip(preds2[0], preds2[1], preds2[2], preds2[3]):
            final_prediction2.append(
                    round((weight[0] * p1 +
                           weight[1] * p2 +
                           weight[2] * p3 +
                           weight[3] * p4)
                           / sum(weight)))
            final_prediction3.append(
                    round((weight[0] * p4 +
                           weight[1] * p3 +
                           weight[2] * p2 +
                           weight[3] * p1)
                           / sum(weight)))

        self.calculated['ec_weigtht_strong'] = r2_score(self.y_true, final_prediction2)
        self.calculated['ec_weigtht_weak'] = r2_score(self.y_true, final_prediction3)

        self.metrics['EnsembleClassifier'] = r2_score(self.y_true, final_prediction1)

        print(self.calculated)

    def clean_cache(self):
        self.metrics = dict()
        self.y_predicted = dict()
        self.log_info("Cleaned cache on Master Agent")

    def map_weight(self):
        metrics = sorted([value for key, value in self.metrics.items()], reverse = True)
        sorted_agents = [find_key_by_value(self.metrics, metric) for metric in metrics]
        return sorted_agents

    def debug(self):
        name = self.name

def find_key_by_value(dictionary, value_to_map):
    for key, value in dictionary.items():
        if value == value_to_map:
            return key

