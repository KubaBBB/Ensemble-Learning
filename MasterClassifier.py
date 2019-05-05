from osbrain import Agent
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, accuracy_score
from EnumStorage import Ensemble


weight = [5, 2, 2, 1]

class MasterClassifier(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.metrics = dict()
        self.y_predicted = dict()

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
        final_prediction = list()
        if average == Ensemble.ARITHMETIC:
            preds = list(self.y_predicted.values())
            for p1, p2, p3, p4 in zip(preds[0], preds[1], preds[2], preds[3]):
                final_prediction.append(round((p1 + p2 + p3 + p4) / 4))
        elif average == Ensemble.WEIGHTED:
            sorted_agent = self.map_weight()
            preds = [self.y_predicted[agent] for agent in sorted_agent ]
            for p1, p2, p3, p4 in zip(preds[0], preds[1], preds[2], preds[3]):
                final_prediction.append(
                    round((weight[0] * p1 +
                           weight[1] * p2 +
                           weight[2] * p3 +
                           weight[3] * p4)
                           / sum(weight)))
        else:
            raise NotImplementedError('Wrong average\'s name provided')

        self.metrics['EnsembleClassifier'] = r2_score(self.y_true, final_prediction)

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

