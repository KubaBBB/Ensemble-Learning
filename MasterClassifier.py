from osbrain import Agent
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, accuracy_score


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

    def calculate_final_prediction(self):
        final_prediction = list()
        preds = list(self.y_predicted.values())
        for p1, p2, p3, p4 in zip(preds[0], preds[1], preds[2], preds[3]):
            final_prediction.append(round((p1+p2+p3+p4)/4))
        self.metrics['EnsembleClassifier'] = r2_score(self.y_true, final_prediction)

    def clean_cache(self):
        self.metrics = dict()
        self.y_predicted = dict()
        self.log_info("Cleaned cache on Master Agent")

    def debug(self):
        name = self.name