from osbrain import Agent

class MasterClassifier(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.metrics = dict()
        self.y_predicted = dict()

    def define_addr_conn(self,agents):
        for agent in agents:
            self.connect(agent.addr('main'), handler=self.handle_full_message)

    def handle_full_message(self, msg):
        agent_name = msg['name']
        self.metrics[agent_name] = msg['metrics']
        self.y_predicted[agent_name] = msg['y_predicted']
        self.log_info(f'Message from {agent_name} was handled')

    def get_metrics(self):
        return self.metrics

    def calculate_final_prediction(self):
        return 2.0

    def debug(self):
        name = self.name