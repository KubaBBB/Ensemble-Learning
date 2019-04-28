from osbrain import Agent

def log_message(agent, message):
    agent.log_info('Received: %s' % message)

class MasterClassifier(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self._mse = dict()
        self._y_predicted = dict()
        self._r2_score = dict()

    def define_addr_conn(self,agents):
        for agent in agents:
            self.connect(agent.addr('main'), handler=self.handle_full_message)

    def handle_full_message(self, msg):
        agent_name = msg['name']
        self._r2_score[agent_name] = msg['r2_score']
        self._mse[agent_name] = msg['mse']
        self._y_predicted[agent_name] = msg['y_predicted']
        self.log_info(f'Message from {agent_name} was handled')

    def get_metrics(self):
        return self._mse, self._r2_score

    def debug(self):
        name = self.name