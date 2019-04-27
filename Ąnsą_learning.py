from osbrain import Agent
from matplotlib import pyplot as plt
import json

def log_message(agent, message):
    agent.log_info('Received: %s' % message)

class Ąnsą_learning(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self._mse = dict()
        self._y_predicted = dict()
        self._r2_score = dict()

    def send_info(self):
        self.send('main', f'{self.name} MSE:{self._mse}')

    def define_addr_conn(self,agents):
        for agent in agents:
            #self.connect(agent.addr('main'), handler=log_message)
            self.connect(agent.addr('main'), handler=self.handle_full_message)

    def handle_full_message(self, msg):
        self._r2_score[msg['name']] = msg['r2_score']
        self._mse[msg['name']] = msg['mse']
        self._y_predicted[msg['name']] = msg['y_predicted']

    def get_metrics(self):
        return self._mse, self._r2_score

    def debug(self):
        name = self.name