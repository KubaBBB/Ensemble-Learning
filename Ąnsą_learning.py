from osbrain import Agent

def log_message(agent, message):
    agent.log_info('Received: %s' % message)

class Ąnsą_learning(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self._mse = 0.0

    def send_info(self):
        self.send('main', f'{self.name} MSE:{self._mse}')

    def define_addr_conn(self,agents):
        for agent in agents:
            self.connect(agent.addr('main'), handler=log_message)
