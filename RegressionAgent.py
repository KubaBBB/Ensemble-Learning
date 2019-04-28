from osbrain import Agent
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import json

classifier_mapper = {'linear' : 'LinearRegressionAgent',
                      'decision' : 'DecisionTreeAgent',
                      'mlp' : 'MLPAgent',
                      'logistic' : 'LogisticRegressionAgent'
                     }

class RegressionAgent(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.metrics = {}
        self.type = None

    def initialize_agent(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def send_full_message(self):
        msg = {}
        msg['mse'] = self.metrics['mse']
        msg['r2_score'] = self.metrics['r2_score'];
        msg['y_predicted'] = self.y_predicted
        msg['name'] = self.name
        self.send('main', msg)

    def calculate(self):
        self.model.fit(self.x_train, self.y_train)
        y_predicted = self.model.predict(self.x_test)
        self.y_predicted = y_predicted
        self.calculate_metrics(y_predicted)

    def calculate_metrics(self, y_predicted):
        self.metrics['mse'] = mean_squared_error(self.y_test, y_predicted)
        self.metrics['r2_score'] = r2_score(self.y_test, y_predicted)