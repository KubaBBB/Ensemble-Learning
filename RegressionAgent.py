from osbrain import Agent
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import json

regressor_types = ['LinearRegression', 'DecisionTreeRegressor', 'LogisticRegression']


class RegressionAgent(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.metrics = {}
        self.type = None

    def initialize_agent(self, model, x_train, y_train, x_test, y_test, iterator):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.type = type(self.model).__name__ + str(iterator)

    def send_full_message(self):
        msg = {}
        msg['metrics'] = self.metrics
        msg['y_predicted'] = self.y_predicted
        msg['name'] = self.type
        self.send('main', msg)

    def calculate(self):
        if self.type is 'LinearRegression':
            l = 8
        elif self.type is 'DecisionTreeRegressor':
            o = 9.0
        elif self.type is 'LogisticRegression':
            self.y_train = self.y_train.values.ravel()


        self.model.fit(self.x_train, self.y_train)
        y_predicted = self.model.predict(self.x_test)
        self.y_predicted = y_predicted
        self.calculate_metrics(y_predicted)

    def calculate_metrics(self, y_predicted):
        self.metrics['mse'] = mean_squared_error(self.y_test, y_predicted)
        self.metrics['r2_score'] = r2_score(self.y_test, y_predicted)
        self.metrics['median'] = median_absolute_error(self.y_test, y_predicted)