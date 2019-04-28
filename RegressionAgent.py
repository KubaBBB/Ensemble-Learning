from osbrain import Agent
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from models import Model


class RegressionAgent(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.metrics = {}
        self.type = None

    def choose_model(self, model_name):
        if model_name == Model.DECISION_TREE:
            return DecisionTreeRegressor()
        elif model_name == Model.LINEAR_REGRESSION:
            return LinearRegression(n_jobs=-1)
        elif model_name == Model.LOGISTIC_REGRESSION:
            return LogisticRegression(n_jobs=-1)
        elif model_name == Model.MLP:
            return MLPRegressor()
        else:
            raise NotImplementedError('Wrong model\'s name provided')

    def initialize_agent(self, model_name, x_train, y_train, x_test, y_test, iterator):
        self.model = self.choose_model(model_name)
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