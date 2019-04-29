from osbrain import Agent
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from models import Model






class RegressionAgent(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self.type = None

    def choose_model(self, model_name):
        if model_name == Model.DECISION_TREE:
            return DecisionTreeClassifier()
        elif model_name == Model.SVR:
            return SVR(kernel='linear', C=1e2, degree=5)
        elif model_name == Model.BAYESIAN_RIDGE:
            return BayesianRidge()
        elif model_name == Model.K_NEIGHBORS:
            return KNeighborsRegressor(n_jobs=-1)
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
        msg['metrics'] = self.r2
        msg['y_predicted'] = self.y_predicted
        msg['name'] = self.type
        self.send('main', msg)

    def calculate(self):
        self.model.fit(self.x_train, self.y_train)
        self.y_predicted = self.model.predict(self.x_test)
        self.calculate_metrics(self.y_predicted)

    def calculate_metrics(self, y_predicted):
        self.r2 = r2_score(self.y_test, y_predicted)
