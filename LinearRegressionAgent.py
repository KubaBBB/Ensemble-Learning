from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np


class LinearRegressionAgent:
    def __init__(self, df, agent):
        self._df = df;
        self._model = LinearRegression();
        self._agent = agent;

    def split_dataframe(self):
        train_set, test_set = train_test_split(self._df, test_size=0.2, random_state=42)
        self._train_set = train_set
        self._test_set = test_set
        self._X_train = train_set.drop(['id', 'date', 'price'], axis=1)
        self._y_train = train_set[['price']]

        self._X_test = test_set.drop(['id', 'date', 'price'], axis=1)
        self._y_test = test_set[['price']]

    def calculate(self):
        self._model.fit(self._X_train, self._y_train)
        y_predicted = self._model.predict(self._X_test)
        mse = mean_squared_error(self._y_test, y_predicted)
        print(f'MSE linear regression: {mse}')

        #self.send_info(f'Mean Square error: {np.sqrt(mse)}')

    def send_info(self, msg):
        self._agent.send('main', msg)

    def define_handler(self):
        addr2 = self._agent.bind('PUSH', alias='main')

        self._agent.connect(addr2, handler=self.log_message)

    def log_message(self, message):
        self._agent.log_info("ELO + ")
        self._y = message;