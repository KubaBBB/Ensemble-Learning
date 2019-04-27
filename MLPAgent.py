from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from osbrain import Agent

class MLPAgent(Agent):
    def on_init(self):
        self.bind('PUSH', alias='main')
        self._mse = 0.0

    def initialize_model(self, df):
        self._df = df;
        self._model = MLPRegressor();

    def send_info(self):
        self.send('main', f'{self.name} MSE:{self._mse}')

    def split_dataframe(self):
        train_set, test_set = train_test_split(self._df, test_size=0.2)
        self._train_set = train_set
        self._test_set = test_set
        self._X_train = train_set.drop(['id', 'date', 'price'], axis=1)
        self._y_train = train_set[['price']]

        self._X_test = test_set.drop(['id', 'date', 'price'], axis=1)
        self._y_test = test_set[['price']]

    def calculate(self):
        self._model.fit(self._X_train, self._y_train)
        y_predicted = self._model.predict(self._X_test)
        self._mse = mean_squared_error(self._y_test, y_predicted)
        return self._mse
