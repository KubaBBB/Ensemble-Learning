from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class DecisionTreeRegressorAgent:
    def __init__(self, df, agent):
        self._df = df;
        self._tree = DecisionTreeRegressor();
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
        self._tree.fit(self._X_train, self._y_train)
        y_predicted = self._tree.predict(self._X_test)
        mse = mean_squared_error(self._y_test, y_predicted)
        print(f'MSE decision tree: {mse}')
        return mse
        #self.send_info(f'Mean Square error: {np.sqrt(mse)}')

    def send_info(self, msg):
        self._agent.send('main', msg)
