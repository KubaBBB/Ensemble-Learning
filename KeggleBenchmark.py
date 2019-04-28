import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np


class KG:
    def __init__(self, df, agent):
        self.siema = "elo"
        self._df = df
        self._agent = agent


    def calc(self):
        train_data, test_data = train_test_split(self._df, train_size=0.8, random_state=3)

        lr = LinearRegression()
        X_train = np.array(train_data['sqft_living'], dtype=pd.Series).reshape(-1, 1)
        y_train = np.array(train_data['price'], dtype=pd.Series)
        lr.fit(X_train, y_train)

        X_test = np.array(test_data['sqft_living'], dtype=pd.Series).reshape(-1, 1)
        y_test = np.array(test_data['price'], dtype=pd.Series)

        pred = lr.predict(X_test)
        msesm = float(format(np.sqrt(mean_squared_error(y_test, pred)), '.3f'))
        rtrsm = float(format(lr.score(X_train, y_train), '.3f'))
        rtesm = float(format(lr.score(X_test, y_test), '.3f'))
        cv = float(format(cross_val_score(lr, self._df[['sqft_living']], self._df['price'], cv=5).mean(), '.3f'))
        print(np.sqrt(mean_squared_error(y_test, pred)))
        print("Average Price for Test Data: {:.3f}".format(y_test.mean()))
        print('Intercept: {}'.format(lr.intercept_))
        print('Coefficient: {}'.format(lr.coef_))
        evaluation = pd.DataFrame({'Model': [],
                                   'Details': [],
                                   'Mean Squared Error (MSE)': [],
                                   'R-squared (training)': [],
                                   'Adjusted R-squared (training)': [],
                                   'R-squared (test)': [],
                                   'Adjusted R-squared (test)': [],
                                   '5-Fold Cross Validation': []})
        r = evaluation.shape[0]
        evaluation.loc[r] = ['Simple Linear Regression', '-', msesm, rtrsm, '-', rtesm, '-', cv]
        evaluation