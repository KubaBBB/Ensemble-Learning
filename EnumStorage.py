from enum import Enum

class LabelMapper(Enum):
    bedrooms = 0,
    bathrooms = 1,
    sqft_living = 2,
    sqft_lot = 3,
    floors = 4,
    waterfront = 5,
    view = 6,
    condition = 7,
    grade = 8,
    sqft_above = 9,
    sqft_basement = 10,
    yr_built = 11,
    yr_renovated = 12,
    zipcode = 13,
    lat = 14,
    long = 15,
    sqft_living15= 16,
    sqft_lot15 = 17,


class SplitDataset(Enum):
    NONE = 0,
    BAGGING = 1,
    AGENT = 2,

class Ensemble(Enum):
    ARITHMETIC = 0,
    WEIGHTED = 1,

class Model(Enum):
    BAYESIAN_RIDGE = 0
    K_NEIGHBORS = 1
    DECISION_TREE = 2
    SVR = 3