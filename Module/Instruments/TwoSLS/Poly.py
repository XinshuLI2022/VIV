from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from utils import set_seed

def run(exp, data, train_dict, log, device, resultDir, others):
    set_seed(train_dict['seed'])

    data.numpy()

    params = dict(poly__degree=range(1, 4), ridge__alpha=np.logspace(-5, 5, 11))
    pipe = Pipeline([('poly', PolynomialFeatures()), ('ridge', Ridge())])
    stage_1 = GridSearchCV(pipe, param_grid=params, cv=5)
    stage_1.fit(np.concatenate([data.train.z, 1-data.train.z, data.train.x], axis=1), data.train.t)
    t_hat = stage_1.predict(np.concatenate([data.train.z, 1-data.train.z, data.train.x], axis=1))

    pipe2 = Pipeline([('poly', PolynomialFeatures()), ('ridge', Ridge())])
    stage_2 = GridSearchCV(pipe2, param_grid=params, cv=5)
    stage_2.fit(np.concatenate([t_hat, data.train.x], axis=1), data.train.y)

    def estimation(data):
        return stage_2.predict(np.concatenate([data.t-data.t, data.x], axis=1)), stage_2.predict(np.concatenate([data.t, data.x], axis=1)), stage_2.predict(np.concatenate([data.e, data.x], axis=1))

    return estimation