import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class RegressionMetricArtifact:
    def __init__(self, rmse, mae, mape, ioa, ds):
        self.rmse = rmse
        self.mae = mae
        self.mape = mape
        self.ioa = ioa
        self.ds = ds

def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # IOA - Index of Agreement
    denominator = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    ioa = 1 - (np.sum((y_true - y_pred) ** 2) / denominator) if denominator != 0 else 0

    # DS - Nash–Sutcliffe Efficiency
    ds_numerator = np.sum((y_true - y_pred) ** 2)
    ds_denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    ds = 1 - ds_numerator / ds_denominator if ds_denominator != 0 else 0

    return RegressionMetricArtifact(rmse=rmse, mae=mae, mape=mape, ioa=ioa, ds=ds)
