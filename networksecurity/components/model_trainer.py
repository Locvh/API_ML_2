import os
import sys
import numpy as np

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
# from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.metric.regression_metric import get_regression_score


from sklearn.linear_model import LinearRegression
# from statsmodels.tsa.arima.model import ARIMA
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import (
#     AdaBoostClassifier,
#     GradientBoostingClassifier,
#     RandomForestClassifier,
# )
import mlflow
# from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='Locvh', repo_name='API_ML_2', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/Locvh/API_ML_2.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="Locvh"
os.environ["MLFLOW_TRACKING_PASSWORD"]="6ace8175dbe0347825f710613a326fc22ba5270e"




class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def track_mlflow(self,best_model,regressionmetric):
        mlflow.set_tracking_uri("https://dagshub.com/Locvh/API_ML_2.mlflow")
        mlflow.set_registry_uri("https://dagshub.com/Locvh/API_ML_2.mlflow")
        # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            rmse=regressionmetric.rmse
            mae=regressionmetric.mae
            mape=regressionmetric.mape
            ioa=regressionmetric.ioa
            ds=regressionmetric.ds

            

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)
            mlflow.log_metric("ioa", ioa)
            mlflow.log_metric("ds", ds)

            mlflow.sklearn.log_model(best_model, "model")
        

            # Model registry does not work with file store
            # if tracking_url_type_store != "file":

            #     # Register the model
            #     # There are other ways to use the Model Registry, which depends on the use case,
            #     # please refer to the doc for more information:
            #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            #     mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            # else:
            #     mlflow.sklearn.log_model(best_model, "model")


        
    def train_model(self,X_train,y_train,x_test,y_test):
        models = {
                "Multiple Linear Regression": LinearRegression(),
                # "ARIMA": None,  # ARIMA will be initialized during fitting due to its specific requirements
                # "Support Vector Regression": SVR(verbose=True),
                # "Neural Network": MLPRegressor( verbose=True,early_stopping=True,max_iter=100,random_state=42),
                # "Gradient Boosting Regression Tree": GradientBoostingRegressor( verbose=True,n_iter_no_change=5,validation_fraction=0.1,random_state=42),
            }
        params={
            "Multiple Linear Regression": {},
            # "ARIMA": {'order': [(5, 1, 0), (2, 1, 2), (4, 1, 1)]},
            # "Support Vector Regression": {
            #     'kernel': ['linear',  'rbf'],
            #     # 'kernel': ['linear'],
            #     'C': [1, 10],
            #     'gamma': ['scale']
            # },
            # "Neural Network": {
            #     'hidden_layer_sizes': [(10,), (50,)],
            #     'learning_rate_init': [0.01, 0.001],
            #     'alpha': [0.0001, 0.001, 0.01] 
            # },
            # "Gradient Boosting Regression Tree": {
            #     'learning_rate': [0.01, 0.05],
            #     'subsample': [ 0.7,  0.9],
            #     'n_estimators': [32, 64]
            # }
        }


         
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)

        regression_train_metric=get_regression_score(y_true=y_train,y_pred=y_train_pred)

        ## Track the experiements with mlflow
        self.track_mlflow(best_model,regression_train_metric)


        y_test_pred=best_model.predict(x_test)
        regression_test_metric=get_regression_score(y_true=y_test,y_pred=y_test_pred)

        self.track_mlflow(best_model,regression_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        #model pusher
        save_object("final_model/model.pkl",best_model)
        

        # Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=regression_train_metric,
                             test_metric_artifact=regression_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact



    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)