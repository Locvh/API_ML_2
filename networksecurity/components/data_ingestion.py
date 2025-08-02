from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


## configuration of the Data Ingestion Config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
            """
            Read data from mongodb
            """
            try:
                database_name=self.data_ingestion_config.database_name
                collection_name_aircarft = self.data_ingestion_config.collection_name_aircraft  # First collection
                collection_name_networkdata = self.data_ingestion_config.collection_name_product  # Second collection
                self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
                collection_aircarft = self.mongo_client[database_name][collection_name_aircarft]
                collection_name_networkdata = self.mongo_client[database_name][collection_name_networkdata]
                
                # df_aircraft=pd.DataFrame(list(collection_aircarft.find()))
                df_product=pd.DataFrame(list(collection_name_networkdata.find()))
                # Filter out rows where product_name is null
                # df_product = df_product[df_product['product_name'].notna()]
                df_product =  df_product.dropna(subset=['product_name', 'product_qty', 'product_weight_g'])

 
                n = 1000 # Số lượng sản phẩm trong df_product

                # Lấy dữ liệu chiều dài, chiều rộng, chiều cao và cân nặng của sản phẩm
                data = df_product.head(n)[['product_name', 'product_qty', 'product_weight_g', 
                                        'product_length_cm', 'product_width_cm', 'product_height_cm']].copy()

                # Đổi tên cột cho ngắn gọn và tính bkvol
                data.rename(columns={'product_length_cm': 'length',
                                    'product_width_cm': 'width',
                                    'product_height_cm': 'height',
                                    'product_weight_g': 'bkwt'}, inplace=True)

                # Tạo cột 'days_until_departure' giả lập
                data['days_until_departure'] = np.random.randint(1, 4, len(data))

                # Tính booked volume
                data['rcsvol'] = data['length'] * data['width'] * data['height']

                # Xác định các cột nhóm để phát hiện DMV
                group_cols = ['product_name', 'product_qty', 'bkwt']

                # Khởi tạo các cột cần thiết để tính toán
                data['bkvol'] = np.nan
                data['diff'] = np.nan
                data['DMV_flag'] = 0

                # Tính rcsvol và phát hiện DMV
                for key, group in data.groupby(group_cols):
                    mean_bkvol = group['rcsvol'].mean()

                    # Gán rcsvol là trung bình bkvol của nhóm
                    data.loc[group.index, 'bkvol']= mean_bkvol

                    # Tính phần trăm chênh lệch và đánh dấu DMV
                    diff = np.abs(group['rcsvol'] - mean_bkvol) / mean_bkvol
                    data.loc[group.index, 'diff'] = diff

                    data.loc[group.index, 'DMV_flag'] = (diff > 0.3).astype(int)

                # Gán product_type ngẫu nhiên cho từng nhóm
                unique_keys = list(data.groupby(group_cols).groups.keys())
                choices = ['gen', 'val', 'per', 'avi', 'dgr', 'arm', 'vun', 'hum']
                np.random.seed(123)
                group2type = dict(zip(unique_keys, np.random.choice(choices, len(unique_keys))))
                # group2type = {key: np.random.choice(choices, np.random.randint(1, 4), replace=False).tolist() for key in unique_keys}

                data['product_type'] = data[group_cols].apply(lambda row: group2type[tuple(row)], axis=1)
                data = pd.get_dummies(data, columns=['product_type'])
                data.drop(columns=['product_name'], inplace=True)
                
                # unique_keys = list(data.groupby(group_cols).groups.keys())
                # choices = ['gen', 'val', 'per', 'avi', 'dgr', 'arm', 'vun', 'hum']
                # np.random.seed(123)
                # group2type = {key: np.random.choice(choices, np.random.randint(1, 4), replace=False).tolist() for key in unique_keys}

                # data['product_type'] = data[group_cols].apply(lambda row: group2type[tuple(row)], axis=1)
                # data = data.explode('product_type', ignore_index=True)
                # data = pd.get_dummies(data, columns=['product_type'])

                return data
            
            except Exception as e:
                raise NetworkSecurityException
    

    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")

            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            logging.info(f"Exported train and test file path.")

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        


    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise NetworkSecurityException