# from networksecurity.exception.exception import NetworkSecurityException
# from networksecurity.logging.logger import logging


# ## configuration of the Data Ingestion Config

# from networksecurity.entity.config_entity import DataIngestionConfig
# from networksecurity.entity.artifact_entity import DataIngestionArtifact
# import os
# import sys
# import numpy as np
# import pandas as pd
# import pymongo
# from typing import List
# from sklearn.model_selection import train_test_split
# from dotenv import load_dotenv
# load_dotenv()

# MONGO_DB_URL=os.getenv("MONGO_DB_URL")


# class DataIngestion:
#     def __init__(self,data_ingestion_config:DataIngestionConfig):
#         try:
#             self.data_ingestion_config=data_ingestion_config
#         except Exception as e:
#             raise NetworkSecurityException(e,sys)
        
#     def export_collection_as_dataframe(self):
#         """
#         Read data from mongodb
#         """
#         try:
#             database_name=self.data_ingestion_config.database_name
#             collection_name=self.data_ingestion_config.collection_name
#             self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
#             collection=self.mongo_client[database_name][collection_name]

#             df=pd.DataFrame(list(collection.find()))
#             if "_id" in df.columns.to_list():
#                 df=df.drop(columns=["_id"],axis=1)
            
#             df.replace({"na":np.nan},inplace=True)
#             return df
#         except Exception as e:
#             raise NetworkSecurityException
        
#     def export_data_into_feature_store(self,dataframe: pd.DataFrame):
#         try:
#             feature_store_file_path=self.data_ingestion_config.feature_store_file_path
#             #creating folder
#             dir_path = os.path.dirname(feature_store_file_path)
#             os.makedirs(dir_path,exist_ok=True)
#             dataframe.to_csv(feature_store_file_path,index=False,header=True)
#             return dataframe
            
#         except Exception as e:
#             raise NetworkSecurityException(e,sys)
        
#     def split_data_as_train_test(self,dataframe: pd.DataFrame):
#         try:
            
#             train_set, test_set = train_test_split(
#                 dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
#             )
#             logging.info("Performed train test split on the dataframe")

#             logging.info(
#                 "Exited split_data_as_train_test method of Data_Ingestion class"
#             )
            
#             dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
#             os.makedirs(dir_path, exist_ok=True)
            
#             logging.info(f"Exporting train and test file path.")
            
#             train_set.to_csv(
#                 self.data_ingestion_config.training_file_path, index=False, header=True
#             )

#             test_set.to_csv(
#                 self.data_ingestion_config.testing_file_path, index=False, header=True
#             )
#             logging.info(f"Exported train and test file path.")

            
#         except Exception as e:
#             raise NetworkSecurityException(e,sys)
        
        
#     def initiate_data_ingestion(self):
#         try:
#             dataframe=self.export_collection_as_dataframe()
#             dataframe=self.export_data_into_feature_store(dataframe)
            
#             self.split_data_as_train_test(dataframe)
#             dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
#                                                         test_file_path=self.data_ingestion_config.testing_file_path)
#             return dataingestionartifact

#         except Exception as e:
#             raise NetworkSecurityException

# ---------------------------------------------------------------------------------------------------------------------------------------


# from networksecurity.exception.exception import NetworkSecurityException
# from networksecurity.logging.logger import logging


# ## configuration of the Data Ingestion Config

# from networksecurity.entity.config_entity import DataIngestionConfig
# from networksecurity.entity.artifact_entity import DataIngestionArtifact
# import os
# import sys
# import numpy as np
# import pandas as pd
# import pymongo
# from typing import List
# from sklearn.model_selection import train_test_split
# from dotenv import load_dotenv
# load_dotenv()

# MONGO_DB_URL = os.getenv("MONGO_DB_URL")

# class DataIngestion:
#     def __init__(self, data_ingestion_config: DataIngestionConfig):
#         try:
#             self.data_ingestion_config = data_ingestion_config
#         except Exception as e:
#             raise NetworkSecurityException(e, sys)

#     def detect_dmvs(self, dataframe: pd.DataFrame, target_col: str, suspected_col: str, threshold_g1=0.5, threshold_g2=0.7):
#         """
#         Detect Disguised Missing Values (DMVs) in a suspected column based on a target column.
#         Args:
#             dataframe: DataFrame containing the data
#             target_col: Column with actual values (e.g., 'received_value')
#             suspected_col: Column suspected to contain DMVs (e.g., 'booked_value')
#             threshold_g1: Threshold for mean distance
#             threshold_g2: Threshold for normalized entropy
#         Returns:
#             DataFrame with an additional 'DMV_flag' column indicating suspected DMVs
#         """
#         try:
#             logging.info("Starting DMV detection")
#             dmv_flags = []

#             # Group by suspected column to calculate g1 and g2
#             grouped = dataframe.groupby(suspected_col)[target_col].apply(list)
            
#             for value, target_values in grouped.items():
#                 target_values = np.array(target_values)
                
#                 # Calculate g1: Mean distance
#                 mean_target = np.mean(target_values)
#                 g1 = (mean_target - value) ** 2
                
#                 # Calculate g2: Normalized entropy
#                 unique_vals, counts = np.unique(target_values, return_counts=True)
#                 probs = counts / len(target_values)
#                 entropy = -np.sum(probs * np.log(probs + 1e-10))  # Add small epsilon to avoid log(0)
#                 max_entropy = np.log(len(target_values) + 1e-10)
#                 g2 = entropy / max_entropy if max_entropy > 0 else 0
                
#                 # Flag as DMV if g1 or g2 exceeds thresholds
#                 is_dmv = (g1 > threshold_g1) or (g2 > threshold_g2)
#                 dmv_flags.extend([is_dmv] * len(target_values))

#             # Add DMV flag column to the original dataframe
#             dataframe['DMV_flag'] = dmv_flags
#             logging.info(f"Detected DMVs: {sum(dmv_flags)} out of {len(dataframe)} rows")
#             return  

#         except Exception as e:
#             raise NetworkSecurityException(e, sys)

#     # def export_collection_as_dataframe(self):
#     #     """
#     #     Read data from MongoDB and apply DMV detection
#     #     """
#     #     try:
#     #         database_name = self.data_ingestion_config.database_name
#     #         # collection_name = self.data_ingestion_config.collection_name
#     #         collection_name_aircarft = self.data_ingestion_config.collection_name_aricraft  # First collection
#     #         collection_name_networkdata = self.data_ingestion_config.collection_name_networkdata  # Second collection


#     #         self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
#     #         collection_aircarft = self.mongo_client[database_name][collection_name_aircarft]
#     #         collection_name_networkdata = self.mongo_client[database_name][collection_name_networkdata]
#     #         print(  f"Database Name: {database_name}")
#     #         print(  f"collection_name: {collection_aircarft}")
#     #         print(  f"collection: {collection_name_networkdata}")

#     #         # df = pd.DataFrame(list(collection.find()))
#     #         df_aircraft = pd.DataFrame(list(collection_aircarft.find()))
#     #         df_name_networkdata = pd.DataFrame(list(collection_name_networkdata.find()))


#     #         # Drop MongoDB '_id' column if present
#     #         if "_id" in df_aircraft.columns:
#     #             df_aircraft = df_aircraft.drop(columns=["_id"], axis=1)
#     #         if "_id" in df_name_networkdata.columns:
#     #             df_name_networkdata = df_name_networkdata.drop(columns=["_id"], axis=1)

#     #         # Replace 'na' with np.nan
#     #         df_aircraft.replace({"na": np.nan}, inplace=True)
#     #         df_name_networkdata.replace({"na": np.nan}, inplace=True)

#     #         # print('---------------------------------------------------')
#     #         # print(df_aircraft)
#     #         # print('---------------------------------------------------')
#     #         print(df_name_networkdata.columns)
#     #         print('---------------------------------------------------')
            
#     #         # # Apply DMV detection if required columns exist
#     #         # for df, name in [(df_aircraft, "aircraft"), (df_name_networkdata, "networkdata")]:
#     #         #     if 'target_column' in df.columns and 'suspected_column' in df.columns:
#     #         #         df = self.detect_dmvs(df, target_col='target_column', suspected_col='suspected_column')
#     #         #         logging.info(f"DMV detection completed for {name} dataset")
#     #         #     else:
#     #         #         logging.warning(f"DMV detection skipped for {name}: Required columns not found")

#     #         # return df_aircraft, df_name_networkdata

#     #         # Apply DMV detection only to network data
#     #         if 'resvol' in df_name_networkdata.columns and 'bkvol' in df_name_networkdata.columns:
#     #             df_networkdata = self.detect_dmvs(df_networkdata, target_col='resvol', suspected_col='bkvol')
#     #             logging.info("DMV detection completed for networkdata dataset")
#     #         else:
#     #             logging.warning("DMV detection skipped for networkdata: Required columns not found")

#     #         # Skip DMV detection for aircraft data
#     #         logging.info("Skipping DMV detection for aircraft data")

#     #         return df_aircraft, df_networkdata

#     #     except Exception as e:
#     #         logging.error(f"Error in export_collection_as_dataframe: {str(e)}")
#     #         raise NetworkSecurityException(e, sys)


#     def export_collection_as_dataframe(self):
#         """
#         Read data from MongoDB and apply DMV detection
#         """
#         try:
#             database_name = self.data_ingestion_config.database_name
#             collection_name_aircraft = self.data_ingestion_config.collection_name_aricraft  # First collection
#             collection_name_networkdata = self.data_ingestion_config.collection_name_networkdata  # Second collection

#             self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
#             collection_aircraft = self.mongo_client[database_name][collection_name_aircraft]
#             collection_networkdata = self.mongo_client[database_name][collection_name_networkdata]
            
#             print(f"Database Name: {database_name}")
#             print(f"collection_name: {collection_aircraft}")
#             print(f"collection: {collection_networkdata}")

#             # Lấy dữ liệu từ collection
#             df_aircraft = pd.DataFrame(list(collection_aircraft.find()))
#             df_networkdata = pd.DataFrame(list(collection_networkdata.find()))  

#             # Drop MongoDB '_id' column nếu có
#             if "_id" in df_aircraft.columns:
#                 df_aircraft = df_aircraft.drop(columns=["_id"], axis=1)
#             if "_id" in df_networkdata.columns:
#                 df_networkdata = df_networkdata.drop(columns=["_id"], axis=1)

#             # Thay 'na' bằng np.nan
#             df_aircraft.replace({"na": np.nan}, inplace=True)
#             df_networkdata.replace({"na": np.nan}, inplace=True)

#             # Áp dụng DMV detection chỉ cho network data
#             if 'resvol' in df_networkdata.columns and 'bkvol' in df_networkdata.columns:
#                 df_networkdata = self.detect_dmvs(df_networkdata, target_col='resvol', suspected_col='bkvol')
#                 logging.info("DMV detection completed for networkdata dataset")
#             else:
#                 logging.warning("DMV detection skipped for networkdata: Required columns not found")

#             # Bỏ qua DMV detection cho aircraft data
#             logging.info("Skipping DMV detection for aircraft data")

#             return df_aircraft, df_networkdata

#         except Exception as e:
#             logging.error(f"Error in export_collection_as_dataframe: {str(e)}")
#             raise NetworkSecurityException(e, sys)

#     def export_data_into_feature_store(self, dataframe: pd.DataFrame):
#         try:

#             # feature_store_file_path = self.data_ingestion_config.feature_store_file_path
#             # dir_path = os.path.dirname(feature_store_file_path)
#             # os.makedirs(dir_path, exist_ok=True)
#             # print('feature_store_file_path',feature_store_file_path)
#             # print('dir_path',dir_path)
#             # dataframe.to_csv(feature_store_file_path, index=False, header=True)
#             # return dataframe
        
#             df_aricraft, df_networkdata = dataframe
#             path_aricraft = self.data_ingestion_config.feature_store_file_path_aircarft
#             path_networkdata = self.data_ingestion_config.feature_store_file_path_networkdata

#             os.makedirs(os.path.dirname(path_aricraft), exist_ok=True)
#             os.makedirs(os.path.dirname(path_networkdata), exist_ok=True)

#             df_aricraft.to_csv(path_aricraft, index=False)
#             df_networkdata.to_csv(path_networkdata, index=False)

#             return df_aricraft, df_networkdata
#         except Exception as e:
#             raise NetworkSecurityException(e, sys)

#     def split_data_as_train_test(self, dataframe: pd.DataFrame):
#         try:
#              # Optional: Balance data based on DMV_flag if needed
#             if 'DMV_flag' in dataframe.columns:
#                 logging.info("Balancing data by undersampling non-DMV entries")
#                 dmv_df = dataframe[dataframe['DMV_flag'] == True]
#                 non_dmv_df = dataframe[dataframe['DMV_flag'] == False]
#                 if len(non_dmv_df) > len(dmv_df):
#                     non_dmv_df = non_dmv_df.sample(n=len(dmv_df), random_state=42)
#                 balanced_df = pd.concat([dmv_df, non_dmv_df])
#             else:
#                 balanced_df = dataframe

#             train_set, test_set = train_test_split(
#                 balanced_df, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42
#             )
#             logging.info("Performed train-test split on the dataframe")

#             dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
#             os.makedirs(dir_path, exist_ok=True)

#             train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
#             test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)
#             logging.info("Exported train and test file paths")
         
#         except Exception as e:
#             raise NetworkSecurityException(e, sys)

#     def initiate_data_ingestion(self):
#         try:
#             # dataframe = self.export_collection_as_dataframe()
#             # # dataframe = self.export_data_into_feature_store(dataframe)
#             # dataframe = self.export_data_into_feature_store(self.export_collection_as_dataframe())
#             # self.split_data_as_train_test(dataframe)
#             # data_ingestion_artifact = DataIngestionArtifact(
#             #     trained_file_path=self.data_ingestion_config.training_file_path,
#             #     test_file_path=self.data_ingestion_config.testing_file_path
#             # )
#             # return data_ingestion_artifact

#             df_aircraft, df_networkdata = self.export_collection_as_dataframe()
#             self.export_data_into_feature_store((df_aircraft, df_networkdata))
#             self.split_data_as_train_test(df_networkdata)
#             data_ingestion_artifact = DataIngestionArtifact(
#                 trained_file_path=self.data_ingestion_config.training_file_path,
#                 test_file_path=self.data_ingestion_config.testing_file_path
#             )
#             return data_ingestion_artifact
#         except Exception as e:
#             raise NetworkSecurityException(e, sys)



# ---------------------------------------------------------------------------------------------------------------------------------------




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
                
                df_aircraft=pd.DataFrame(list(collection_aircarft.find()))
                df_product=pd.DataFrame(list(collection_name_networkdata.find()))
                # Filter out rows where product_name is null
                df_product = df_product[df_product['product_name'].notna()]

                records = []
                # n_samples_per_label = 10000
                # records_0 = []
                # records_1 = []
                while len(records) < 20000:
                # while len(records_0) < n_samples_per_label or len(records_1) < n_samples_per_label:
                    aircarft = df_aircraft.sample(1).iloc[0]
                    num_items_on_aircarft = np.random.randint(0, 15)  # Số lượng sản phẩm trên máy bay
                    curr_weight = np.random.uniform(0, aircarft['Max_Weight_kg'] * 1.2)  # Cho vượt lên 120%
                    curr_volume = np.random.uniform(0, aircarft['Max_Volume_m'] * 1.2)
                    item = df_product.sample(1).iloc[0]
                    will_fit = (curr_weight + (item['product_weight_kg']*item['product_qty'])  <= aircarft['Max_Weight_kg']) or (curr_volume + (item['product_volume_m']*item['product_qty'])<= aircarft['Max_Volume_m'])
                    label = int(will_fit)
                    # Chỉ append nếu muốn tỉ lệ nhãn đều nhau
                    if label == 1 and np.random.rand() < 0.5:   # Giảm bớt nhãn 1
                        continue
                 
                    record = {
                        'Max_Weight_kg': aircarft['Max_Weight_kg'],
                        'Max_Volume_m': aircarft['Max_Volume_m'],
                        'current_weight': curr_weight,
                        'current_volume': curr_volume,
                        'num_items_on_aircarft': num_items_on_aircarft,
                        'item_weight': item['product_weight_kg'],
                        'item_volume': item['product_volume_m'],
                        'weight_left': aircarft['Max_Weight_kg'] - curr_weight,
                        'volume_left': aircarft['Max_Volume_m'] - curr_volume,
                        'weight_ratio': item['product_weight_kg'] / (aircarft['Max_Weight_kg'] - curr_weight + 1e-6),
                        'volume_ratio': item['product_volume_m'] / (aircarft['Max_Volume_m'] - curr_volume + 1e-6),
                        'label': label,
                        'DMV_FLAG':1
                    }
                #     if label == 0 and len(records_0) < n_samples_per_label:
                #         records_0.append(record)
                #     elif label == 1 and len(records_1) < n_samples_per_label:
                #         records_1.append(record)
                #     # Nếu đã đủ mỗi label thì thôi, khỏi append nữa

                # # Ghép lại thành DataFrame tổng hợp
                # df_ml = pd.DataFrame(records_0 + records_1)
                # df_ml = df_ml.sample(frac=1).reset_index(drop=True)  # shuffle lại cho ngẫu nhiên


                    records.append(record)

                df_ml = pd.DataFrame(records)

                return df_ml
            
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