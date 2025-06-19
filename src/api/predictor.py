import joblib 
import pandas as pd
from src.preprocessing.preprocessing_file import Preprocess
import os
import yaml

current_path = os.getcwd()
config_files_path = os.path.join(current_path, "../../config/config_files.yml" )

with open(os.path.join(current_path,config_files_path ), 'r') as f:
    config = yaml.safe_load(f)


class Predictor:
    def __init__(self,df):
        self.df = df

    def transformation(self):
        df_tr = self.df
        preprocess_obj = Preprocess(df_tr)
        df_tr = preprocess_obj.values_transformation()
        df_tr = preprocess_obj.features_transformation()

        return df_tr

    def load_model(self):
        model_path = config["files_path"]["model_full_path"]
        return joblib.load(model_path)
    
    def predict_churn(self, df):
        df = self.transformation(df)
        model = self.load_model()

        prediction = model.predict(df)

        return prediction[0]