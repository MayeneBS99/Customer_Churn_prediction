import joblib 
import pandas as pd
from src.preprocessing.preprocessing_file import Preprocess
import os
import yaml
from fastapi import HTTPException

current_path = os.path.dirname(os.path.abspath(__file__))
config_files_path = os.path.join(current_path, "../../config/config_files.yml" )

with open(config_files_path, 'r') as f:
    config = yaml.safe_load(f)


class Predictor:
    def __init__(self,dict):
        self.df = dict

    def transformation(self):
        df_tr = pd.DataFrame([self.df])
        preprocess_obj = Preprocess(df_tr)
        df_tr = preprocess_obj.run()

        return df_tr

    def load_model(self):
        model_path = config["files_path"]["model_full_path"]
        return joblib.load(model_path)
    
    def predict_churn(self):
        df = self.transformation()
        model = self.load_model()
        try :
            prediction = model.predict(df)
            return prediction[0]
        except Exception as e :
            print(f"[DEBUG] Erreur prédiction : {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")  
    
    def run(self):
        return self.predict_churn()