# files dedicated to the preprocessinf of data before prediction

import pandas as pd
import numpy as np
import yaml
import os
import joblib

current_path = os.path.dirname(os.path.abspath(__file__))
config_files_path = os.path.join(current_path, "../../config/config_files.yml" )

with open(config_files_path, 'r') as f:
    config = yaml.safe_load(f)


class Preprocess:

    """
    This class allow us to modify data before perform prediction on it

    """
    def __init__(self, df):
        self.df = df

    def values_transformation(self):
        """
        Transform values of categorical features

        """
        transformation_1_df = self.df.copy()
        transformation_1_df["COLLEGE"] = transformation_1_df["COLLEGE"].replace({'zero': 0,
                               'one' : 1}).astype(int)

        transformation_1_df["LESSTHAN600k"] = transformation_1_df["LESSTHAN600k"].replace({'False': 0,
                                    'True': 1}
                                   ).astype(int)

        transformation_1_df["REPORTED_SATISFACTION"] = transformation_1_df["REPORTED_SATISFACTION"].replace({'very_unsat': 0,
                                                    'unsat': 1,
                                                    'avg' : 2,
                                                    'sat': 3,
                                                    'very_sat': 4}
                                                    ).astype(int)
        transformation_1_df["REPORTED_USAGE_LEVEL"]= transformation_1_df["REPORTED_USAGE_LEVEL"].replace({'very_little': 0,
                                             'little': 1,
                                             'avg' : 2,
                                             'high': 3,
                                             'very_high': 4}
                                             ).astype(int)

        transformation_1_df["CONSIDERING_CHANGE_OF_PLAN"]= transformation_1_df["CONSIDERING_CHANGE_OF_PLAN"].replace({'actively_looking_into_it': 0,
                                                    'considering': 1,
                                                    'perhaps' : 2,
                                                    'no': 3,
                                                    'never_thought': 4}
                                                    ).astype(int)
       
        self.df = transformation_1_df

        return self.df
        
    def features_transformation(self):
        """
        In this method we are giong to transform our features in order to allow 
        prediction

        """
        # object loading
        imputer_cat = joblib.load(os.path.join(current_path, config["files_path"]["imputer_cat"]))
        imputer_quant = joblib.load(os.path.join(current_path, config["files_path"]["imputer_quant"]))
        encoder = joblib.load(os.path.join(current_path, config["files_path"]["encoder_path"]))
        standard = joblib.load(os.path.join(current_path, config["files_path"]["standardization"]))

        categoricals_variables = ['COLLEGE',
                                    'LESSTHAN600k',
                                    'JOB_CLASS',
                                    'REPORTED_SATISFACTION',
                                    'REPORTED_USAGE_LEVEL',
                                    'CONSIDERING_CHANGE_OF_PLAN']
        
        numerical_variables = ['DATA',
                                    'INCOME',
                                    'OVERCHARGE',
                                    'LEFTOVER',
                                    'HOUSE',
                                    'CHILD',
                                    'REVENUE',
                                    'HANDSET_PRICE',
                                    'OVER_15MINS_CALLS_PER_MONTH',
                                    'TIME_CLIENT',
                                    'AVERAGE_CALL_DURATION']

        transformation_2_df = self.df.copy()

        transformation_2_df_cat = imputer_cat.transform(transformation_2_df[categoricals_variables])
        transformation_2_df_cat = pd.DataFrame(transformation_2_df_cat, columns = categoricals_variables)

        cat_2= encoder.transform(transformation_2_df_cat[["JOB_CLASS"]])
        variables = encoder.get_feature_names_out()
        cat_2 = pd.DataFrame(cat_2.toarray(), columns =variables )

        for var in variables :
            cat_2[var] = cat_2[var].astype("int")

        var2 = [var for var in categoricals_variables if var != "JOB_CLASS"]
        transformation_2_df_cat = pd.concat([transformation_2_df_cat[var2], cat_2], axis = 1)
        

        # quantitative variables 
        
        transformation_2_df_quant = imputer_quant.transform(transformation_2_df[numerical_variables])
        transformation_2_df_quant = pd.DataFrame(transformation_2_df_quant, columns = numerical_variables)

        transformation_2_df_quant = standard.transform(transformation_2_df_quant)
        transformation_2_df_quant = pd.DataFrame(transformation_2_df_quant, columns = numerical_variables)

        # data final
        transformation_final_df = pd.concat([transformation_2_df_quant, transformation_2_df_cat], axis = 1)
        
            

        self.df = transformation_final_df.copy()
        return self.df
    
    def run(self):
        self.values_transformation()
        self.features_transformation()
        return self.df
                