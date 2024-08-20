import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, TunedThresholdClassifierCV
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.svm import SVR

class DataObject:

    def __init__(self, df = None):
        self.df = df if df is not None else pd.DataFrame()
    
    def load_csv(self, filepath):
        '''Loads CSV file into dataframe from given filepath
        
        Args - filepath (str) : Input filepath'''

        try:

            self.df = pd.read_csv(filepath)
        
        except:

            raise NotImplementedError('Could not load file')

    def drop_null_values(self):
        '''Drops any rows containing null values'''
        try:
            
            self.df = self.df.dropna()
        
        except:

            raise NotImplementedError('Could not drop null values')
    
    def fix_data_types(self, column, data_type):
        '''Sets data type of selected column
        
        Args -  column(str) : Specified column from dataframe
                data_type(class) : Specified data_type to convert to
        '''

        try:

            try:
                self.df[column] = self.df[column].astype(data_type, errors='raise')
            
            except:
                if data_type in [float, int]:
                    self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
                else:
                    raise NotImplementedError('Could not convert values')

        
        except:

            raise NotImplementedError('Could not convert values')
        
    
    
    def map_target_variable(self):

        self.df['Churn'] = self.df['Churn'].map({'Yes' : 1, 'No' : 0})


    def remove_outliers(self):
        '''Removes outliers outside of 1.5 x inter-quartile range for numerical columns'''
    

        try:

            for column in self.df:
                if self.df[column].dtype in ['integer', 'float']:
                    q25 = self.df[column].quantile(0.25)
                    q75 = self.df[column].quantile(0.75)
                    iqr = q75-q25

                    self.df = self.df[(self.df[column] > (q25 - iqr*1.5)) & (self.df[column] <= (q75 + iqr*1.5))]

            self.df = self.df.reset_index(drop=True)
        
        except:
            raise NotImplementedError("Unable to remove outliers")



class PreProcessData:

    def __init__(self, df):
        self.df = df.drop(['Churn', 'customerID'], axis = 1)
                    
        

    def yeo_johnson_transform(self, column):
        '''Yeo-Johnson transform selected columns for normal distribution

        Args - column(str) : Column to apply transformation to
        '''

        try:
            pow_trans = PowerTransformer(method='yeo-johnson', standardize=False)

            self.df[column] = pow_trans.fit_transform(self.df[[column]])
        
        except:
            raise NotImplementedError('Unable to transform column')


    def scale_data(self):
        '''Scale numerical data for modelling using robust scaling'''

        try:
            numerical_variables = self.df.select_dtypes(['integer', 'float'])
            numerical_columns = numerical_variables.columns.values
        
            
            scaler = RobustScaler()

            scaled_variables_arr = scaler.fit_transform(numerical_variables)
            scaled_variables_df = pd.DataFrame(data=scaled_variables_arr, columns=numerical_columns)

            for column in scaled_variables_df:
                self.df[column] = scaled_variables_df[column]
        
        except:
            raise NotImplementedError("Unable to scale numerical columns")



    def one_hot_encode_data(self):
        '''One hot encodes categorical variables'''

        try:
            categorical_variables = self.df.select_dtypes('object')
            
            ohe = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

            ohe_variables = ohe.fit_transform(categorical_variables)

            self.df = pd.concat([self.df, ohe_variables], axis=1).drop(categorical_variables.columns, axis = 1)

        except:
            raise NotImplementedError("Unable to encode categorical variables")



class FeatureSelection:
    
    def __init__(self, df, target):
    
        self.df = df
        self.target = target
    

    def rfe_selection(self):
        '''Uses recursive feature elimination to select features with predictive power'''

        try:
            X = self.df
            y = self.target

            estimator = SVR(kernel = 'linear')

            selector = RFECV(estimator, step=1 , cv=5).set_output(transform="pandas")

            self.df = selector.fit_transform(X,y)

        except:
            raise NotImplementedError('Unable to select features')
