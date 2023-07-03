import pandas as pd
import numpy as np

import sklearn
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression

import re

class titanic_data_cleaning():

    def __init__(self, train_csv_path: str, test_csv_path: str):
        '''
        The object performs data cleaning on the Titanic data set.
        '''
        
        self.train_og_: pd.DataFrame = pd.read_csv(train_csv_path)
        self.test_og_: pd.DataFrame = pd.read_csv(test_csv_path)

        # Compute the group means of ages
        age_features_: list = ['Pclass', 'Sex', 'Embarked']
        self.group_mean_dict_: dict = self.train_og_.groupby(age_features_)['Age'].mean().to_dict()

        self.age_group_order_list_: list = ['[0, 10)', '[10, 20)', '[20, 30)', '[30, 40)', '[40, 50)', '[50, 60)', '60+']

        self.train_cleaned_: pd.DataFrame = self.clean_data(self.train_og_)
        self.test_cleaned_: pd.DataFrame = self.clean_data(self.test_og_)

    def clean_data(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        The function cleans the dataframe by filling the missing values
        in the columns 'Embarked', 'Age' accordingly.
        
        Parameters:
        -----------
            dataframe
                pd.DataFrame, Titanic dataset
        
        Returns:
        --------
            pd.DataFrame
                cleaned Titanic dataset
        '''

        # 1. Fix the 'Embarked' column by filling missing values
        dataframe = self.fix_Embarked(dataframe)

        # 2. Fix the 'Cabin' column by using the deck code instead
        dataframe = self.fix_Cabin(dataframe)

        # 3. Fix the 'Age' column by filling missing values
        #    Note this can only be done after fixing 'Embarked'.
        dataframe = self.fix_Age(dataframe)

        # 4. Replace columns 'SibSp', 'Parch' with 'group_size'
        dataframe = self.fix_SibSp_Parch(dataframe)

        # 5. Drop the 'Fare column
        dataframe = dataframe.drop('Fare', axis = 1)

        # 6. Drop the 'Name' column
        dataframe = dataframe.drop('Name', axis = 1)

        # 7. Drop the 'Ticket' column
        dataframe = dataframe.drop('Ticket', axis = 1)

        return dataframe
    
    def fix_Embarked(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        The function fills the missing values in the 'Embarked'
        column.

        Parameters:
        -----------
            dataframe
                pd.DataFrame, Titanic dataset
        
        Returns:
        --------
            pd.DataFrame
                dataframe with missing values in the 'Embarked' column filled.
        '''
        dataframe['Embarked'] = dataframe['Embarked'].fillna('S')

        return dataframe
    
    def fix_Age(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        The function fills the missing values in the 'Age'
        column with group means, and then assigen each row
        to an age group.

        Parameters:
        -----------
            dataframe
                pd.DataFrame, Titanic dataset
        
        Returns:
        --------
            pd.DataFrame
                dataframe with missing values in the 'Age' column filled.
        '''

        # Fill the missing values with 
        dataframe['age_pred'] = dataframe.apply(lambda row: self.group_mean_dict_[(row['Pclass'], row['Sex'], row['Embarked'])], axis = 1)
        dataframe['Age'] = dataframe['Age'].fillna(dataframe['age_pred'])

        # Categorize data into age groups
        # Make good use of vectorization here
        dataframe['age_index'] = np.divmod(dataframe['Age'], 10)[0]
        dataframe['age_index'] = np.where(dataframe['age_index'] > len(self.age_group_order_list_) - 1, len(self.age_group_order_list_) - 1, dataframe['age_index']).astype(int)
        dataframe['age_group'] = dataframe['age_index'].apply(lambda index: self.age_group_order_list_[index])
        dataframe['age_group'] = pd.Categorical(dataframe['age_group'], self.age_group_order_list_)

        # Drop the unwanted columns
        dataframe = dataframe.drop(['Age', 'age_pred', 'age_index'], axis = 1)

        return dataframe
    
    def fix_Cabin(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        The function replaces the original 'Cabin' column with
        a new column 'cabin_code'.

        Parameters:
        -----------
            dataframe
                pd.DataFrame, Titanic dataset
        
        Returns:
        --------
            pd.DataFrame
                dataframe 'Cabin' column replaced by 'cabin_code' column.
        '''

        dataframe['cabin_code']: pd.Series = dataframe['Cabin'].apply(self.get_cabin_code)

        dataframe = dataframe.drop('Cabin', axis = 1)

        dataframe = dataframe.rename(columns = {'cabin_code': 'Cabin'}
                                     )      
        return dataframe
    
    def get_cabin_code(self, my_input) -> str:
        '''
        The function returns the cabin code of a cabin number.

        Parameters:
        -----------
            dataframe
                pd.DataFrame, Titanic dataset
        
        Returns:
        --------
            Str
                Cabin code
        '''
        
        # Check input types first
        # input has to be either string or numpy.NaN
        if not isinstance(my_input, str):
            assert np.isnan(my_input), f'The input {my_input} is not a string nor numpy.NaN.'
            
            # If the input is already numpy.NaN, return deck "G"
            # since it was lowest complete deck that carried passengers.
            return 'G'
    
        # If the input is a string, then we extract the alphabets with regex.
        alpha_list: list = re.findall(r'[A-Za-z]', my_input)
    
        # We choose to return the smallest alphabet
        result: str = sorted(alpha_list)[0]
    
        if ord(result) > ord("G"):
            return "G"
        
        return result
    
    def fix_SibSp_Parch(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        The function replace the 'SibSp' and 'Parch' columns with their sum.

        Parameters:
        -----------
            dataframe
                pd.DataFrame, Titanic dataset
        
        Returns:
        --------
            pd.DataFrame
                dataframe with 'SibSp' and `Parch` columns replaced with
                'group_size'
        '''

        dataframe['group_size'] = dataframe['SibSp'] + dataframe['Parch']

        dataframe = dataframe.drop(['SibSp', 'Parch'], axis = 1)

        return dataframe