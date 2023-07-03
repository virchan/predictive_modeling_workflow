import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Activation)

class titanic_machine_learning_models():

    def __init__(self) -> None:

        self.models_ : dict = {'dummy': DummyClassifier(), 
                               'tree': DecisionTreeClassifier(), 
                               'forest': RandomForestClassifier(n_estimators = 201), 
                               'support_vector': SVC(C = 1000, 
                                                     gamma = 0.01, 
                                                     kernel = 'rbf',
                                                     probability = True), 
                               'neural_network': self.tf_sequential_model(), 
                               'logistic': LogisticRegression(solver = 'liblinear'), 
                               'gaussian_NB': GaussianNB(var_smoothing = 0.1), 
                               'bernoulli_NB': BernoulliNB(), 
                               'adaboost': AdaBoostClassifier(n_estimators = 61)                         
                               }

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.DataFrame) -> None:
        '''
        The function trains the classifiers in
        self.models_

        Parameters:
        -----------
            X 
                pd.DataFrame, features of the dataset

            y
                pd.DataFrame, labels of the dataset
        
        Returns:
        --------
            None
                Classifiers in self.models_ are now fitted.
        '''

        for name in self.models_.keys():
            print(f"Training {name}...")

            if name == 'neural_network':
                self.models_[name].fit(x = X.values,
                                       y = tf.one_hot(y, 2), 
                                       epochs = 100,
                                       verbose = 0
                                      )
            
            else:
                self.models_[name].fit(X, y)
            
            print(f"Model {name} is ready!\n")

        print('All models are ready.')

    def predict(self,
                X: pd.DataFrame, 
                ) -> pd.DataFrame:
        '''
        The function generates predictions for 
        all model given in self.models_.

        Parameters:
        -----------
            X
                pd.DataFrame, dataset
        
        Returns:
        --------
            pd.DataFrame
                Predictions from each self.models_ estimator
        '''

        Predictions_df: pd.DataFrame = pd.DataFrame()

        print('Generating predictions for each model...')
        
        for name, model in self.models_.items():
            print(f'Predicting with {name}...')
            y_pred: np.array = model.predict(X)

            if name == 'neural_network':
                y_pred = np.argmax(y_pred, axis = 1)

            Predictions_df[name] = y_pred
        
        assert Predictions_df.shape[0] == X.shape[0], 'Sample sizes mismatch!'
        assert Predictions_df.shape[1] == len(self.models_.keys()), 'Numbers of models mismatch!'
        assert not (Predictions_df.isnull().values.any()), 'Data frame has NaN!'

        return Predictions_df
    
    def predict_proba(self, 
                      X: pd.DataFrame
                     ) -> pd.DataFrame:
        '''
        The function computes probabilities of positive class 
        for all model given in self.models_.

        Parameters:
        -----------
            X
                pd.DataFrame, samples
        
        Returns:
        --------
            pd.DataFrame
                Positive class probabilities from each self.models_ estimator
        '''

        print('Generating probabilities for each model...')

        survival_probabilities_df: pd.DataFrame = pd.DataFrame()

        for name, model in self.models_.items():
            print(f'Computing {name} survival probabilities...')

            if name == 'neural_network':
                survival_probabilities_df[name] = model.predict(X)[:, 1]
            
            else:
                survival_probabilities_df[name] = model.predict_proba(X)[:, 1]

        print("All probabilities are computed!")

        return survival_probabilities_df

    def transform_data(dataframe: pd.DataFrame) -> pd.DataFrame:
        '''
        The function transforms the categorical columns into binary columns.

        Parameters:
        -----------
            dataframe
                pd.DataFrame, samples
        
        Returns:
        --------
            pd.DataFrame
                transformed dataframe
        '''

        categorical_features_list: list = ['Pclass', 'Sex', 'Embarked', 'Cabin', 'age_group', 'group_size']

        return pd.get_dummies(dataframe,
                              columns = categorical_features_list, 
                              drop_first = True
                              )

    def tf_sequential_model(self) -> tf.keras.models.Sequential:
        '''
        The function returns the pre-defined
        Feedforward Neural Network.

        The neural network used in this binary classification.
        It requires the labels to be one-hot encoded, and
        outputs the probabilities for each class.

        Parameters:
        -----------
            dataframe
                pd.DataFrame, samples
        
        Returns:
        --------
            tf.keras.models.Sequential
                Feedforward Neural Network
        '''
        
        model: tf.keras.models.Sequential = Sequential()
        
        model.add(Dense(units = 26, activation = 'relu'))
        model.add(Dense(units = 13, activation = 'relu'))
        model.add(Dense(units = 7, activation = 'relu'))
        model.add(Dense(units = 4, activation = 'relu'))
        model.add(Dense(units = 2, activation = 'softmax'))

        model.compile(loss = 'binary_crossentropy', 
                      optimizer = 'adam', 
                      metrics = ['accuracy', tf.keras.metrics.AUC()]
                      )
        
        return model
