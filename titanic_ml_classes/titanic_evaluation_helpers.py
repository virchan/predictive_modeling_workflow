import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import (confusion_matrix, 
                             classification_report,
                             roc_curve, 
                             RocCurveDisplay,
                             roc_auc_score, 
                             precision_recall_curve, 
                             PrecisionRecallDisplay
                            )

import scipy.stats as stats
import statsmodels
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import (proportions_ztest, 
                                          proportion_confint)

class titanic_evaluation_helpers():

    def two_proportion_Z_test(data: pd.DataFrame, 
                              alpha_: float = 0.05,
                              ) -> pd.DataFrame:
        '''
        The function performs the two proportion Z-test
        on the given data set. It displays the means of both groups, their confidence
        intervals, p-values, and critical z-scores.

        Parameters:
        -----------
            data 
                pd.DataFrame, data set for the two proportion Z-test.

            alpha_
                Float, significant level of the two proportion Z-test.
                Default: ``0.05``.
        
        Returns:
        --------
            pd.DataFrame
                Conclusion of the two proportion Z-test.
        '''
    
        print(f"Performing (Two-tailed) Two Proportion Z-Test on Training and Testing Accuracies")
        print(f'with signaficant level alpha = {alpha_}.')
        print('The null hypothesis is training and testing accuracies are the same.')
        print('The alternative hypothesis is training and testing accuracies are not the same.')
        print('\n')

        # Initiating the returning dataframe.
        two_proportion_Z_test_df: pd.DataFrame = pd.DataFrame()

        models: pd.indexes = data.drop(['PassengerId', 'train'],
                                       axis = 1
                                       ).columns

        two_proportion_Z_test_df['model'] = models

        columns_list: list = ['accuracy_train', 
                              'accuracy_test', 
                              'CI_train',
                              'CI_test', 
                              'p_value', 
                              'critical_z_score',
                              'reject_null'
                              ]
        
        # Initiating the columns for the returning dataframe.
        accuracy_trains: list = []
        accuracy_tests: list = []
        
        CI_trains: list = []
        CI_tests: list = []  

        z_scores: list = []
        p_values: list = []
        reject_nulls: list = []

        z_critical: float = stats.norm.ppf(1 - alpha_ / 2)

        for model in models:

            # Number of successful trials and total number of observations
            successes, nobs = titanic_evaluation_helpers.get_successes_and_nobs(data, 
                                                                                model
                                                                                )
            
            z_score, p_value = proportions_ztest(count = successes, 
                                                 nobs = nobs
                                                 )

            reject_null: int = titanic_evaluation_helpers.Hypothesis_Test_conclude(z_score, 
                                                                                   z_critical,
                                                                                   p_value, 
                                                                                   alpha_
                                                                                   )

            (lower_train, lower_test), (upper_train, upper_test) = proportion_confint(count = successes, 
                                                                                      nobs = nobs, 
                                                                                      alpha = alpha_
                                                                                     )
            
            # Store the results
            accuracy_trains.append(successes[0] / nobs[0])
            accuracy_tests.append(successes[1] / nobs[1])

            CI_trains.append((lower_train, upper_train))
            CI_tests.append((lower_test, upper_test))

            z_scores.append(z_score)
            p_values.append(p_value)
            reject_nulls.append(reject_null)
        
        # Debug
        assert len(accuracy_trains) == len(accuracy_tests) == len(CI_trains) == len(CI_tests) == len(p_values) == len(z_scores) == len(reject_nulls), 'Sample sizes mismatch!'

        assert accuracy_trains, 'No features are computed!'

        two_proportion_Z_test_df['accuracy_train'] = accuracy_trains
        two_proportion_Z_test_df['accuracy_test'] = accuracy_tests

        two_proportion_Z_test_df['CI_train'] = CI_trains
        two_proportion_Z_test_df['CI_test'] = CI_tests

        two_proportion_Z_test_df['z_score'] = z_scores
        two_proportion_Z_test_df['p_value'] = p_values
        two_proportion_Z_test_df['reject_null'] = reject_nulls

        print('Hypothesis Testing is completed!')

        return two_proportion_Z_test_df
    
    def Hypothesis_Test_conclude(z_score: float,
                                 z_critical: float,
                                 p_value: float, 
                                 alpha_: float
                                 ) -> int:   
        '''
        The function concludes of the hypothesis testing using 
        the p-value approach and the critical z-score approach.
        
        Parameters:
        -----------
            z_scre,
                Float, z-score
            
            z_critical,
                Float, critical z-score

            p_value
                Float, p-value
                
            alpha
                Float, significant level of the hypothesis test.
                Default: ``0.05``.                

        Returns:
        --------
            int
                0 if both approaches failed to reject the null hypothesis,
                1 if both approaches reject the null hypothesis,
                -1 if the approaches disagree
        '''

        reject_p: bool = (p_value <= alpha_)

        reject_z: bool = (z_score > z_critical)

        if reject_p != reject_z:
            return -1
        
        return int(reject_z)
    
    def get_successes_and_nobs(data: pd.DataFrame, 
                               model: str
                               ) -> tuple:
        '''
        The function returns the number of
        successes and number of observations
        from the data.
        
        Parameters:
        -----------

            data
                pd.DataFrame, data

            name
                str, name of the model, should be one of the columns
                of data.

        Returns:
        --------
            tuple
                tuple of two lists, where the 0-th one
                is the number of successes in each data,
                and the 1-st one is the number of observations.
        '''    

        successes_train: int = data[data['train'] == 1][model].sum()
        successes_test: int = data[data['train'] == 0][model].sum()

        nob_train: int = data[data['train'] == 1][model].count()
        nob_test: int = data[data['train'] == 0][model].count()

        return [successes_train, successes_test], [nob_train, nob_test]
    
    def get_accuracy(correct_predictions: pd.DataFrame) -> pd.DataFrame:
        '''
        The function computes the accuracies for each model
        in both training and testing phrase.

        Parameters:
        -----------
            correct_predictions
                pd.DataFrame, binary dataframe indicating
                if a sample (row) is correctly predicted by the
                model (column).
        
        Returns:
        --------
            ret
                pd.DataFrame, dataframe of accuracies.
        '''
        accuracy_df: pd.DataFrame = correct_predictions.drop('PassengerId', axis = 1).groupby('train', as_index = False).agg('mean')

        # Formatting
        accuracy_df = accuracy_df.drop('train', axis = 1).T.reset_index()

        temp_1: pd.DataFrame = accuracy_df[['index', 1]]
        temp_2: pd.DataFrame = accuracy_df[['index', 0]]
            
        temp_1 = temp_1.rename(columns = {'index': 'model', 
                                        1: 'accuracy'
                                        }
                            )

        temp_2 = temp_2.rename(columns = {'index': 'model', 
                                        0: 'accuracy'
                                        }
                            )
            
        temp_1['train'] = 1
        temp_2['train'] = 0

        accuracy_df = pd.concat([temp_1, temp_2], axis = 0).reset_index(drop = True)

        return accuracy_df
    
    def compute_metrics(predictions: pd.DataFrame, 
                        metrics: dict = {}
                       ) -> pd.DataFrame:
        '''
        The function computes the confusion
        matrix-based metrics on a given
        prediction result
        
        Parameters:
        -----------
            predictions
                pd.DataFrame, predictions made by each model.
            
            metrics
                Dict, dictionary of metrics
                Default: ``{}``.

        Returns:
        --------
            pd.DataFrame
                dataframe with models as rows, and
                2 * len(metrics.keys()) columns,
                graded by training and testing.
        '''
        
        model_list: list = predictions.drop(['PassengerId', 'y_true', 'train'], axis = 1).columns
        
        values: np.array = np.array([0] * (2 * len(model_list)) * 5, 
                                    dtype = int
                                ).reshape(2 * len(model_list), 5)
        
        index_list: list = []
        
        for name in model_list:
            index_list.append(name)
            index_list.append(name)
            
        for row, model in enumerate(model_list):
                values[2 * row, 1: ] = confusion_matrix(predictions[predictions['train'] == 1]['y_true'], 
                                                        predictions[predictions['train'] == 1][model]
                                                        ).ravel()
                
                values[2 * row + 1, 1: ] = confusion_matrix(predictions[predictions['train'] == 0]['y_true'], 
                                                            predictions[predictions['train'] == 0][model]
                                                            ).ravel()
                
                values[2 * row, 0] = 1
                values[2 * row + 1, 0] = 0
            
                                    
        result_df: pd.DataFrame = pd.DataFrame(data = values, 
                                            index = index_list, 
                                            columns = ['train', 'TN', 'FP', 'FN', 'TP']
                                            )
            
        result_df = result_df.reset_index()
        
        result_df = result_df.rename(columns = {'index': 'model'})
            
        for name, metric in metrics.items():
            result_df[name] = result_df.agg(metric, axis = 1)
            
                                    
        return result_df 
    
    def metrics_init() -> list:
        '''
        The function 
        
        Parameters:
        -----------

        Returns:
        --------
            List
                list [formula, visual_title, axis_indices]
                of dictionaries. The 0th one contains the actual 
                formula to compute the metric, the 1st
                one contains the strings for the visual title,
                and the 2nd one contains the indices for matplotlib
                axis.
        '''
        # Stores the function that computes the metric.
        metrics_dict: dict = {}
        
        # Stores the visual's title.
        metrics_visual_title: dict = {}
            
        # Stores the axis indices.
        axis_indices: dict = {}

        # Minimizing FN
        # True Positive Rate = Sensitvity
        metrics_dict['TPR'] = lambda row: row['TP'] / (row['TP'] + row['FN'])
        metrics_visual_title['TPR'] = 'True Positive Rate (Sensitivity)\n' + r'$TPR = \frac{TP}{TP+FN}$'
        axis_indices['TPR'] = (0, 0)
        
        # Negative Predictive Value = Precision for the Negative class
        metrics_dict['NPV'] = lambda row: row['TN'] / (row['TN'] + row['FN'])
        metrics_visual_title['NPV'] = 'Negative Predictive Value\n' + r'$NPV = \frac{TN}{TN + FN}$'
        axis_indices['NPV'] = (1, 0)

        # Minimizing FP
        # True negative rate (TNR)
        metrics_dict['TNR'] = lambda row: row['TN'] / (row['FP'] + row['TN'])
        metrics_visual_title['TNR'] = 'True Negative Rate (Specificity)\n' + r'$TNR = \frac{TN}{TN + FP}$'
        axis_indices['TNR'] = (0, 1)

        # Positive predictive value (PPV)
        metrics_dict['PPV'] = lambda row: row['TP'] / (row['FP'] + row['TP']) if (row['FP'] + row['TP']) else 0
        metrics_visual_title['PPV'] = 'Positive Predictive Value\n' + r'$PPV = \frac{TP}{TP + FP}$'
        axis_indices['PPV'] = (1, 1)

        # Balanced metric (minimizing both FN and FP)
        # Balanced accuracy (BA)
        metrics_dict['BA'] = lambda row: 1 / 2 * (row['TP'] / (row['TP'] + row['FN']) + row['TN'] / (row['TN'] + row['FP']))
        metrics_visual_title['BA'] = 'Balanced Accuracy\n' + r'$BA = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$'
        axis_indices['BA'] = (0, 2)

        # F1-score
        metrics_dict['F1'] = lambda row: (2 * row['TP']) / (2 * row['TP'] + row['FP'] + row['FN'])
        metrics_visual_title['F1'] = r'$F$-score' + '\n' + r'$F_1 = \frac{2TP}{2TP + FP + FN}$'
        axis_indices['F1'] = (1, 2)
        
        assert metrics_dict.keys() == metrics_visual_title.keys() == axis_indices.keys(), 'Keys mismatch!'
        
        return [metrics_dict, metrics_visual_title, axis_indices]
    
    def plot_confusion_matrices(data: pd.DataFrame) -> None:
        '''
        The function plots the confusion matrices as bar graphs for the given
        data set.

        Parameters:
        -----------
            data 
                pd.DataFrame, data set for the confusion matrices
        
        Returns:
        --------
            None
                It plots the visual.
        '''

        fig, ax = plt.subplots(2,
                            2,
                            figsize = (30, 13)
                            )

        plt.subplots_adjust(hspace = 0.3)

        # Visual title
        fig.suptitle('Six Confusion Matrices For Binary Classifiers', 
                    fontsize = 40
                    )

        plt.subplots_adjust(hspace = 0.3,
                            wspace = 0.3
                        )

        # The indices for each confusion matrix component.
        axis_dict: dict = {'TP': (0, 0), 
                           'FP': (0, 1),
                           'FN': (1, 0),
                           'TN': (1, 1)
                          }

        # The visual title for each confusion matrix component.    
        title_dict: dict = {'TP': 'True Positives', 
                            'FP': 'False Positives (Type I Error)', 
                            'FN': 'False Negatives (Type II Error)', 
                            'TN': 'True Negatives'
                           }

        # The palette for each confusion matrix component.      
        palette_dict: dict = {'TP': 'binary', 
                              'FP': 'hot',
                              'FN': 'crest', 
                              'TN': 'binary'
                             }

        # Plotting the confusion matrices       
        for key, value in axis_dict.items():
            
            ax[value] = sns.barplot(data = data, 
                                    x = 'model', 
                                    y = key, 
                                    hue = 'train',
                                    hue_order = [1, 0],
                                    ax = ax[value],
                                    width = 0.85,
                                    palette = palette_dict[key]
                                    )

            ax[value].set_title(title_dict[key], 
                                fontsize = 15, 
                                bbox = dict(facecolor = 'azure', 
                                            alpha = 0.5, 
                                            edgecolor = 'azure', 
                                            boxstyle = 'round, pad=0.5'
                                        )
                                )

        # Configuring the coordinate axes and legends
        leg_labels: list = ['train', 'test']
            
        for axis in ax.reshape(-1):
            
            
            # Set the axis spines
            axis.spines[['right', 'top']].set_visible(False)
            
            # Legend position
            axis.legend(bbox_to_anchor = (1.02, 0.5))
            
            # Display data on top of the bars
            for container in axis.containers:
                axis.bar_label(container, 
                            fmt='%.0f', 
                            padding = 2, 
                            fontsize = 8
                            )

            # Legend label
            for text, label in zip(axis.axes.get_legend().texts, leg_labels):
                text.set_text(label)

        plt.show()

    def get_confusion_matrices(predictions: pd.DataFrame
                              ) -> pd.DataFrame:
        '''
        The function computes the confusion
        matrix-based metrics on a given
        prediction result
        
        Parameters:
        -----------
            predictions
                pd.DataFrame, predictions made by each model.
                The true label is given by the column `y_true`.
            
        Returns:
        --------
            pd.DataFrame
                dataframe with fives columns: `TP`, `FP`, `FN`, `TN`, 
                and `train`. The last column indicates whether a sample
                is from the training set.
        '''
        
        model_list: list = predictions.drop(['PassengerId', 'y_true', 'train'], axis = 1).columns
        
        values: np.array = np.array([0] * (2 * len(model_list)) * 5, 
                                    dtype = int
                                ).reshape(2 * len(model_list), 5)
        
        index_list: list = []
        
        for name in model_list:
            index_list.append(name)
            index_list.append(name)
            
        for row, model in enumerate(model_list):
                values[2 * row, 1: ] = confusion_matrix(predictions[predictions['train'] == 1]['y_true'], 
                                                        predictions[predictions['train'] == 1][model]
                                                        ).ravel()
                
                values[2 * row + 1, 1: ] = confusion_matrix(predictions[predictions['train'] == 0]['y_true'], 
                                                            predictions[predictions['train'] == 0][model]
                                                            ).ravel()
                
                values[2 * row, 0] = 1
                values[2 * row + 1, 0] = 0
            
                                    
        confusion_matrices_df: pd.DataFrame = pd.DataFrame(data = values, 
                                                           index = index_list, 
                                                           columns = ['train', 'TN', 'FP', 'FN', 'TP']
                                                          )
            
        confusion_matrices_df = confusion_matrices_df.reset_index()
        
        confusion_matrices_df = confusion_matrices_df.rename(columns = {'index': 'model'})            
                                    
        return confusion_matrices_df
    
    def plot_Kaggle_results(Kaggle_results: pd.DataFrame) -> None:
        '''
        The function plots bar graph for the Kaggle competition accuracies.

        Parameters:
        -----------
            Kaggle_results
                pd.DataFrame, data set of training, testing, and
                Kaggle submission accuracies.
        
        Returns:
        --------
            None
                It plots the visual.
        '''

        fig, ax = plt.subplots(figsize = (25, 8))

        # Plot Title
        fig.suptitle('Kaggle Submission Accuracies', 
                     fontsize = 20
                    )

        # Generate the bar graph
        ax = sns.barplot(data = Kaggle_results,
                         x = 'model', 
                         y = 'accuracy',
                         hue = 'train',
                         hue_order = [1, 0, 2],
                         width = 0.85,
                         palette = 'Paired',
                         ax = ax
                        )

        # Legend's position
        ax.legend(bbox_to_anchor = (1.02, 0.5))

        # Legend labels
        leg_labels: list = ['train', 'test', 'Kaggle\nsubmission']
        for text, label in zip(ax.axes.get_legend().texts, leg_labels):
            text.set_text(label)

        # Spines
        ax.spines[['right', 'top']].set_visible(False)

        # Display values for each bar.
        for container in ax.containers:
            ax.bar_label(container, fmt=f'%.4f')
        
        plt.show()

    def plot_accuracies(confusion_matrices: pd.DataFrame) -> None:
        '''
        The function plots bar graph of training and testing accuracies.

        Parameters:
        -----------
            confusion_matrices
                pd.DataFrame, confusion matrices for each model
                to be evaluated
        
        Returns:
        --------
            None
                It plots the visual.
        '''

        fig, ax = plt.subplots(figsize = (17, 8))

        # Plot Title
        fig.suptitle('Model Accuracies on Titanic Passenger Survival Prediction', 
                    fontsize = 20
                    )

        # The bar graph
        ax = sns.barplot(data = confusion_matrices,
                        x = 'model', 
                        y = 'accuracy',
                        hue = 'train',
                        hue_order = [1, 0],
                        width = 0.8,
                        palette = 'ocean',
                        ax = ax
                        )

        # Legend Position
        ax.legend(bbox_to_anchor = (1.02, 0.5))

        # Legend Labels
        leg_labels: list = ['train', 'test']
        for text, label in zip(ax.axes.get_legend().texts, leg_labels):
            text.set_text(label)

        # Axis' Spines
        ax.spines[['right', 'top']].set_visible(False)

        # Display values on top of each bar
        for container in ax.containers:
            ax.bar_label(container, fmt=f'%.4f')

        plt.show()

    def plot_titanic_metrics(metrics: pd.DataFrame) -> None:
        '''
        The function plots bar graph of the following metrics:
        - True Positive Rate (Sensitivity)
        - Negative Predictive Value
        - True Negative Rate (Specifity)
        - Positive Predictive Value
        - Balanced Accuracy
        - F-score


        Parameters:
        -----------
            metrics
                pd.DataFrame, the data frame with each metric
                as column.
        
        Returns:
        --------
            None
                It plots the visual.
        '''
            
        fig, ax = plt.subplots(2,
                               3,
                               sharey = True,
                               figsize = (45, 13)
                              )

        # Plot title
        fig.suptitle('Six Evaluation Metrics For Binary Classifiers', 
                    x = 0.5, 
                    y = 1.1,
                    fontsize = 40
                    )

        # Spacing between plots
        plt.subplots_adjust(hspace = 0.3,
                            wspace = 0.3
                        )

        # FN-minimizing metrics
        plt.text(-21.5, 2.7, 
                'Minimizing False Negatives (FN)', 
                fontsize = 20, 
                bbox = dict(facecolor = 'green', 
                            alpha = 0.5,
                            edgecolor = 'green', 
                            boxstyle = 'round, pad=0.5'
                            )
                )

        # FP-minimizing metrics
        plt.text(-9.5, 2.7, 
                'Minimizing False Positives (FP)', 
                fontsize = 20, 
                bbox = dict(facecolor = 'orange', 
                            alpha = 0.5,
                            edgecolor = 'orange', 
                            boxstyle = 'round, pad=0.5'
                            )
                )

        # Balanced metrics
        plt.text(2.5, 2.7, 
                'Minimizing both FN and FP', 
                fontsize=20, 
                bbox = dict(facecolor = 'lightskyblue', 
                            alpha = 0.5, 
                            edgecolor = 'lightskyblue', 
                            boxstyle = 'round, pad=0.5'
                            )
                )
        
        # Unpacking two dictionaries for the subplot title
        # and subplot axis coordinates.
        _, title_dict, axis_dict = titanic_evaluation_helpers.metrics_init()

        # Dictionary of palettes
        palette_dict: dict =  {0: 'summer', 
                               1: 'autumn',
                               2: 'winter'}


        # Plotting the graphs
        for metric_name, index in axis_dict.items():
            
                # Subplot title
                ax[index].set_title(title_dict[metric_name], 
                                    fontsize = 15, 
                                    bbox = dict(facecolor = 'azure', 
                                                alpha = 0.5, 
                                                edgecolor = 'azure', 
                                                boxstyle = 'round, pad=0.5'
                                            )
                                    )
                
                # The visual
                ax[index] = sns.barplot(data = metrics, 
                                        x = 'model', 
                                        y = metric_name, 
                                        hue = 'train',
                                        hue_order = [1, 0],
                                        ax = ax[index],
                                        width = 0.85,
                                        palette = palette_dict[index[1]]
                                        )
                    
        for axis in ax.reshape(-1):
                       
            # Set the axis spines
            axis.spines[['right', 'top']].set_visible(False)
            
            # Legend position
            axis.legend(bbox_to_anchor = (1.02, 0.5))
            
            # Display data on top of the bars
            for container in axis.containers:
                axis.bar_label(container, 
                            fmt='%.4f', 
                            padding = 2, 
                            fontsize = 8
                            )

            # Legend labels
            leg_labels: list = ['train', 'test']
            for text, label in zip(axis.axes.get_legend().texts, leg_labels):
                text.set_text(label)

        plt.show()

    def Jaccard_index(intervals: list) -> float:
        '''
        The function computes the Jaccard index between
        two given intervals

        Parameters:
        -----------
            intervals
                List, list of intervals
        
        Returns:
        --------
            Float
                Jaccard index
        '''

        # Sort the intervals
        lower_interval, upper_interval = sorted(intervals, 
                                                key = lambda interval: interval[0]
                                                )
        
        # Non-intersecting sets
        if lower_interval[1] < upper_interval[0]:
            return 0
        
        # Intersecting sets
        numerator: float = min(upper_interval[1], lower_interval[1]) - max(upper_interval[0], lower_interval[0])

        denominator: float = (upper_interval[1] - upper_interval[0]) + (lower_interval[1] - lower_interval[0]) - numerator

        return numerator / denominator