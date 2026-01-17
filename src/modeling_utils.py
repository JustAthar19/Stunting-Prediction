'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling.
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix,precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, make_scorer, average_precision_score
from lightgbm import LGBMClassifier
import time

from sklearn.model_selection import KFold

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

class ColumnDropper(BaseEstimator, TransformerMixin):
    '''
    A transformer class to drop specified columns from a DataFrame.

    Attributes:
        to_drop (list): A list of column names to be dropped.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by dropping specified columns.
    '''

    def __init__(self, to_drop):
        '''
        Initialize the ColumnDropper transformer.

        Args:
            to_drop (list): A list of column names to be dropped.
        '''
        self.to_drop = to_drop

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        This method does nothing and is only provided to comply with the Scikit-learn API.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by dropping specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after dropping specified columns.
        '''
        # Certify that only present columns will be dropped.
        self.to_drop = [col for col in self.to_drop if col in X.columns]
        
        # Drop the specified columns.
        return X.drop(columns=self.to_drop)
    

class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for one-hot encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be one-hot encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by one-hot encoding specified columns.
    '''

    def __init__(self, to_encode):
        '''
        Initialize the OneHotFeatureEncoder transformer.

        Args:
            to_encode (list): A list of column names to be one-hot encoded.
        '''
        self.to_encode = to_encode
        self.encoder = OneHotEncoder(drop='first',
                                     sparse_output=False,
                                     dtype=np.int8,
                                     handle_unknown='ignore',
                                     feature_name_combiner='concat')

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[self.to_encode])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by one-hot encoding specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after one-hot encoding specified columns.
        '''
        # One-hot encode the columns.
        X_one_hot = self.encoder.transform(X[self.to_encode])

        # Create a dataframe for the one-hot encoded data.
        one_hot_df = pd.DataFrame(X_one_hot,
                                  columns=self.encoder.get_feature_names_out(self.to_encode))

        # Reset for mapping and concatenate constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)

        return pd.concat([X_reset.drop(columns=self.to_encode), one_hot_df], axis=1)
    

class StandardFeatureScaler(BaseEstimator, TransformerMixin):
    '''
    A transformer class for standard scaling specified numerical features and retaining feature names.

    Attributes:
        to_scale (list): A list of column names to be scaled.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by standard scaling specified columns and retaining feature names.
    '''
    def __init__(self, to_scale):
        '''
        Initialize the StandardFeatureScaler transformer.

        Args:
            to_scale (list): A list of column names to be scaled.
        '''
        self.to_scale = to_scale
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.scaler.fit(X[self.to_scale])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by standard scaling specified columns and retaining feature names.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after standard scaling specified columns and retaining feature names.
        '''
        # Scale the columns.
        X_scaled = self.scaler.transform(X[self.to_scale])
        
        # Create a dataframe for the scaled data.
        scaled_df = pd.DataFrame(X_scaled,
                                 columns=self.scaler.get_feature_names_out(self.to_scale))
        
        # Reset for mapping and concatenated constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=self.to_scale), scaled_df], axis=1)
    
    
    
class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for ordinal encoding specified categorical features and retaining feature names.

    Attributes:
        to_encode (dict): A dictionary where keys are column names and values are lists representing the desired category orders.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by ordinal encoding specified columns and retaining feature names.
    '''
    def __init__(self, to_encode):
        '''
        Initialize the OrdinalFeatureEncoder transformer.

        Args:
            to_encode (dict): A dictionary where keys are column names and values are lists representing the desired category orders.
        '''
        self.to_encode = to_encode
        self.encoder = OrdinalEncoder(dtype=np.int8, 
                                      categories=[to_encode[col] for col in to_encode])

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[list(self.to_encode.keys())])
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by ordinal encoding specified columns and retaining feature names.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after ordinal encoding specified columns and retaining feature names.
        '''
        # Ordinal encode the columns.
        X_ordinal = self.encoder.transform(X[list(self.to_encode.keys())])
        
        # Create a dataframe for the ordinal encoded data.
        ordinal_encoded_df = pd.DataFrame(X_ordinal,
                                          columns=self.encoder.get_feature_names_out(list(self.to_encode.keys())))
        
        # Reset for mapping and concatenated constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)
        
        return pd.concat([X_reset.drop(columns=list(self.to_encode.keys())), ordinal_encoded_df], axis=1)
    

class TargetFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for target encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be target encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by target encoding specified columns.
    '''

    def __init__(self, to_encode):
        '''
        Initialize the TargetFeatureEncoder transformer.

        Args:
            to_encode (list): A list of column names to be target encoded.
        '''
        self.to_encode = to_encode
        self.encoder = TargetEncoder()

    def fit(self, X, y):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns an instance of self.
        '''
        self.encoder.fit(X[self.to_encode], y)
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by target encoding specified columns.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after target encoding specified columns.
        '''
        # Target encode the columns.
        X_target = self.encoder.transform(X[self.to_encode])

        # Create a dataframe for the target encoded data.
        target_df = pd.DataFrame(X_target,
                                 columns=self.encoder.get_feature_names_out(self.to_encode))

        # Reset for mapping and concatenate constructing a final dataframe of features.
        X_reset = X.reset_index(drop=True)

        return pd.concat([X_reset.drop(columns=self.to_encode), target_df], axis=1)
    

class FeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    A transformer class for performing feature engineering on churn-related data.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by engineering churn-related features.
    '''

    def fit(self, X, y=None):
        '''
        Fit the transformer to the data.

        This method does nothing and is only provided to comply with the Scikit-learn API.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like, default=None): Target labels. Ignored.

        Returns:
            self: Returns an instance of self.
        '''
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by engineering churn-related features.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after engineering churn-related features.
        '''
        X_copy = X.copy()
        discrete_features = ['customer_age', 
                            'dependent_count', 
                            'months_on_book', 
                            'total_relationship_count', 
                            'months_inactive_12_mon', 
                            'contacts_count_12_mon',
                            'total_trans_ct']
        continuous_features = ['credit_limit', 
                               'total_revolving_bal', 
                               'total_amt_chng_q4_q1', 
                               'total_trans_amt',  
                               'total_ct_chng_q4_q1']
        
        X_copy[discrete_features] = X_copy[discrete_features].astype('int32')
        X_copy[continuous_features] = X_copy[continuous_features].astype('float32')
        
        # Construct ratio features.
        X_copy['products_per_dependent'] = (X_copy['total_relationship_count'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_amt_per_dependent'] = (X_copy['total_trans_amt'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_ct_per_dependent'] = (X_copy['total_trans_ct'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_amt_per_products'] = (X_copy['total_trans_amt'] / X_copy['total_relationship_count']).astype('float32')
        X_copy['trans_ct_per_products'] = (X_copy['total_trans_ct'] / X_copy['total_relationship_count']).astype('float32')
        X_copy['avg_trans_amt'] = (X_copy['total_trans_amt'] / X_copy['total_trans_ct']).astype('float32')
        X_copy['credit_util_rate'] = (X_copy['total_revolving_bal'] / X_copy['credit_limit']).astype('float32')
        X_copy['proportion_inactive_months'] = (X_copy['months_inactive_12_mon'] / X_copy['months_on_book']).astype('float32')
        X_copy['products_per_tenure'] = (X_copy['total_relationship_count'] / X_copy['months_on_book']).astype('float32')
        X_copy['products_per_contacts'] = (X_copy['total_relationship_count'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['dependents_per_contacts'] = (X_copy['dependent_count'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['trans_ct_per_contacts'] = (X_copy['total_trans_ct'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['products_per_inactivity'] = (X_copy['total_relationship_count'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['dependents_per_inactivity'] = (X_copy['dependent_count'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['trans_ct_per_inactivity'] = (X_copy['total_trans_ct'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['trans_amt_per_credit_limit'] = (X_copy['total_trans_amt'] / X_copy['credit_limit']).astype('float32')
        X_copy['age_per_tenure'] = (X_copy['customer_age'] / X_copy['months_on_book']).astype('float32')
        X_copy['trans_ct_per_tenure'] = (X_copy['total_trans_ct'] / X_copy['months_on_book']).astype('float32')
        X_copy['trans_amt_per_tenure'] = (X_copy['total_trans_amt'] / X_copy['months_on_book']).astype('float32')
        
        # Replace division by zero values with zero. It will have this 'zero' meaning.
        X_copy = X_copy.replace({np.inf: 0, 
                                 np.nan: 0})
        
        # Construct sum features.
        X_copy['total_spending'] = (X_copy['total_trans_amt'] + X_copy['total_revolving_bal']).astype('int32')
        X_copy['inactivity_contacts'] = (X_copy['contacts_count_12_mon'] + X_copy['months_inactive_12_mon']).astype('int32')
        
        # Map to ordinal education level and income to sum them.
        education_mapping = {
            'Uneducated': 0,
            'High School': 1,
            'College': 2,
            'Graduate': 3,
            'Post-Graduate': 4,
            'Doctorate': 5,
            'Unknown': 0
        }
        
        income_mapping = {
            'Less than $40K': 0,
            '$40K - $60K': 1,
            '$60K - $80K': 2,
            '$80K - $120K': 3,
            '$120K +': 4,
            'Unknown': 0
        }
        
        # Construct interaction between education and income based feature.
        X_copy['education_numeric'] = X_copy['education_level'].map(education_mapping).astype('int32')
        X_copy['income_numeric'] = X_copy['income_category'].map(income_mapping).astype('int32')
        X_copy['education_income_levels'] = (X_copy['education_numeric'] + X_copy['income_numeric']).astype('int32')
        X_copy = X_copy.drop(columns=['education_numeric', 'income_numeric'])
        
        return X_copy
    
    
class RecursiveFeatureEliminator(BaseEstimator, TransformerMixin):
    '''
    A transformer class for selecting features based on the Recursive Feature Elimination (RFE) technique.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by recursively selecting the features with highest feature 
        importances until a final desired number of features is obtained through stratified k-fold cross validation.
    '''

    def __init__(self, estimator=LGBMClassifier(), scoring='roc_auc', n_folds=5):
        '''
        Initialize the Recursive Feature Elimination (RFE) transformer.
        
        Args:
            estimator (object, default=LGBMCLassifier): The model to obtain feature importances.
            n_folds (int, default=5): The number of folds for stratified k-fold cross validation.
            scoring (object, default='roc_auc'): The scoring for cross-validation.
            
        '''
        stratified_kfold = StratifiedKFold(n_splits=n_folds, 
                                           shuffle=True, 
                                           random_state=42)
        self.rfe = RFECV(estimator=estimator, 
                         cv=stratified_kfold,
                         scoring=scoring)

    def fit(self, X, y):
        '''
        Fit the transformer to the data.

        Args:
            X (pandas.DataFrame): Input features.
            y (array-like): Target labels.

        Returns:
            self: Returns an instance of self.
        '''
        self.rfe.fit(X, y)
        
        return self

    def transform(self, X):
        '''
        Transform the input DataFrame by recursively selecting the features with highest feature 
        importances.

        Args:
            X (pandas.DataFrame): Input features.

        Returns:
            pandas.DataFrame: Transformed DataFrame after recursively selecting the features with highest feature 
            importances.
        '''
        # Recursively select the features with highest feature importances.
        X_selected = self.rfe.transform(X)

        # Create a dataframe for the final selected features.
        selected_df = pd.DataFrame(X_selected,
                                  columns=self.rfe.get_feature_names_out())

        return selected_df


def classification_kfold_cv(models, X_train, y_train, n_folds=5):
    '''
    Evaluate multiple machine learning models using k-fold cross-validation.

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Stratified KFold in order to maintain the target proportion on each validation fold - dealing with imbalanced target.
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Get the model object from the key with his name.
            model_instance = models[model]

            # Measure training time.
            start_time = time.time()
            
            # Fit the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict_proba(X_train.values)
            # y_train_pred = model_instance.decision_function(X_train.values)
            # train_score = roc_auc_score(y_train, y_train_pred, multi_class="ovo")
            train_score = average_precision_score(y_train, y_train_pred)

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train.values, y_train, scoring='average_precision', cv=stratified_kfold)
            avg_val_score = val_scores.mean()
            val_score_std = val_scores.std()

            # Add the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Print the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Convert scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['model', 'avg_val_score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['model', 'train_score'])
        eval_df = val_df.merge(train_df, on='model')

        # Sort the dataframe by the best ROC-AUC score.
        eval_df  = eval_df.sort_values(['avg_val_score'], ascending=False).reset_index(drop=True)
        
        return eval_df
    
    except Exception as e:
        raise(CustomException(e, sys))


def plot_classification_kfold_cv(eval_df, figsize=(20, 7), bar_width=0.35, title_size=15,
                             title_pad=30, label_size=11, labelpad=20, legend_x=0.08, legend_y=1.08):
    '''
    Plot classification performance using k-fold cross-validation.

    Parameters:
        eval_df (DataFrame): DataFrame containing evaluation metrics for different models.
        figsize (tuple, optional): Figure size (width, height). Defaults to (20, 7).
        bar_width (float, optional): Width of bars in the plot. Defaults to 0.35.
        title_size (int, optional): Font size of the plot title. Defaults to 15.
        title_pad (int, optional): Padding of the plot title. Defaults to 30.
        label_size (int, optional): Font size of axis labels. Defaults to 11.
        labelpad (int, optional): Padding of axis labels. Defaults to 20.
        legend_x (float, optional): x-coordinate of legend position. Defaults to 0.08.
        legend_y (float, optional): y-coordinate of legend position. Defaults to 1.08.

    Raises:
        CustomException: Raised if an unexpected error occurs.

    Returns:
        None
    '''
    try:
        # Plot each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(eval_df['model']))
        y = np.arange(len(eval_df['train_score']))

        val_bars = ax.bar(x - bar_width/2, eval_df['avg_val_score'], bar_width, label='Val score', color='#023047')
        train_bars = ax.bar(x + bar_width/2, eval_df['train_score'], bar_width, label='Train score', color='#0077b6')

        ax.set_xlabel('Model', labelpad=labelpad, fontsize=label_size, loc='left')
        ax.set_ylabel('Average_Precision_Score', labelpad=labelpad, fontsize=label_size, loc='top')
        ax.set_title("Models' performances", fontweight='bold', fontsize=title_size, pad=title_pad, loc='left')
        ax.set_xticks(x, eval_df['model'], rotation=0, fontsize=10.8)
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.tick_params(axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)
        
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')

        # Define handles and labels for the legend with adjusted sizes
        handles = [plt.Rectangle((0,0), 0.1, 0.1, fc='#023047', edgecolor = 'none'),
                plt.Rectangle((0,0), 0.1, 0.1, fc='#0077b6', edgecolor = 'none')]
        labels = ['Val score', 'Train score']
            
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(legend_x, legend_y), frameon=False, ncol=2, fontsize=10)
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classifier(y_true, y_pred, probas):
    '''
    Evaluate the performance of a binary classifier and visualize the results.

    This function calculates and displays various evaluation metrics for a binary classifier,
    including the classification report, confusion matrix, ROC curve and AUC, PR curve and AUC,
    brier score, gini and ks.

    Args:
    - y_true (pd.series): True binary labels.
    - y_pred (pd.series): Predicted binary labels.
    - probas (pd.series): Predicted probabilities of positive class.

    Returns:
    - model_metrics (pd.DataFrame): A dataframe containing the classification metrics for the passed set.

    Raises:
    - CustomException: If an error occurs during evaluation.
    '''

    try:
        # Print classification report and calculate its metrics to include in the final metrics df.
        print(classification_report(y_true, y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average="micro")
        f1 = f1_score(y_true, y_pred, average="micro")
        maP = average_precision_score(y_true, probas)

        # Calculate and print brier score, gini and ks.
        # brier_score = brier_score_loss(y_true, probas)
        # print(f'Brier Score: {round(brier_score, 2)}')
        
        # fpr, tpr, thresholds = roc_curve(y_true, probas)
        # roc_auc = roc_auc_score(y_true, probas, multi_class='ovo')
        # gini = 2 * roc_auc - 1
        # print(f'Gini: {round(gini, 2)}')
        
        # Confusion matrix.
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
        
        # Construct a DataFrame with metrics for passed sets.
        model_metrics = pd.DataFrame({
                                    'Metric': ['Accuracy',
                                               'Precision',
                                               'Recall',
                                               'F1-Score',
                                               "Mean-Average-Precision-Score"
                                               ],
                                    'Value': [accuracy, 
                                              precision, 
                                              recall,
                                              f1,
                                              maP
                                              ],
                                    })
        
        return model_metrics

    except Exception as e:
        raise CustomException(e, sys)

    

def plot_feature_importances(model, data):
    '''
    Plot feature importances of a given machine learning model.

    This function takes a trained machine learning model and the corresponding dataset used for training, and plots the
    feature importances of the model's attributes. Feature importances are sorted in descending order for visualization.

    Args:
        model (object): The trained machine learning model with a feature_importances_ attribute.
        data (DataFrame): The dataset containing the features used for training the model.

    Returns:
        None (displays feature importances).

    Raises:
        CustomException: If an error occurs while plotting feature importances.

    '''
    
    try:
        # Get feature importances
        importances = model.feature_importances_
        feature_names = data.columns 


        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(12, 3))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), sorted_importances, tick_label=sorted_feature_names, color='#023047')
        plt.xticks(rotation=90)
        plt.show()

    except Exception as e:
        raise CustomException(e, sys)
    
