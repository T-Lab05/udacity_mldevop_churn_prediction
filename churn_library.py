""" Library for EDA, model building, and evaluation

This script provides the following functions:
- preserve images for EDA `images/eda/*.png`
- model building (logistic regression and random forest classifier)
 `models/*.pkl`
- preserve images/report for model metrics `images/results/*.png`

Authur: xxxxxxxxx
Created: 2021/10/28
"""

# Import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import CATEGORY_LST, KEEP_COLS, RESPONSE
sns.set()


def import_data(pth):
    '''Returns dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def make_churn_col(df):
    '''Prepare 'Churn' column from 'Attrition_Flag' column.
    
    input:
            df: pandas dataframe
    output:
            df: pandas dataframe
    '''
    # Prepare Churn column
    def func(val): return 0 if val == "Existing Customer" else 1
    df['Churn'] = df['Attrition_Flag'].apply(func)
    return df


def perform_eda(df):
    '''Perform eda on df and save figures to images folder
    
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Distribution of Churn
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(f"images/eda/churn_distribution.png")

    # Distribution of Customer age
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(f"images/eda/customer_age_distribution.png")

    # Distribution of marital status
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(f"images/eda/marital_status_distribution.png")

    # Distribution of Total transaction
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig(f"images/eda/total_transaction_distribution.png")

    # Heatmap of correlation matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(f"images/eda/heatmap.png")


def encoder_helper(df, category_lst, response):
    '''
    Helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from 
    the notebook.

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        group_mean_dict = df.groupby(cat)[response].mean().to_dict()
        new_col = f"{cat}_{response}"
        df[new_col] = df[cat].map(group_mean_dict)
    return df


def perform_feature_engineering(df, response):
    ''' A function for feature engineering.

    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Encoding category columns
    df = encoder_helper(df, CATEGORY_LST, RESPONSE)

    # Select certains columns as features
    X = df[KEEP_COLS]
    y = df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    Produces classification report for training and testing results and stores
    report as image in images folder.

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    model_labels = ["Logitic Regression", "Random Forest"]
    outputs = ["logistic_results.png", "rf_results.png"]
    y_tests = [y_test_preds_lr, y_test_preds_rf]
    y_trains = [y_train_preds_lr, y_train_preds_rf]

    for ml, output, y_test_preds, y_train_preds in zip(
            model_labels, outputs, y_tests, y_trains):
        output = "images/results/" + output
        plt.figure()
        plt.rc('figure', figsize=(8, 5))
        plt.text(0.01, 1.25, 
                str(f'{ml} Train'), 
                {'fontsize': 10}, 
                fontproperties='monospace')
        plt.text(0.01, 0.05, 
                str(classification_report(y_test, y_test_preds)),
                {'fontsize': 10}, 
                fontproperties='monospace')
        plt.text(0.01, 0.6, 
                str(f'{ml} Test'), 
                {'fontsize': 10}, 
                fontproperties='monospace')
        plt.text(0.01, 0.7, 
                str(classification_report(y_train, y_train_preds)),
                {'fontsize': 10}, 
                fontproperties='monospace')
        plt.axis('off')
        plt.savefig(f"{output}")


def feature_importance_plot(model, X_data, output_pth):
    '''Creates and stores the feature importances in pth.
    
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''Train, store model results: images + scores, and store models
    
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Random forest model with parameters by grid search
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    best_rf = cv_rfc.best_estimator_
    y_train_preds_rf = best_rf.predict(X_train)
    y_test_preds_rf = best_rf.predict(X_test)
    joblib.dump(best_rf, "models/rfc_model.pkl")

    # Logistic regression model
    lrc = LogisticRegression()
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    joblib.dump(lrc, "models/logistic_model.pkl")

    # Plot feature importance
    importance_image_pth = "images/results/feature_importances.png"
    feature_importance_plot(best_rf, X_train, importance_image_pth)

    # Plot ROC curve
    _, ax = plt.subplots(figsize=(15, 8))
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(best_rf, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig("images/results/roc_curve_result.png")

    # Output classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
