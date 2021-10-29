""" Test script for churn_library.py

This script tests functions defined in churn_library.py. The result will be
logged in `logs/churn_library.log`

Authur: xxxxxxxx
Created: 2021/10/28
"""

import os
import logging
import pandas as pd
import numpy as np
from churn_library import (
    encoder_helper,
    import_data,
    perform_eda,
    make_churn_col,
    perform_feature_engineering,
    train_models
)
from constants import CATEGORY_LST, KEEP_COLS, RESPONSE

# Set configuration for logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Set an envitonment variable to avoid error
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def test_import(import_data):
    '''
    Test data import - This example is completed for you to assist with the 
    other test functions.
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0, "Number of rows should be more than zero"
        assert df.shape[1] > 0, "Number of columns should be more than zero"
    except AssertionError as err:
        logging.error(str(err))
        raise err


def test_eda(perform_eda):
    '''
    Test perform eda function.
    '''
    df = import_data("./data/bank_data.csv")
    df = make_churn_col(df)
    try:
        perform_eda(df)
    except BaseException:
        logging.error("perform_eda function caseses error")

    try:
        pngs = [
            "churn_distribution.png",
            "customer_age_distribution.png",
            "marital_status_distribution.png",
            "total_transaction_distribution.png",
            "heatmap.png"
        ]
        for png in pngs:
            pth = os.path.join("images", "eda", png)
            assert os.path.exists(pth), f"{pth} doesn't exist"
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(str(err))
        raise err


def test_encoder_helper(encoder_helper):
    '''
    Test encoder helper function.
    '''
    df = import_data("./data/bank_data.csv")
    df = make_churn_col(df)

    try:
        df = encoder_helper(df, CATEGORY_LST, RESPONSE)

        # Check new columns are added for all category column with certain col
        # names
        for cat in CATEGORY_LST:
            new_col = f"{cat}_{RESPONSE}"
            assert new_col in df.columns, \
                   f"The function didn't prepare new cols for {cat}"
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(str(err))


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering function.
    '''
    df = import_data("./data/bank_data.csv")
    df = make_churn_col(df)
    # Test the function is executed without an error
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df, RESPONSE)
    except BaseException:
        logging.error("perform_feature_engineering function causes an error")

    # Test four outputs are expected instances
    try:
        assert isinstance(X_train, pd.DataFrame), \
               "X_train is not an instance of pandas.DataFrame"
        assert isinstance(X_test, pd.DataFrame), \
                "X_test is not an instance of pandas.DataFrame"
        assert isinstance(y_train, pd.Series), \
                "y_train is not an instance pandas.Series"
        assert isinstance(y_test, pd.Series), \
                "y_test is not an instance of pandas.Series"
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(str(err))


def test_train_models(train_models):
    '''
    Test train_models function.
    '''
    df = import_data("./data/bank_data.csv")
    df = make_churn_col(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, RESPONSE)

    # Test the function is executed without an error
    try:
        train_models(X_train, X_test, y_train, y_test)
    except BaseException as err:
        logging.error(f"train_models function causes an error: {str(err)}")
    try:
        # Test all images exist
        images = [
            "feature_importances.png",
            "logistic_results.png",
            "rf_results.png",
            "roc_curve_result.png"
        ]
        for image in images:
            image_pth = "images/results/" + image
            assert os.path.exists(image_pth), f"{image} doesn't exist"

        # Test all models exist
        models = ["logistic_model.pkl", "rfc_model.pkl"]
        for model in models:
            model_pth = "models/" + model
            assert os.path.exists(model_pth), f"{model} doesn't exist"
        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error(str(err))


if __name__ == "__main__":
    test_import(import_data)
    test_eda(perform_eda)
    test_encoder_helper(encoder_helper)
    test_perform_feature_engineering(perform_feature_engineering)
    test_train_models(train_models)
