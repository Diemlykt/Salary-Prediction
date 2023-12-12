# train_model.py
from multiprocessing import freeze_support
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as pre
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from xgboost import XGBRegressor

# Load the dataset

data = pd.read_csv("survey_results_public.csv")


# Select features and target variable
select = ['Age', 'Employment', 'RemoteWork', 'EdLevel', 'LearnCode', 'YearsCodePro','Industry', 'DevType', 'OrgSize',
        'ConvertedCompYearly', 'Country', 'LanguageHaveWorkedWith','DatabaseHaveWorkedWith','PlatformHaveWorkedWith',
        'WebframeHaveWorkedWith']
data_df = data[select]

# Drop rows with missing salary values
df_no_missing = data_df.dropna(subset=['ConvertedCompYearly'])

# Log-transform the target variable
y = np.log(df_no_missing['ConvertedCompYearly']).copy()

# Drop the target variable from the feature set
filtered_df = df_no_missing.drop(columns=['ConvertedCompYearly'])

# Preprocess the data

# Function to fit and transform the data using encoders
encoders = {}
def fit_transform_column(column):
    encoder = {val: idx for idx, val in enumerate(column.unique())}
    encoders[column.name] = encoder
    return column.map(encoder)
def pre_process(filtered_df):
    learn_code=filtered_df['LearnCode'].str.get_dummies(sep=';')
    languages = filtered_df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')
    Employ_situation = filtered_df['Employment'].str.get_dummies(sep=';')
    Database = filtered_df['DatabaseHaveWorkedWith'].str.get_dummies(sep=';')
    Platform = filtered_df['PlatformHaveWorkedWith'].str.get_dummies(sep=';')
    Webframe=filtered_df['WebframeHaveWorkedWith'].str.get_dummies(sep=';')
    Merge_data = (
    filtered_df
    .merge(Employ_situation, left_index=True, right_index=True)
    .merge(languages, left_index=True, right_index=True)
    .merge(learn_code, left_index=True, right_index=True)
    .merge(Database, left_index=True, right_index=True)
    .merge(Platform, left_index=True, right_index=True)
    .merge(Webframe, left_index=True, right_index=True)
)
    columns_to_drop = ["Employment", "LearnCode", "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith", "PlatformHaveWorkedWith", 'WebframeHaveWorkedWith']
    New_data = Merge_data.drop(columns=columns_to_drop)
    table = New_data.copy()
    wanted_cols = table.select_dtypes(include=['object']).columns

    for col in wanted_cols:
        # Use the actual column name from wanted_cols
        column_name = col
        
        # Assuming you have the fit_transform_column function defined
        table[column_name] = fit_transform_column(table[column_name])

   # Create an imputer object and specify the imputation strategy
    imputer = SimpleImputer(strategy="most_frequent")

   # Fit the imputer to the data
    imputer.fit(table)

   # Transform the data to impute missing values
    table_transformed = imputer.transform(table)

   # Convert the result back to a DataFrame
    table_transformed = pd.DataFrame(table_transformed, columns=table.columns)
    return table_transformed
if __name__ == '__main__':
    freeze_support()
    # Split the data into training and testing sets
    X = pre_process(filtered_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Train the Linear Regression model
    xgb_model = XGBRegressor(subsample=0.7, n_estimators=400, min_child_weight=5, max_depth=5, learning_rate=0.05, colsample_bytree=0.5)
    xgb_model.fit(X_train, y_train)

    # Save the trained model and preprocessing parameters
    joblib.dump(xgb_model, 'xgb_model.joblib')
    joblib.dump(X.columns, 'features_columns.joblib')
    joblib.dump(encoders, 'encoders.joblib')
