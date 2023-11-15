# train_model.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import sklearn.preprocessing as pre
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset

data = pd.read_csv("survey_results_public.csv")

# Select features and target variable
select = ['Age', 'Employment', 'RemoteWork', 'EdLevel', 'LearnCode', 'YearsCodePro',
          'Industry', 'DevType', 'OrgSize', 'ConvertedCompYearly', 'Country', 'LanguageHaveWorkedWith']
data_df = data[select]

# Drop rows with missing salary values
df_no_missing = data_df.dropna(subset=['ConvertedCompYearly'])

# Log-transform the target variable
y = np.log(df_no_missing['ConvertedCompYearly']).copy()

# Drop the target variable from the feature set
filtered_df = df_no_missing.drop(columns=['ConvertedCompYearly'])

# Preprocess the data
def pre_process(filtered_df):
    learn_code=filtered_df['LearnCode'].str.get_dummies(sep=';')
    languages = filtered_df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')
    Employ_situation = filtered_df['Employment'].str.get_dummies(sep=';')
    Merge_data=filtered_df.merge(Employ_situation,left_index=True, right_index=True).merge(languages,left_index=True, right_index=True).merge(learn_code,left_index=True, right_index=True)
    New_data=Merge_data.drop(columns=["Employment", "LearnCode", "LanguageHaveWorkedWith"])
    table = New_data.copy()
    wanted_cols = table.select_dtypes(include=['object']).columns
    encoders = [pre.LabelEncoder()]*len(wanted_cols)
    colname2encoder = dict(zip(wanted_cols, encoders))
   # Transform columns
    for colname in colname2encoder.keys():
        table[colname] = colname2encoder[colname].fit_transform(table[colname])
        
   # Create an imputer object and specify the imputation strategy
    imputer = SimpleImputer(strategy="most_frequent")

   # Fit the imputer to the data
    imputer.fit(table)

   # Transform the data to impute missing values
    table_transformed = imputer.transform(table)

   # Convert the result back to a DataFrame
    table_transformed = pd.DataFrame(table_transformed, columns=table.columns)
    return table_transformed

# Split the data into training and testing sets
X = pre_process(filtered_df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Save the trained model and preprocessing parameters
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(X.columns, 'features_columns.joblib')
