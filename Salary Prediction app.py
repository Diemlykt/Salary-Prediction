import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import gdown
import subprocess

# URL of the file on Google Drive
url = 'https://drive.google.com/uc?id=1U87DdXraW6xwZwiZ87TCCfpzrrvwkB2G'

# Output file path
output = 'stack-overflow-developer-survey-2023.zip'  # Assuming it's a zip file

# Download the file
gdown.download(url, output, quiet=False)

# Unzip the file
subprocess.run(["unzip", "stack-overflow-developer-survey-2023.zip"])


def main():
    st.title("2023 Developer Salary Web App")
    st.sidebar.title("2023 Developer Salary Web App")
    st.markdown("Do you want to see salary analysis or predict salary?")
    st.sidebar.markdown("Do you want to see salary analysis or predict salary?")


    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv("stack-overflow-developer-survey-2023.csv")
        return data
