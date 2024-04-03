import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

data = pd.read_csv(r'C:\Users\asuss\Downloads\cleaned_data.csv')

data['Full Time Employees'] = data['Full Time Employees'].str.replace(',', '').astype(float)


label_encoder = LabelEncoder()
data['ESG Risk Level'] = label_encoder.fit_transform(data['ESG Risk Level'])


numerical_columns = ['Full Time Employees', 'Total ESG Risk score', 'Environment Risk Score',
                     'Governance Risk Score', 'Social Risk Score', 'Controversy Score',
                     'ESG Risk Percentile']
categorical_columns = ['Sector', 'Controversy Level']

numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

X = preprocessor.fit_transform(data.drop(['ESG Risk Level', 'Name'], axis=1))
y = data['ESG Risk Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    LogisticRegression(),
    DecisionTreeClassifier()
]

for model in models:
    model_name = model.__class__.__name__
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"{model_name} accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def sidebar_filters():
    st.sidebar.title("Filters")
    sector_filter = st.sidebar.multiselect("Select Sectors", data["Sector"].unique(), default=data["Sector"].unique())
    risk_level_filter = st.sidebar.multiselect("Select Risk Levels", data["ESG Risk Level"].unique(), default=data["ESG Risk Level"].unique())
    return sector_filter, risk_level_filter

@st.cache_data
def preprocess_data(sector_filter, risk_level_filter):
    filtered_data = data[
        (data["Sector"].isin(sector_filter)) &
        (data["ESG Risk Level"].isin(risk_level_filter))
    ]
    filtered_data["ESG Risk Percentile"] = filtered_data["ESG Risk Percentile"].str.extract(r'^(\d+)', expand=False).astype(int)
    return filtered_data

def feature_engineering(filtered_data):
    X = filtered_data.drop(["Name", "Description", "HasControversy"], axis=1)
    y = filtered_data["HasControversy"]
    cat_cols = X.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col])
    if not X.empty:
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y

def company_details(filtered_data):
    st.subheader("Company Details")
    company_name = st.selectbox("Select a Company", filtered_data["Name"].unique(), key="company_details")
    company_data = filtered_data[filtered_data["Name"] == company_name]
    st.write(company_data)

def esg_risk_analysis(filtered_data):
    st.subheader("ESG Risk Analysis")
    company_name = st.selectbox("Select a Company", filtered_data["Name"].unique(), key="esg_risk_analysis")
    company_data = filtered_data[filtered_data["Name"] == company_name]
    other_companies = filtered_data[filtered_data["Name"] != company_name]
    risk_metric = st.selectbox("Select Risk Metric", ["Total ESG Risk score", "Environment Risk Score", "Governance Risk Score", "Social Risk Score"], key="risk_metric")
    fig, ax = plt.subplots()
    sns.histplot(data=company_data, x=risk_metric, label=company_name, ax=ax)
    sns.histplot(data=other_companies, x=risk_metric, label="Other Companies", ax=ax)
    ax.set_title(f"Distribution of {risk_metric}")
    ax.set_xlabel(risk_metric)
    st.pyplot(fig)

def sector_comparison(filtered_data):
    st.subheader("Sector Comparison")
    company_name = st.selectbox("Select a Company", filtered_data["Name"].unique(), key="sector_comparison")
    company_data = filtered_data[filtered_data["Name"] == company_name]
    sector_data = filtered_data.groupby("Sector")["Total ESG Risk score"].mean().reset_index()
    company_sector = company_data["Sector"].iloc[0]
    company_risk_score = company_data["Total ESG Risk score"].iloc[0]
    fig, ax = plt.subplots()
    sns.barplot(data=sector_data, x="Sector", y="Total ESG Risk score", ax=ax)
    ax.set_title(f"Average Total ESG Risk Score by Sector")
    ax.text(company_sector, company_risk_score, f"{company_name} ({company_risk_score})", ha="center", va="bottom")
    st.pyplot(fig)

def risk_matrix(filtered_data):
    st.subheader("Risk Matrix")
    company_name = st.selectbox("Select a Company", filtered_data["Name"].unique(), key="risk_matrix")
    company_data = filtered_data[filtered_data["Name"] == company_name]
    other_companies = filtered_data[filtered_data["Name"] != company_name]
    fig, ax = plt.subplots()
    sns.scatterplot(data=other_companies, x="Governance Risk Score", y="Social Risk Score", hue="Total ESG Risk score", palette="viridis", ax=ax)
    sns.scatterplot(data=company_data, x="Governance Risk Score", y="Social Risk Score", color="black", marker="D", s=100, ax=ax)
    ax.set_title("Risk Matrix")
    ax.set_xlabel("Governance Risk Score")
    ax.set_ylabel("Social Risk Score")
    st.pyplot(fig)

def main():
    st.title("Finance Risk Analysis and Prediction Dashboard")
    sector_filter, risk_level_filter = sidebar_filters()
    filtered_data = preprocess_data(sector_filter, risk_level_filter)
    X, y = feature_engineering(filtered_data)

    company_details(filtered_data)
    esg_risk_analysis(filtered_data)
    sector_comparison(filtered_data)
    risk_matrix(filtered_data)

if __name__ == "__main__":
    data = pd.read_csv(r"C:\Users\asuss\Downloads\cleaned_data.csv")
    main()