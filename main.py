import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
from plotly import offline
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")
import pickle


def preprocess_data():
    df = pd.read_csv("./input/raw_data/telecom_customer_churn.csv")
    df.columns = df.columns.str.replace(" ", "").str.lower()
    df["avgmonthlylongdistancecharges"] = df["avgmonthlylongdistancecharges"].fillna(
        0.0
    )
    df.multiplelines = df.multiplelines.fillna("no phone service")
    no_internet = [
        "internettype",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotectionplan",
        "premiumtechsupport",
        "streamingtv",
        "streamingmovies",
        "streamingmusic",
        "unlimiteddata",
    ]
    df[no_internet] = df[no_internet].fillna("no internet service")
    df["avgmonthlygbdownload"] = df["avgmonthlygbdownload"].fillna(0)
    df = df.drop(
        columns=[
            "customerid",
            "churncategory",
            "churnreason",
            "totalrefunds",
            "zipcode",
            "longitude",
            "latitude",
            "city",
        ]
    )
    df = df.loc[~df.customerstatus.str.contains("Join")]
    df.reset_index(drop=True, inplace=True)
    return df


def plot_churn_distribution(df):
    type_ = ["No", "yes"]
    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(
        go.Pie(
            labels=type_,
            values=df["customerstatus"].value_counts(),
            name="customerstatus",
        )
    )

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=0.4, hoverinfo="label+percent+name", textfont_size=16)

    fig.update_layout(
        title_text="Churn Distributions",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text="Churn", x=0.5, y=0.5, font_size=20, showarrow=False)],
    )
    fig.show()


def count_by_gender(df):
    return (
        df["customerstatus"][df["customerstatus"] == "Stayed"]
        .groupby(by=df["gender"])
        .count(),
        df["customerstatus"][df["customerstatus"] == "Churned"]
        .groupby(by=df["gender"])
        .count(),
    )


def plot_contract_distribution(df):
    fig = px.histogram(
        df,
        x="customerstatus",
        color="contract",
        barmode="group",
        title="<b>Customer contract distribution<b>",
    )
    fig.update_layout(width=700, height=500, bargap=0.2)
    fig.show()


def exploratory_analysis(df):
    plot_churn_distribution(df)
    stayed_gender_count, churned_gender_count = count_by_gender(df)
    plot_contract_distribution(df)


def encode_labels(df):
    # Create a label encoder object
    le = LabelEncoder()
    # Label Encoding will be used for columns with 2 or less unique

    le_count = 0
    for col in df.columns[1:]:
        if df[col].dtype == "object":
            if len(list(df[col].unique())) <= 2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                le_count += 1
    print("{} columns were label encoded.".format(le_count))


def convert_gender_to_bool(df):
    df["gender"] = [1 if each == "Female" else 0 for each in df["gender"]]


def encode_data(dataframe):
    if dataframe.dtype == "object":
        dataframe = LabelEncoder().fit_transform(dataframe)
    return dataframe


def feature_transformation(df):
    encode_labels(df)
    convert_gender_to_bool(df)
    df = df.apply(lambda x: encode_data(x))
    return df


def main():
    df = preprocess_data()
    exploratory_analysis(df)
    df = feature_transformation(df)
    df.to_csv("./input/clean_data/telecom_customer_churn.csv", index=False)


if __name__ == "__main__":
    main()
