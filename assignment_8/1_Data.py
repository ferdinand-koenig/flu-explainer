import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fairness_functions as ff
import dalex as dx 
import plotly
import plotly.express as px
import warnings
import pickle
# from time import time
# from ucimlrepo import fetch_ucirepo
# from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
# from sklearn.neural_network import MLPClassifier

plotly.offline.init_notebook_mode()
warnings.filterwarnings("ignore")

st.set_page_config(page_title = "Data", page_icon="📊")

with open("obj_dict.pkl", "rb") as file:
    # Deserialize and load the object from the file
    obj_dict = pickle.load(file)

for name, obj in obj_dict.items():
    exec(name + "= obj")

st.title("CDC Diabetes Health Indicators Data")
st.write("Investigating CDC Diabetes Health Indicators Dataset which can be accessed via the UCI ML Reporitory: https://www.archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators.")

st.write('''
         \"The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. The 35 features consist of some demographics, lab test results, and answers to survey questions for each patient. 
         The target variable for classification is whether a patient has diabetes, is pre-diabetic, or healthy.\"
         Further information can also be found on the Kaggle webpage: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
         ''')
         
var_desc = {
    # Binary
    "HighBP": {"subtitle": "High blood pressure", 
               "mappings": {0: "no high BP", 1: "high BP"}},
    "HighChol": {"subtitle": "High cholesterol", 
                 "mappings": {0: "no high cholesterol", 1: "high cholesterol"}},
    "CholCheck": {"subtitle": "Cholesterol check in 5 years",
                  "mappings": {0: "no", 1: "yes"}},
    "Smoker": {"subtitle": "Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]",
               "mappings": {0: "no", 1: "yes"}},
    "Stroke": {"subtitle": "(Ever told) you had a stroke?",
               "mappings": {0: "no", 1: "yes"}},
    "HeartDiseaseorAttack": {"subtitle": "Coronary heart disease (CHD) or myocardial infarction (MI)",
                             "mappings": {0: "no", 1: "yes"}},
    "PhysActivity": {"subtitle": "physical activity in past 30 days - not including job",
                     "mappings": {0: "no", 1: "yes"}},
    "Fruits": {"subtitle": "Consume Fruit 1 or more times per day",
               "mappings": {0: "no", 1: "yes"}},
    "Veggies": {"subtitle": "Consume Vegetables 1 or more times per day",
                "mappings": {0: "no", 1: "yes"}},
    "HvyAlcoholConsump": {"subtitle": "Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)",
                          "mappings": {0: "no", 1: "yes"}},
    "AnyHealthcare": {"subtitle": "Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc.",
                      "mappings": {0: "no", 1: "yes"}},
    "NoDocbcCost": {"subtitle": "Was there a time in the past 12 months when you needed to see a doctor but could not because of cost?",
                    "mappings": {0: "no", 1: "yes"}},
    "Sex": {"subtitle": "Sex of the individual",
            "mappings": {0: "female", 1: "male"}},
    "DiffWalk": {"subtitle": "Do you have serious difficulty walking or climbing stairs?",
                 "mappings": {0: "no", 1: "yes"}},
    
    # Non-binary
    "BMI": "Body Mass Index",
    "GenHlth": "Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor",
    "MentHlth": "Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? Scale 1-30 days",
    "PhysHlth": "Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? Scale 1-30 days",
    "Age": "13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older",
    "Education": "Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate)",
    "Income": "Income scale (INCOME2 see codebook) scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more"
}

def get_info(var):
    return "\n".join([str(x) + ": " + y for x, y in var_desc[var]["mappings"].items()])

# Sidebar with options
st.sidebar.header("Filter Options")

# Target Feature
st.sidebar.write("Target Feature")
diabetes_binary = st.sidebar.multiselect("Diabetes Binary", df["Diabetes_binary"].unique(), df["Diabetes_binary"].unique(), help = "0: No diabetes\n1: Diabetes")
# Protected Characteristics
protected_chars = st.sidebar.expander("Protected Characteristics")
with protected_chars:
    age         = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())), help = var_desc["Age"])
    education   = st.slider("Education", int(df["Education"].min()), int(df["Education"].max()), (int(df["Education"].min()), int(df["Education"].max())), help = var_desc["Education"])
    income      = st.slider("Income", int(df["Income"].min()), int(df["Income"].max()), (int(df["Income"].min()), int(df["Income"].max())), help = var_desc["Income"])
    sex         = st.multiselect("Sex", df["Sex"].unique(), df["Sex"].unique(), help = get_info("Sex"))

# Other Characteristics
other_chars = st.sidebar.expander("Other Characteristics")
with other_chars:
    bmi         = st.slider("BMI", int(df["BMI"].min()), int(df["BMI"].max()), (int(df["BMI"].min()), int(df["BMI"].max())), help = var_desc["BMI"])
    genhlth     = st.slider("GenHlth", int(df["GenHlth"].min()), int(df["GenHlth"].max()), (int(df["GenHlth"].min()), int(df["GenHlth"].max())), help = var_desc["GenHlth"])
    menhlth     = st.slider("MentHlth", int(df["MentHlth"].min()), int(df["MentHlth"].max()), (int(df["MentHlth"].min()), int(df["MentHlth"].max())), help = var_desc["MentHlth"])
    physhlth    = st.slider("PhysHlth", int(df["PhysHlth"].min()), int(df["PhysHlth"].max()), (int(df["PhysHlth"].min()), int(df["PhysHlth"].max())), help = var_desc["PhysHlth"])

    highbp          = st.multiselect("HighBP", df["HighBP"].unique(), df["HighBP"].unique(), help = get_info("HighBP"))
    highchol        = st.multiselect("HighChol", df["HighChol"].unique(), df["HighChol"].unique(), help = get_info("HighChol"))
    cholcheck       = st.multiselect("CholCheck", df["CholCheck"].unique(), df["CholCheck"].unique(), help = get_info("CholCheck"))
    smoker          = st.multiselect("Smoker", df["Smoker"].unique(), df["Smoker"].unique(), help = get_info("Smoker"))
    stroke          = st.multiselect("Stroke", df["Stroke"].unique(), df["Stroke"].unique(), help = get_info("Stroke"))
    heartdisease    = st.multiselect("HeartDiseaseorAttack", df["HeartDiseaseorAttack"].unique(), df["HeartDiseaseorAttack"].unique(), help = get_info("HeartDiseaseorAttack"))
    physact         = st.multiselect("PhysActivity", df["PhysActivity"].unique(), df["PhysActivity"].unique(), help = get_info("PhysActivity"))
    fruits          = st.multiselect("Fruits", df["Fruits"].unique(), df["Fruits"].unique(), help = get_info("Fruits"))
    vegs            = st.multiselect("Veggies", df["Veggies"].unique(), df["Veggies"].unique(), help = get_info("Veggies"))
    alcohol         = st.multiselect("HvyAlcoholConsump", df["HvyAlcoholConsump"].unique(), df["HvyAlcoholConsump"].unique(), help = get_info("HvyAlcoholConsump"))
    healthcare      = st.multiselect("AnyHealthcare", df["AnyHealthcare"].unique(), df["AnyHealthcare"].unique(), help = get_info("AnyHealthcare"))
    doccosts        = st.multiselect("NoDocbcCost", df["NoDocbcCost"].unique(), df["NoDocbcCost"].unique(), help = get_info("NoDocbcCost"))
    diffwalk        = st.multiselect("DiffWalk", df["DiffWalk"].unique(), df["DiffWalk"].unique(), help = get_info("DiffWalk"))

filtered_df = df[
    # Target
    (df["Diabetes_binary"].isin(diabetes_binary)) &
    # Protected Characteristics
    (df["Age"] >= age[0]) & (df["Age"] <= age[1]) & 
    (df["Education"] >= education[0]) & (df["Education"] <= education[1]) & 
    (df["Income"] >= income[0]) & (df["Income"] <= income[1]) & 
    (df["Sex"].isin(sex)) &
    # Other Characteristics
    (df["BMI"] >= bmi[0]) & (df["BMI"] <= bmi[1]) & 
    (df["GenHlth"] >= genhlth[0]) & (df["GenHlth"] <= genhlth[1]) & 
    (df["MentHlth"] >= menhlth[0]) & (df["MentHlth"] <= menhlth[1]) & 
    (df["PhysHlth"] >= physhlth[0]) & (df["PhysHlth"] <= physhlth[1]) & 
    (df["HighBP"].isin(highbp)) &
    (df["HighChol"].isin(highchol)) &
    (df["CholCheck"].isin(cholcheck)) &
    (df["Smoker"].isin(smoker)) &
    (df["Stroke"].isin(stroke)) &
    (df["HeartDiseaseorAttack"].isin(heartdisease)) &
    (df["PhysActivity"].isin(physact)) &
    (df["Fruits"].isin(fruits)) &
    (df["Veggies"].isin(vegs)) &
    (df["HvyAlcoholConsump"].isin(alcohol)) &
    (df["AnyHealthcare"].isin(healthcare)) &
    (df["NoDocbcCost"].isin(doccosts)) &
    (df["DiffWalk"].isin(diffwalk))
]

# Display the dataset
st.write("### Inspecting the Data")
st.dataframe(filtered_df)

def prettyDescribe(data):
    return round(data.describe(), 2).drop("count", axis = 0)
    
st.write(f"{len(filtered_df):,} observations | {len(df) - len(filtered_df):,} ({100 * (len(df) - len(filtered_df))/len(df):.1f}%) of observations filtered out")
st.dataframe(prettyDescribe(filtered_df))

st.write("### Distributions of Diabetes Health Indicators")

# Plotly
# Plot distributions
plot_type = st.radio("Select plot type", ["Histogram", "100% Stacked Bar"])
if plot_type == "Histogram":
    barnorm_val = None
    text_auto_val = ","
else:
    barnorm_val = "percent"
    text_auto_val = ".1f"

to_plot = [x for x in df.columns if x not in ["ID", "Diabetes_binary"]]
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20, tab21 = st.tabs(to_plot)
tabs = [tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14, tab15, tab16, tab17, tab18, tab19, tab20, tab21]

for tab, var in zip(tabs, to_plot):
    with tab:
        st.write(f"### {var}")
        if len(set(filtered_df[var])) == 2:
            st.write(f"{var_desc[var]['subtitle']}")
            fig = px.histogram(filtered_df.sort_values(by = "Diabetes_binary", ascending = False), x = var, color = "Diabetes_binary", 
                   color_discrete_map = {0: "#636EFA", 1: "#EF553B"},
                   barnorm = barnorm_val, text_auto = text_auto_val)
            fig.update_layout(xaxis = dict(tickvals = list(var_desc[var]["mappings"].keys()), ticktext = list(var_desc[var]["mappings"].values())))
        else:
            st.write(f"{var_desc[var]}")
            fig = px.histogram(filtered_df.sort_values(by = "Diabetes_binary", ascending = False), x = var, color = "Diabetes_binary", 
                   color_discrete_map = {0: "#636EFA", 1: "#EF553B"}, 
                   barnorm = barnorm_val, text_auto = text_auto_val, marginal = "box")
        fig.update_traces(marker_line_width = 1, marker_line_color = "white")
        st.plotly_chart(fig)
        
