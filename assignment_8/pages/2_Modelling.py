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

st.set_page_config(page_title = "Modelling", page_icon="🤖")

with open("obj_dict.pkl", "rb") as file:
    # Deserialize and load the object from the file
    obj_dict = pickle.load(file)

for name, obj in obj_dict.items():
    exec(name + "= obj")

exp = dx.Explainer(mlp, X_train, y_train)

st.write("### Modelling")
st.markdown('''
            The model used for classifying between "Diabetes" and "No Diabetes" is a multi-layer Perceptron classifier which optinises the log-loss function using LBFGS or stochastic gradient descent.
            From [scikit-learn: 1.17.1. Multi-layer Perceptron](https://scikit-learn.org/stable/modules/neural_networks_supervised.html):

            \"
            Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function $f: \mathrm R^m \\to \mathrm R^o$ by training on a dataset, 
            where $m$ is the number of dimensions for input and $o$ is the number of dimensions for output. Given a set of features $X = x_1, x_2, \ldots, x_m$ and a target $y$, 
            it can learn a non-linear function approximator for either classification or regression. It is different from logistic regression, 
            in that between the input and the output layer, there can be one or more non-linear layers, called hidden layers. 

            The leftmost layer, known as the input layer, consists of a set of neurons $\{x_i | x_1, x_2, \ldots, x_m\}$
            representing the input features. Each neuron in the hidden layer transforms the values from the previous layer with a weighted linear summation
            $w_1 x_1 + w_2 x_2 + \ldots + w_m x_m $, 
            followed by a non-linear activation function $g(\cdot): \mathrm R \\to \mathrm R$ - like the hyperbolic tan function. 
            The output layer receives the values from the last hidden layer and transforms them into output values.
            \"
         ''')

tab22, tab23, tab24 = st.tabs(["Model Performance", "Model Explanations", "Model Fairnesss"])

# Model Performance
with tab22:
    fig, ax = plt.subplots()
    ax.hist(df_preds["predicted_proba"], color = "skyblue", edgecolor = "black")
    ax.set_title("MLP Classifier: Distribution of Predicted Values")
    st.pyplot(fig)

    # MLP
    st.write("MLP:\n\nClassification Report:")
    st.dataframe(pd.DataFrame(classification_report(df_preds['Diabetes_binary'], df_preds['predicted_binary'], output_dict = True)).T)

    confusion_matrix_explained = st.expander("Confusion Matrix Explained")
    with confusion_matrix_explained:
        st.markdown('''
                    In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as error matrix, is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one; in unsupervised learning it is usually called a matching matrix.
                    Each row of the matrix represents the instances in an actual class while each column represents the instances in a predicted class, or vice versa – both variants are found in the literature. The name stems from the fact that it makes it easy to see whether the system is confusing two classes (i.e. commonly mislabeling one as another).
                    It is a special kind of contingency table, with two dimensions ("actual" and "predicted"), and identical sets of "classes" in both dimensions (each combination of dimension and class is a variable in the contingency table).
                    
                    [source](https://en.wikipedia.org/wiki/Confusion_matrix)
                    ''')

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(df_preds["Diabetes_binary"], df_preds["predicted_binary"])).plot(ax = ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    roc_curve_explained = st.expander("ROC Curve Explained")
    with roc_curve_explained:
        st.markdown('''
                    A receiver operating characteristic curve, or ROC curve, is a graphical plot that illustrates the performance of a binary classifier model (can be used for multi class classification as well) at varying threshold values.
                    The ROC curve is the plot of the true positive rate (TPR) against the false positive rate (FPR) at each threshold setting.
                    The ROC can also be thought of as a plot of the statistical power as a function of the Type I Error of the decision rule (when the performance is calculated from just a sample of the population, it can be thought of as estimators of these quantities). The ROC curve is thus the sensitivity or recall as a function of false positive rate.
                    Given the probability distributions for both true positive and false positive are known, the ROC curve is obtained as the cumulative distribution function (CDF, area under the probability distribution from 
                    $- \infty$ to the discrimination threshold) of the detection probability in the y-axis versus the CDF of the false positive probability on the x-axis.
                    ROC analysis provides tools to select possibly optimal models and to discard suboptimal ones independently from (and prior to specifying) the cost context or the class distribution. ROC analysis is related in a direct and natural way to the cost/benefit analysis of diagnostic decision making.
                    
                    [source](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
                    ''')
        
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = roc_curve(df_preds["Diabetes_binary"], df_preds["predicted_proba"]) 
    roc_auc = auc(fpr, tpr)
    # Plot the ROC curve 
    ax.plot(fpr, tpr, label = f"ROC curve (area = {roc_auc:0.2f})")
    ax.plot([0, 1], [0, 1], 'k--', label='No Skill')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)

# Model Explanations
with tab23:
    st.write("#### Variable Importance")
    
    feature_importance_explained = st.expander("Feature Importance Explained")
    with feature_importance_explained:
        st.markdown('''
                    Feature importance is a step in building a machine learning model that involves calculating the score for all input features in a model to establish the importance of each feature in the decision-making process. 
                    The higher the score for a feature, the larger effect it has on the model to predict a certain variable. 
                    
                    [source](https://builtin.com/data-science/feature-importance#:~:text=Feature%20importance%20is%20a%20step,to%20predict%20a%20certain%20variable.)
                    ''')

    st.plotly_chart(vi.plot(show = False))
    st.markdown('''
                Here, variable importance is computed by the drop-out loss. 

                $$\\text{Dropout Loss} = \\text{Performance with all variables} − \\text{Performance with variable dropped}$$

                A larger dropout loss indicates that the variable is more important because its removal significantly impacts the model's performance.

                As we can see from the variable importance plot, our top 5 most 'influential' features in the model are: `GenHealth`, `BMI`, `Age`, `HighBP`, `HighChol`.

                The least 'influential' features in the model are: `PhysActivity`, `Veggies`, `AnyHealthcare`, `Education`, `NoDocbcCost`.
                ''')
    
    st.plotly_chart(vi_grouped.plot(show = False))
    st.markdown('''
                Here, we also want to assess the variable importance of features when they are grouped. Most interestingly, we can look to see how protected characteristics (i.e. `Sex`, `Age`, `Education`, `Income`) impact a model's performance. This is to explore if they can be dropped fully or not with any significant changes to the mode's predictive ability.
                
                As we can see from the plot above, protected characteristics have limited influence over the model's performance when permuted from the feature list (compared to 'unprotected' charaxcteristics). Therefore, it can be said that dropping these features will be of limited consequence for the accuracy of the model, whilst improving other statistics (such as when we perform fairness checks).
                ''')

    st.write("#### Partial Dependence & Accumulated Local Dependence")
    
    partial_dependence_explained = st.expander("Partial Dependence & Accumulated Local Dependence Explained")
    with partial_dependence_explained:
        st.markdown('''
                    The general idea underlying the construction of Partial-dependence (PD) profiles is to show how does the expected value of model prediction behave as a function of a selected explanatory variable. For a single model, one can construct an overall PD profile by using all observations from a dataset, or several profiles for sub-groups of the observations. Comparison of sub-group-specific profiles may provide important insight into, for instance, the stability of the model’s predictions.
                    PD profiles are easy to explain and interpret, especially given their estimation as the mean of ceteris-paribus (CP) profiles. However, the profiles may be misleading if, for instance, explanatory variables are correlated (in ceteris-paribus profiles).
                    Accumulated-local profiles address this issue, and provides a more refined understanding compared to traditional partial dependence plots by focusing on local changes rather than averaging over the entire dataset.

                    [source 1](https://ema.drwhy.ai/partialDependenceProfiles.html)

                    [source 2](https://ema.drwhy.ai/accumulatedLocalProfiles.html)
                    ''')

    st.plotly_chart(pdp_num.plot(ale_num, show = False))

    st.markdown('''
                Most significant findings from the partial dependence plots:

                * Predictions that an individual has diabetes increase when:  
                    * has high blood pressure
                    * has high cholesterol
                    * has heart disease or has had an attack
                    * (perceived) general health decreases
                    * ages
                    * has a lower income

                Some very interesting discoveries:

                * Predictions increase with BMI, but then begins to decrease with a BMI > 55. I do not have any intuition as to why.

                * Most interestingly (similar to our logistic regression results), heavy alcohol consumption reduces the model's expectation a person has diabetes. Again I do not have any intuition as to why.
                ''')

    st.plotly_chart(pdp_cat.plot(ale_cat, show = False))

    st.markdown('''
                We choose two of the most influential features from our model: `GenHlth` and `BMI`. 

                `GenHlth` appears linearly correlated with our predictions. As our value increases (i.e. perceived general health gets worse), our prediction of a positive class also increases.

                Similiar to what we saw in our PD plot, predictions of a positive class increase when `BMI` does up until an approximate value of 55, when it then begins to decrease.
                ''')

    st.write("#### Prediction Breakdown & Shapley Values")

    shapley_values_explained = st.expander("Shapley Values Explained")
    with shapley_values_explained:
        st.markdown('''
                    In explainable machine learning the Shapley value is used to measure the contributions of input features to the output of a machine learning model at the instance level. Given a specific data point, the goal is to decompose the model prediction and assign Shapley values to individual features of the instance.

                    [source](https://www.ijcai.org/proceedings/2022/0778.pdf)
                    ''')

    st.write("#### Most confident model prediction")
    st.write("#### Positive Class (i.e. Diabetes_binary = 1)")

    st.plotly_chart(bd_1.plot(bd_interactions_1, show = False))
    st.markdown('''All features increase the average response indicating this person has diabetes.''')

    st.plotly_chart(sh_1.plot(show = False))
    st.markdown('''For the most confident prediction that the individual has diabetes, `BMI = 64`, `GenHlth = 5.0`, and `PhysHlth = 30` are the most influential features in this specific prediction.''')

    st.write("#### Negative Class (i.e. Diabetes_binary = 0)")
    
    st.plotly_chart(bd_0.plot(bd_interactions_0, show = False))
    st.markdown('''In the breakdown `Age = 1.0` and `GenHlth = 1.0` decrease the average response most significantly.''')

    st.plotly_chart(sh_0.plot(show = False))
    st.markdown('''For the most confident prediction that the individual does not have diabetes, `GenHlth = 1.0`, `HighBP = 0.0`, and `HvyAlcoholConsump = 1.0` are the most influential features in this specific prediction.''')
    
    st.write("#### Most indecesive model prediction")
    
    st.plotly_chart(bd_05.plot(bd_interactions_05, show = False))
    st.markdown('''For the most indecisive prediction whether an individual has diabetes for not, all features provide an average increase in the response besides `Education = 6.0`, `HeartDiseaseorAttack = 0.0` and `PhysHlth = 0.0`''')

    st.plotly_chart(sh_05.plot(show = False))
    st.markdown('''For the most indecisive prediction that the individual does not have diabetes, `BMI = 38.0`, `Age = 11`, and `HighBP = 1.0` are the most influential features in this specific prediction, and have an average increase of the response.''')

    st.write("#### Ceteris Paribus Profiles for Most Confident Predictions")

    ceteris_paribus_explained = st.expander("Ceteris Paribus Profiles Explained")
    with ceteris_paribus_explained:
        st.markdown('''
                    Ceteris-paribus (CP) profiles show how a model’s prediction would change if the value of a single exploratory variable changed. In essence, a CP profile shows the dependence of the conditional expectation of the dependent variable (response) on the values of the particular explanatory variable.

                    [source](https://ema.drwhy.ai/ceterisParibus.html)
                    ''')

    st.plotly_chart(cp_1.plot(cp_0, show = False))

    st.markdown('''
                Largely, when comparing the most confident predictions when an individual has diabetes vs does not have diabetes - we can see the values of their features are polarised (in other words, they are on opposite ends of the scale). There are:
                * HighBP
                * HighChol
                * BMI
                * Smoker
                * Heart Disease of Attack
                * Physical Activity
                * Veggies
                * Heavy Alcohol Consumption
                * General Health
                * Mental Health
                * Physical Health
                * Difficulty Walking
                * Age 
                * Income
                ''')

# Model Fairness
with tab24:

    # fairness_var = st.radio("Select protected characteristic", ["Age", "Sex", "Education", "Income"])
    fairness_sex_0 = ff.group_fairness(df_preds, "Sex", 0, "predicted_binary", 1)
    fairness_sex_1 = ff.group_fairness(df_preds, "Sex", 1, "predicted_binary", 1)

    fairness_smoker_sex_0 = ff.conditional_statistical_parity(df_preds, "Sex", 0, "predicted_binary", 1, "Smoker", 1)
    fairness_smoker_sex_1 = ff.conditional_statistical_parity(df_preds, "Sex", 1, "predicted_binary", 1, "Smoker", 1)

    parity_sex_0 = ff.predictive_parity(df_preds, "Sex", 0, "predicted_binary", "Diabetes_binary")
    parity_sex_1 = ff.predictive_parity(df_preds, "Sex", 1, "predicted_binary", "Diabetes_binary")

    er_sex_0 = ff.fp_error_rate_balance(df_preds, "Sex", 0, "predicted_binary", "Diabetes_binary")
    er_sex_1 = ff.fp_error_rate_balance(df_preds, "Sex", 1, "predicted_binary", "Diabetes_binary")

    st.write("### Fairness Metrics")

    st.write("#### Group Fairness")
    st.write("Members of each group need to have the same probability of being assigned to the positively predicted class.")
    st.write(f"Sex class 0 (female):\t{fairness_sex_0}")
    st.write(f"Sex class 1 (male):\t{fairness_sex_1}")
    st.write(f"Difference: {np.abs(fairness_sex_0 - fairness_sex_1):.3f}")

    st.write("#### Conditional Statistical Parity")
    st.write("Members of each group need to have the same probability of being assigned to the positive class under the same set of conditions.")
    st.write(f"Smoker + Sex class 0 (female):\t{fairness_smoker_sex_0}")
    st.write(f"Smoker + Sex class 1 (male):\t{fairness_smoker_sex_1}")
    st.write(f"Difference: {np.abs(fairness_smoker_sex_0 - fairness_smoker_sex_1):.3f}")

    st.write(f"#### Predictive Parity")
    st.write("Members of each group have the same Positive Predictive Value (PPV) — the probability of a subject with Positive Predicted Value to truly belong to the positive class.")
    st.write(f"Sex class 0 (female):\t{parity_sex_0:.3f}")
    st.write(f"Sex class 1 (male):\t{parity_sex_1:.3f}")
    st.write(f"Difference: {np.abs(parity_sex_0 - parity_sex_1):.3f}")

    st.write(f"#### False Positive Error Rate Balance")
    st.write("Members of each group have the same False Positive Rate (FPR) — the probability of a subject in the negative class to have a positive predicted value.")
    st.write(f"Sex class 0 (female):\t{er_sex_0:.3f}")
    st.write(f"Sex class 1 (male):\t{er_sex_1:.3f}")
    st.write(f"Difference: {np.abs(er_sex_0 - er_sex_1):.3f}")