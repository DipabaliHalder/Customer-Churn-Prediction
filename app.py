import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from mlxtend.plotting import plot_confusion_matrix
import pickle,os,joblib
from xgboost import XGBClassifier

df=pd.read_csv("Churn_Prediction.csv")
det="dataset_details.txt"
pick=False

st.header("Customer Churn Prediction")
tab1,tab2,tab3,tab4=st.tabs(["Home","Exploratory Data Analysis","Training & Evaluation","Prediction"])

with tab1:
    st.subheader("What is this?")
    st.write("Churn prediction involves identifying at-risk customers who are likely to cancel their subscriptions or close/abandon their accounts. A churn model works by passing previous customer data through a machine learning model to identify the connections between features and targets and make predictions about new customers.")
    st.subheader("About Dataset")
    with open(det, 'r') as file:
        content = file.read()
    st.markdown(content)
    st.subheader("Sample Data")
    st.write(df.sample(3))

with tab2:    
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.subheader("Correlation Heatmap(Only Numerical Columns)")
    data=df.drop(columns=['Geography','Gender'])
    fig,ax=plt.subplots()
    #data=df.drop(columns=['Geography','Gender'])
    sns.heatmap(data.corr(),annot=True)
    st.pyplot(fig)
    st.subheader("Tenure vs Exited")
    fig, ax = plt.subplots()
    sns.kdeplot(x=df['Tenure'], hue=df['Exited'], ax=ax, shade=True)
    st.pyplot(fig)
    
    d=st.columns(2)
    c=["Geography","Gender"]
    for i,j in zip(d,c):
        with i:
            fig, ax = plt.subplots()
            st.subheader(f"{j} vs Exited")
            sns.countplot(x=df[j],hue=df['Exited'],palette="muted")
            st.pyplot(fig)

    d=st.columns(2)
    c=["IsActiveMember","HasCrCard"]
    for i,j in zip(d,c):
        with i:
            fig, ax = plt.subplots()
            st.subheader(f"{j} vs Exited")
            sns.countplot(x=df[j],hue=df['Exited'],palette="muted")
            st.pyplot(fig)

    st.subheader("Credit Score vs Churn")
    fig,ax=plt.subplots()
    sns.histplot(x=df['CreditScore'],hue=df['Exited'],bins=60)
    st.pyplot(fig)

with tab3:
    data=df.drop(columns=['Geography','Gender'])
    models = [[SVC(), "Support Vector Machine"],
              [XGBClassifier(), "XG Boost Classifier"],
         [LogisticRegression(), "Logistic regression"],
         [RandomForestClassifier(), "Random Forest"],
         [DecisionTreeClassifier(), "Decision Trees"]]
    algo=[]
    for i in models:
        algo.append(i[1])
    ag=option = st.selectbox("Choose the Algorithm to be used:",sorted(algo),index=None,placeholder="Select an option")
    per=st.slider("Choose the percentage of data for Training:",min_value=10,max_value=100)
    y=data['Exited']
    X=data.drop(['Exited'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100-per)/100,random_state=42)
    if st.button("Train the Model"):
        st.success('model.pkl File created')
        pick=True
        for i in models:
            if i[1] == ag:
                name = i[1]
                model = i[0]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                col1,col2=st.columns(2)
                with col1:
                    cnf = confusion_matrix(y_test,y_pred)
                    fig, ax = plot_confusion_matrix(conf_mat = cnf)
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)
                with col2:
                    st.write(f'Algorithm: {name}')
                    st.write(f'Accuracy: {accuracy*100:0.2f}%')
                    st.write(f'Precision: {precision*100:0.2f}%')
                    st.write(f'Recall: {recall*100:.2f}%')
                    st.write(f'F1 Score: {f1*100:.2f}%')
                
                pickle.dump(model,open('model.pkl','wb'))
            else:
                continue

with tab4:
    cl=["CreditScore","Age","Balance","EstimatedSalary","Tenure","NumOfProducts"]
    input={}
    
    for i in df.columns.delete(-1):
        if df[f"{i}"].dtype=="int64":
            if i in cl:
                v=st.number_input(f"Enter value for {i}:", min_value=0)
        elif df[f"{i}"].dtype=="float64":
            if i in cl:
                v=float(st.number_input(f"Enter value for {i}:", min_value=0))
            else:
                option=df[f"{i}"].unique()
                val= st.selectbox(f"Enter value for {i}:",["Yes","No"],index=None,placeholder="Select an option")
                if val == "Yes": v=1
                else: v=0
        else:
            v=st.selectbox(f"Enter value for {i}:",df[f"{i}"].unique(),index=None,placeholder="Select an option")
        input[i]=v
    
    input_df = pd.DataFrame([input])
    if st.button("Predict"):
        if not os.path.exists("model.pkl"):
            st.error(f"Model file not found. Please train the model first.")
        else:
            input_df.drop(columns=["Gender","Geography"],inplace=True)
            model = joblib.load("model.pkl")
            st.write(input_df)
            prediction = model.predict(input_df)
            if prediction[0] == 1:
                st.info("Churn")
            else:
                st.info("No Churn")
