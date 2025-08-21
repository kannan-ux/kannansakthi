import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from streamlit_option_menu import option_menu
with st.sidebar:
    st.markdown(
        """
        <style>
            /* Change Sidebar background color */
            section[data-testid="stSidebar"] {
                background-color: #D3D3D3; ;
            }

            /* Optional: change sidebar text color */
            section[data-testid="stSidebar"] * {
                color: black;  /* you can also use white if bg is dark */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    selection = option_menu("🔮 Start Your Prediction Journey", ["About","Prediction"],
                            menu_icon="heart_fill",
                            icons=['heart','star-fill'],
                            default_index=0)
if selection=="About":
    st.title("🩺 Billing Denial Prediction")
    st.markdown('<h1 style="text-align: center;">Go ahead and upload your CSV and Excel file, and I’ll take care of</h1>',unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center;">Model Training</h3>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align:center;">Predicting Outcome</h3>', unsafe_allow_html=True)
    st.image(r"C:\Users\sakth\Downloads\medical-billing-systems.jpeg")
if selection=="Prediction":
    train_file = st.file_uploader("Choose CSV or Excel file ", type=["csv", "xlsx"])
    if train_file is not None:
        if train_file.name.endswith(".csv"):
            df_train = pd.read_csv(train_file, skiprows=[0, 1])
        elif train_file.name.endswith(".xlsx"):
            df_train = pd.read_excel(train_file, skiprows=[0, 1])
        df_train.drop("#", axis=1, inplace=True)
        df_train.dropna(subset=['CPT Code', 'Insurance Company', 'Physician Name', 'Payment Amount', 'Balance'],
                    inplace=True)
        df_train['CPT Code'] = df_train['CPT Code'].astype(int)
        df_train['Payment Amount'] = df_train['Payment Amount'].replace('[$]', "", regex=True).astype(float)
        df_train['Balance'] = df_train['Balance'].replace('[$]', "", regex=True).astype(float)
        df_train['Denial Reason'] = df_train['Denial Reason'].fillna('paid')
        le_ins = LabelEncoder()
        le_doc = LabelEncoder()
        df_train['Insurance Company'] = le_ins.fit_transform(df_train['Insurance Company'])
        df_train['Physician Name'] = le_doc.fit_transform(df_train['Physician Name'])
        st.dataframe(df_train)
        X = df_train[['CPT Code', 'Insurance Company', 'Physician Name', 'Payment Amount', 'Balance']]
        y = df_train['Denial Reason']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        st.success("Model trained successfully!")
        st.subheader("Predict Denial Reason")
        cpt = st.selectbox("CPT Code", df_train['CPT Code'].unique())
        ins = st.selectbox("Insurance Company", df_train['Insurance Company'].unique())
        doc = st.selectbox("Physician Name", df_train['Physician Name'].unique())
        pay = st.number_input("Payment Amount", min_value=0.0, value=0.0)
        bal = st.number_input("Balance", min_value=0.0, value=0.0)
        if st.button("Predict Denial Reason"):
            input_df = pd.DataFrame({
            'CPT Code': [cpt],
            'Insurance Company': [ins],
            'Physician Name': [doc],
            'Payment Amount': [pay],
            'Balance': [bal]
        })
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Denial Reason: {prediction}")



