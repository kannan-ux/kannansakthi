import pandas as pd
import  streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import  warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pydeck.data_utils.viewport_helpers import k_nearest_neighbors
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from sklearn.metrics import  mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import recall_score
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import  DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy  as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from streamlit_option_menu import option_menu
multichoice = st.sidebar.radio( "🌟 **Select Your Role**", ["🛍️I'm here to Predict "]  )
if multichoice == "🛍️I'm here to Predict ":
    st.markdown("""
            <style>
                .stApp {
                    background-color:grey;  
                }
            </style>
        """, unsafe_allow_html=True)
    st.snow()
    with st.sidebar:
        selection = option_menu("🔮 Start Your Prediction Journey", [ "Prediction"],
                                menu_icon="heart_fill",
                                icons=[ 'heart'],
                                default_index=0)
    if selection == "Prediction":
        st.header("Prediction Starts Here")
        data = st.file_uploader("Choose A File", type=["csv", "xlsx", "txt"])
        if data is not None:
            if data.name.endswith('.csv'):
                df = pd.read_csv(data)
                st.subheader("Exploratory Data Analysis")
                st.subheader("Handle Missing Values")
                target_column1 = st.selectbox("select the target columns", df.columns)
                if target_column1:
                    feature_column = [col for col in df.columns if col != target_column1]
                    selected_feature = st.multiselect("select a feature", feature_column)
                if selected_feature:
                    X = df[feature_column]
                    Y = df[target_column1]
                    numerical_cols = X.select_dtypes(include=["number"]).columns
                    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
                    if np.issubdtype(Y.dtype, np.number) or len(
                            numerical_cols) > 0:

                        num_strategy = st.selectbox(
                            "Choose a strategy for numerical columns", ["mean", "median", "constant"]
                        )
                        if num_strategy == "constant":
                            num_fill_value = st.text_input("Enter the constant value for numerical columns", value="0")
                            num_imputer = SimpleImputer(strategy=num_strategy, fill_value=float(num_fill_value))
                        else:
                            num_imputer = SimpleImputer(strategy=num_strategy)
                        df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
                        st.write(f"Numerical columns imputed using `{num_strategy}` strategy.")

                    if len(categorical_cols) > 0:

                        cat_strategy = st.selectbox(
                            "Choose a strategy for categorical columns", ["most_frequent", "constant"]
                        )
                        if cat_strategy == "constant":
                            cat_fill_value = st.text_input("Enter the constant value for categorical columns", value="")
                            cat_imputer = SimpleImputer(strategy=cat_strategy, fill_value=cat_fill_value)
                        else:
                            cat_imputer = SimpleImputer(strategy=cat_strategy)
                        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
                        st.write(f"Categorical columns imputed using `{cat_strategy}` strategy.")


                    st.subheader("Convert Categorical to Numerical")
                    categorical_cols = X.select_dtypes(include=["object", "category"]).columns
                    if Y.dtype in ['object', 'category']:
                        encoder1 = LabelEncoder()
                        Y_encoded = encoder1.fit_transform(Y)
                        df[target_column1 + "_encoded"] = Y_encoded
                    else:
                        st.write("Y does not have categorical data.")
                    if not categorical_cols.empty:

                        encoding_strategy = st.selectbox(
                            "Choose an encoding strategy", ["Label Encoding", "One-Hot Encoding", "Custom Mapping"]
                        )
                        if encoding_strategy == "Label Encoding":
                            for col in categorical_cols:
                                encoder = LabelEncoder()
                                df[col] = encoder.fit_transform(df[col])

                        elif encoding_strategy == "One-Hot Encoding":
                            st.write("Applying One-Hot Encoding")
                            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                            st.write("One-Hot Encoding completed. New columns:")


                        elif encoding_strategy == "Custom Mapping":
                            for col in categorical_cols:
                                st.write(f"Provide a mapping for column: `{col}`")
                                unique_values = df[col].unique()
                                st.write(f"Unique values: {unique_values}")
                                mapping = {}
                                for value in unique_values:
                                    mapped_value = st.number_input(f"Map `{value}` in `{col}` to:", value=0,
                                                                   key=f"{col}_{value}")
                                    mapping[value] = mapped_value
                                df[col] = df[col].map(mapping)
                                st.write(f"Mapping applied for `{col}`: {mapping}")
                        st.subheader("Dataset After Encoding")
                        st.dataframe(df)
                    else:
                        st.write("No categorical columns found in the dataset.")
                    num_values = Y.nunique()
                    if Y.dtype == 'object' or Y.dtype == 'int64':
                        if num_values == 2:
                            algorithm = "Binary Class"
                        elif Y.dtype == 'object':
                            algorithm = "Multi Class"

                    if Y.dtype == 'int64':
                        if num_values > 10:
                            algorithm = "Linear Model"
                        else:
                            algorithm = "Non linear Model"
                    st.write(f"Chosen algorithm is: {algorithm}")

                    if algorithm == "Binary Class":
                        r = st.selectbox("choose the algorthim:", ['Logistic Regression', "SVC Algorthim"])
                        X_encode = pd.get_dummies(X)
                        original_columns = X_encode.columns
                        imputer = SimpleImputer(strategy="mean")
                        X_encoded_imputed = imputer.fit_transform(X_encode)
                        X_encode = pd.DataFrame(X_encoded_imputed, columns=original_columns)
                        if X_encode.isna().sum().sum() > 0:
                            st.write("NaNs found in feature matrix after imputation!")
                            X_encode = X_encode.fillna(0)
                        X_encode_train, X_encode_test, Y_train, Y_test = train_test_split(
                            X_encode, Y, test_size=0.2, train_size=0.8, random_state=42
                        )
                        if Y_train.isna().sum() > 0:
                            st.write("NaNs found in target data! Showing rows with NaN values:")
                            rows_with_nan = Y_train[Y_train.isna()]
                            Y_train = Y_train.dropna()
                            X_encode_train = X_encode_train.loc[Y_train.index]
                        if r == 'Logistic Regression':

                            label_encoder = LabelEncoder()
                            Y_test_encoded = label_encoder.fit_transform(Y_test)

                            logistic_model = LogisticRegression()
                            logistic_model.fit(X_encode_train, Y_train)
                            logistic_y_pred = logistic_model.predict(X_encode_test)
                            logistic_y_pred_encoded = label_encoder.transform(logistic_y_pred)
                            logistic_precision = precision_score(Y_test_encoded, logistic_y_pred_encoded)
                            logistic_recall = recall_score(Y_test_encoded, logistic_y_pred_encoded)
                            st.write("Using Logistic Regression (Meets Precision/Recall thresholds)")
                            st.write(f"Precision: {logistic_precision}")
                            st.write(f"Recall: {logistic_recall}")
                            st.write(
                                f"Classification Report:\n{classification_report(Y_test_encoded, logistic_y_pred_encoded)}")
                            selected_model = logistic_model
                            user_input = {}
                            for feature in selected_feature:
                                user_input[feature] = st.text_input(f"enter the value{feature}:")
                            if all(value != "" for value in user_input.values()):
                                user_input_df = pd.DataFrame(user_input, index=[0])
                                user_input_encode = pd.get_dummies(user_input_df)
                                user_input_encode = user_input_encode.reindex(columns=X_encode.columns, fill_value=0)
                                prediction = selected_model.predict(user_input_encode)
                                st.write(f" the predict model is :{prediction[0]}")
                                st.success("🚀 Boom! Here's your final prediction!")
                        if r == 'SVC Algorthim':
                            st.write("Switching to SVC Classifier (Precision/Recall thresholds not met)")
                            label_encoder = LabelEncoder()
                            Y_test_encoded = label_encoder.fit_transform(Y_test)
                            base_estimator = SVC(kernel='linear', class_weight='balanced', random_state=42)
                            bagging_model = BaggingClassifier(estimator=base_estimator, n_estimators=10,
                                                              random_state=42)
                            bagging_model.fit(X_encode_train, Y_train)
                            bagging_y_pred = bagging_model.predict(X_encode_test)
                            bagging_y_pred_encoded = label_encoder.transform(bagging_y_pred)
                            bagging_precision = precision_score(Y_test_encoded, bagging_y_pred_encoded)
                            bagging_recall = recall_score(Y_test_encoded, bagging_y_pred_encoded)
                            st.write(f"Precision: {bagging_precision}")
                            st.write(f"Recall: {bagging_recall}")
                            st.write(f"Classification Report:\n{classification_report(Y_test, bagging_y_pred)}")
                            selected_model = bagging_model
                            user_input1 = {}
                            for feature in selected_feature:
                                user_input1[feature] = st.text_input(f"enter the value{feature}:")
                            if all(value != "" for value in user_input1.values()):
                                user_input1_df = pd.DataFrame(user_input1, index=[0])
                                user_input1_encode = pd.get_dummies(user_input1_df)
                                user_input1_encode = user_input1_encode.reindex(columns=X_encode.columns, fill_value=0)
                                prediction = selected_model.predict(user_input1_encode)
                                st.write(f" the predict model is :{prediction[0]}")
                                st.success("🚀 Boom! Here's your final prediction!")
                    if algorithm == "Multi Class":
                        r1 = st.selectbox("choose the algorthim:",
                                          ["Desicion tree Classifier", "Random Forest Classifier", "Xgclassifier",
                                           "Knn Classifier"])
                        X_encode = pd.get_dummies(X)
                        original_columns = X_encode.columns
                        imputer = SimpleImputer(strategy="mean")
                        X_encoded_imputed = imputer.fit_transform(X_encode)
                        X_encode = pd.DataFrame(X_encoded_imputed, columns=original_columns)
                        if X_encode.isna().sum().sum() > 0:
                            st.write("NaNs found in feature matrix after imputation!")
                            X_encode = X_encode.fillna(0)
                        X_encode_train, X_encode_test, Y_train, Y_test = train_test_split(
                            X_encode, Y, test_size=0.2, train_size=0.8, random_state=42
                        )
                        if Y_train.isna().sum() > 0:
                            st.write("NaNs found in target data! Showing rows with NaN values:")
                            rows_with_nan = Y_train[Y_train.isna()]
                            Y_train = Y_train.dropna()
                            X_encode_train = X_encode_train.loc[Y_train.index]
                        if r1 == "Desicion tree Classifier":
                            desicion_model1 = DecisionTreeClassifier(class_weight="balanced", random_state=42)
                            desicion_model1.fit(X_encode_train, Y_train)
                            des_pred = desicion_model1.predict(X_encode_test)
                            Y_test_str = Y_test.astype(str)
                            des_pred_str = des_pred.astype(str)
                            unique_labels = sorted(set(Y_test_str) | set(des_pred_str))
                            desicion_precision = precision_score(Y_test_str, des_pred_str, average="weighted",
                                                                 labels=unique_labels, zero_division=0)
                            desicion_recall = recall_score(Y_test_str, des_pred_str, average="weighted",
                                                           labels=unique_labels,
                                                           zero_division=0)

                            st.write(f"Precision: {desicion_precision}")
                            st.write(f"Recall: {desicion_recall}")

                            selected_model = desicion_model1
                            user_input2 = {}
                            for feature in selected_feature:
                                user_input2[feature] = st.text_input(f"enter the value{feature}:")
                            if all(value != "" for value in user_input2.values()):
                                user_input2_df = pd.DataFrame(user_input2, index=[0])
                                user_input2_encode = pd.get_dummies(user_input2_df)
                                user_input2_encode = user_input2_encode.reindex(columns=X_encode.columns, fill_value=0)
                                prediction = selected_model.predict(user_input2_encode)
                                st.write(f" the predict model is :{prediction[0]}")
                                st.success("🚀 Boom! Here's your final prediction!")

                        elif r1 == "Random Forest Classifier":
                            randomforest = RandomForestClassifier(class_weight="balanced", random_state=42)
                            randomforest.fit(X_encode_train, Y_train)
                            rand_pred = randomforest.predict(X_encode_test)
                            Y_test_str = Y_test.astype(str)
                            rand_pred_str = rand_pred.astype(str)
                            unique_labels = sorted(set(Y_test_str) | set(rand_pred_str))
                            rand_presicion = precision_score(Y_test_str, rand_pred.astype(str), average="weighted",
                                                             labels=unique_labels, zero_division=0)
                            rand_recall = recall_score(Y_test_str, rand_pred.astype(str), average="weighted",
                                                       labels=unique_labels, zero_division=0)
                            st.write(f"Precision: {rand_presicion}")
                            st.write(f"Recall: {rand_recall}")

                            selected_model = randomforest
                            user_input3 = {}
                            for feature in selected_feature:
                                user_input3[feature] = st.text_input(f"enter the value{feature}:")
                            if all(value != "" for value in user_input3.values()):
                                user_input3_df = pd.DataFrame(user_input3, index=[0])
                                user_input3_encode = pd.get_dummies(user_input3_df)
                                user_input3_encode = user_input3_encode.reindex(columns=X_encode.columns, fill_value=0)
                                prediction = selected_model.predict(user_input3_encode)
                                st.write(f" the predict model is :{prediction[0]}")
                                st.success("🚀 Boom! Here's your final prediction!")

                        elif r1 == "Knn Classifier":
                            knnmodel = KNeighborsClassifier()
                            knnmodel.fit(X_encode_train, Y_train)
                            knn_pred = knnmodel.predict(X_encode_test)
                            Y_test_str = Y_test.astype(str)
                            knn_pred_str = knn_pred.astype(str)
                            unique_labels = sorted(set(Y_test_str) | set(knn_pred_str))
                            knn_presicion = precision_score(Y_test_str, knn_pred.astype(str), average="weighted",
                                                            labels=unique_labels, zero_division=0)
                            knn_recall = recall_score(Y_test_str, knn_pred.astype(str), average="weighted",
                                                      labels=unique_labels, zero_division=0)
                            st.write(f"Precision: {knn_presicion}")
                            st.write(f"Recall: {knn_recall}")

                            selected_model = knnmodel
                            user_input4 = {}
                            for feature in selected_feature:
                                user_input4[feature] = st.text_input(f"enter the value{feature}:")
                            if all(value != "" for value in user_input4.values()):
                                user_input4_df = pd.DataFrame(user_input4, index=[0])
                                user_input4_encode = pd.get_dummies(user_input4_df)
                                user_input4_encode = user_input4_encode.reindex(columns=X_encode.columns, fill_value=0)
                                prediction = selected_model.predict(user_input4_encode)
                                st.write(f" the predict model is :{prediction[0]}")
                                st.success("🚀 Boom! Here's your final prediction!")
                        elif r1 == "Xgclassifier":
                            if Y_test.isna().sum() > 0:
                                st.write("NaNs found in target data! Imputing NaNs with mode.")
                                mode_value = Y_test.mode()[0]
                                Y_test.fillna(mode_value, inplace=True)
                            label_encoder = LabelEncoder()
                            Y_train_encoded = label_encoder.fit_transform(Y_train)
                            Y_test_encoded = label_encoder.transform(Y_test)
                            Xg_model1 = XGBClassifier()
                            Xg_model1.fit(X_encode_train, Y_train_encoded)
                            Xg_pred = Xg_model1.predict(X_encode_test)
                            Xg_pred_str = label_encoder.inverse_transform(Xg_pred)
                            Y_test_str = label_encoder.inverse_transform(Y_test_encoded)
                            unique_labels = sorted(set(Y_test_str) | set(Xg_pred_str))
                            decision_precision = precision_score(Y_test_str, Xg_pred_str, average="weighted",
                                                                 labels=unique_labels, zero_division=0)
                            decision_recall = recall_score(Y_test_str, Xg_pred_str, average="weighted",
                                                           labels=unique_labels,
                                                           zero_division=0)
                            st.write(f"Precision: {decision_precision}")
                            st.write(f"Recall: {decision_recall}")

                            selected_model = Xg_model1
                            user_input5 = {}
                            for feature in selected_feature:
                                user_input5[feature] = st.text_input(f"enter the value{feature}:")
                            if all(value != "" for value in user_input5.values()):
                                user_input5_df = pd.DataFrame(user_input5, index=[0])
                                user_input5_encode = pd.get_dummies(user_input5_df)
                                user_input5_encode = user_input5_encode.reindex(columns=X_encode.columns, fill_value=0)
                                prediction = selected_model.predict(user_input5_encode)
                                st.write(f" the predict model is :{prediction[0]}")
                                st.success("🚀 Boom! Here's your final prediction!")
                    if algorithm == "Linear Model":
                        model_choice = st.selectbox("Choose the algorithm:",
                                                    ["Linear Regression", "Polynomial Regression", "Lasso Regression",
                                                     "Ridge Regression"])
                        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, train_size=0.2)
                        if model_choice == "Linear Regression":
                            linearmodel = LinearRegression()
                            linearmodel.fit(X_train, Y_train)
                            linear_y_pred = linearmodel.predict(X_test)
                            mae = mean_absolute_error(Y_test, linear_y_pred)
                            mse = mean_squared_error(Y_test, linear_y_pred)
                            r2_score_value = r2_score(Y_test, linear_y_pred)

                            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                            st.write(f"R-squared Score (R²): {r2_score_value:.2f}")
                            selected_model = linearmodel
                            user_input6 = {}
                            for feature in selected_feature:
                                user_input6[feature] = st.text_input(f"Enter the value for {feature}:")
                            if all(value != "" for value in user_input6.values()):
                                user_input6_df = pd.DataFrame(user_input6, index=[0])
                                user_input6_df = user_input6_df.apply(pd.to_numeric, errors='coerce')
                                user_input6_df = user_input6_df.reindex(columns=X_train.columns, fill_value=0)
                                prediction = selected_model.predict(user_input6_df)
                                st.write(f"The predicted {target_column1}  is: {prediction[0]:,.2f}")
                                st.success("🚀 Boom! Here's your final prediction!")
                        if model_choice == "Polynomial Regression":
                            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8, train_size=0.2)
                            degree = 3
                            poly = PolynomialFeatures(degree=degree)
                            poly_Xtrain = poly.fit_transform(X_train)
                            poly_Xtest = poly.transform(X_test)
                            poly_model = LinearRegression()
                            poly_model.fit(poly_Xtrain, Y_train)
                            Y_pred = poly_model.predict(poly_Xtest)
                            mse = mean_squared_error(Y_test, Y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(Y_test, Y_pred)
                            st.write(f"Mean Absolute Error (MAE): {mse:.2f}")
                            st.write(f" root Mean Squared Error (MSE): {rmse:.2f}")
                            st.write(f"R-squared Score (R²): {r2:.2f}")
                            selected_model = poly_model
                            user_input7 = {}
                            for feature in selected_feature:
                                user_input7[feature] = st.text_input(f"enter the value{feature}:")
                                if all(value != "" for value in user_input7.values()):
                                    user_input7_df = pd.DataFrame(user_input7, index=[0])
                                    user_input_7_df = user_input7_df.apply(pd.to_numeric, errors='coerce')
                                    user_input7_transform = poly.transform(user_input7_df)
                                    prediction = selected_model.predict(user_input7_transform)
                                    st.write(f" the predicted{target_column1}:{prediction[0]:,.2f}")
                                    st.success("🚀 Boom! Here's your final prediction!")
                        if model_choice == "Lasso Regression":
                            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)
                            ridge_model = Lasso(alpha=1.0)
                            ridge_model.fit(X_train, Y_train)
                            Y_pred = ridge_model.predict(X_test)
                            mse = mean_squared_error(Y_test, Y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(Y_test, Y_pred)
                            st.write(f" the mean squared error {mse:.2f}")
                            st.write(f" the root mean squared error {rmse:.2f}")
                            st.write(f" the r2 score is {r2:.2f}")
                            selected_model = ridge_model
                            user_input8 = {}
                            for feature in selected_feature:
                                user_input8[feature] = st.text_input(f" enter the value of {feature}")
                                if all(value != "" for value in user_input8.values()):
                                    user_input8_df = pd.DataFrame(user_input8, index=[0])
                                    user_input8_df = user_input8_df.apply(pd.to_numeric, errors='coerce')
                                    user_input8_df = user_input8_df.reindex(columns=X_train.columns, fill_value=0)
                                    prediction = selected_model.predict(user_input8_df)
                                    st.write(f" the predicted{target_column1} value is :{prediction[0]:,.2f}")
                                    st.success("🚀 Boom! Here's your final prediction!")
                        if model_choice == "Ridge Regression":
                            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, train_size=0.8)
                            ridge_model = Ridge(alpha=1.0)
                            ridge_model.fit(X_train, Y_train)
                            Y_pred = ridge_model.predict(X_test)
                            mse = mean_squared_error(Y_test, Y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(Y_test, Y_pred)
                            st.write(f" the mean squared error {mse:.2f}")
                            st.write(f" the root mean squared error {rmse:.2f}")
                            st.write(f" the r2 score is {r2:.2f}")
                            selected_model = ridge_model
                            user_input9 = {}
                            for feature in selected_feature:
                                user_input9[feature] = st.text_input(f" enter the value of {feature}")
                                if all(value != "" for value in user_input9.values()):
                                    user_input9_df = pd.DataFrame(user_input9, index=[0])
                                    user_input9_df = user_input9_df.apply(pd.to_numeric, errors='coerce')
                                    user_input9_df = user_input9_df.reindex(columns=X_train.columns, fill_value=0)
                                    prediction = selected_model.predict(user_input9_df)
                                    st.write(f" the predicted{target_column1} value is :{prediction[0]:,.2f}")
                                    st.success("🚀 Boom! Here's your final prediction!")
                    if algorithm == "Non linear Model":
                        choices = st.selectbox("Choose the algorthim:",
                                               ["Desicion Tree Regresssor", " Random forest Regressor", "knn Regressor",
                                                "Xg boost regressor "])
                        if choices == "Desicion Tree Regresssor":
                            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
                            des_model = DecisionTreeRegressor()
                            des_model.fit(X_train, Y_train)
                            Y_pred = des_model.predict(X_test)
                            mae = mean_absolute_error(Y_test, Y_pred)
                            mse = mean_squared_error(Y_test, Y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(Y_test, Y_pred)
                            st.write(f" the mean squared error{mse:.2f}")
                            st.write(f" the root  mean squared error {rmse:.2f}")
                            st.write(f" the adjusted r2 score {r2:.2f}")
                            selected_model = des_model
                            user_input10 = {}
                            for feature in selected_feature:
                                user_input10[feature] = st.text_input(f" enter the value of {feature}")
                                if all(value != "" for value in user_input10.values()):
                                    user_input10_df = pd.DataFrame(user_input10, index=[0])
                                    user_input10_df = user_input10_df.apply(pd.to_numeric, errors='coerce')
                                    user_input10_df = user_input10_df.reindex(columns=X_train.columns)
                                    prediction = selected_model.predict(user_input10_df)
                                    st.write(f"the prediction value{target_column1}:{prediction[0]:,.2f}")
                                    st.success("🚀 Boom! Here's your final prediction!")
                        if choices == " Random forest Regressor":
                            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
                            reg_model = RandomForestRegressor()
                            reg_model.fit(X_train, Y_train)
                            Y_pred = reg_model.predict(X_test)
                            mae = mean_absolute_error(Y_test, Y_pred)
                            mse = mean_squared_error(Y_test, Y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(Y_test, Y_pred)
                            st.write(f" the mean squared error{mse:.2f}")
                            st.write(f" the root  mean squared error {rmse:.2f}")
                            st.write(f" the adjusted r2 score {r2:.2f}")
                            selected_model = reg_model
                            user_input11 = {}
                            for feature in selected_feature:
                                user_input11[feature] = st.text_input(f" enter the value of {feature}")
                                if all(value != "" for value in user_input11.values()):
                                    user_input11_df = pd.DataFrame(user_input11, index=[0])
                                    user_input11_df = user_input11_df.apply(pd.to_numeric, errors='coerce')
                                    user_input11_df = user_input11_df.reindex(columns=X_train.columns)
                                    prediction = selected_model.predict(user_input11_df)
                                    st.write(f"the prediction value{target_column1}:{prediction[0]:,.2f}")
                                    st.success("🚀 Boom! Here's your final prediction!")
                        if choices == "knn Regressor":
                            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
                            knn1_model = KNeighborsRegressor()
                            knn1_model.fit(X_train, Y_train)
                            Y_pred = knn1_model.predict(X_test)
                            mae = mean_absolute_error(Y_test, Y_pred)
                            mse = mean_squared_error(Y_test, Y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(Y_test, Y_pred)
                            st.write(f" the mean squared error{mse:.2f}")
                            st.write(f" the root  mean squared error {rmse:.2f}")
                            st.write(f" the adjusted r2 score {r2:.2f}")
                            selected_model = knn1_model
                            user_input12 = {}
                            for feature in selected_feature:
                                user_input12[feature] = st.text_input(f" enter the value of {feature}")
                                if all(value != "" for value in user_input12.values()):
                                    user_input12_df = pd.DataFrame(user_input12, index=[0])
                                    user_input12_df = user_input12_df.apply(pd.to_numeric, errors='coerce')
                                    user_input12_df = user_input12_df.reindex(columns=X_train.columns)
                                    prediction = selected_model.predict(user_input12_df)
                                    st.write(f"the prediction value{target_column1}:{prediction[0]:,.2f}")
                                    st.success("🚀 Boom! Here's your final prediction!")
                        if choices == "Xg boost regressor ":
                            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
                            XGB1_model = XGBRegressor()
                            XGB1_model.fit(X_train, Y_train)
                            Y_pred = XGB1_model.predict(X_test)
                            mae = mean_absolute_error(Y_test, Y_pred)
                            mse = mean_squared_error(Y_test, Y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(Y_test, Y_pred)
                            st.write(f" the mean squared error{mse:.2f}")
                            st.write(f" the root mean squared error {rmse:.2f}")
                            st.write(f" the adjusted r2 score {r2:.2f}")
                            selected_model = XGB1_model
                            user_input13 = {}
                            for feature in selected_feature:
                                user_input13[feature] = st.text_input(f" enter the value of {feature}")
                                if all(value != "" for value in user_input13.values()):
                                    user_input13_df = pd.DataFrame(user_input13, index=[0])
                                    user_input13_df = user_input13_df.apply(pd.to_numeric, errors='coerce')
                                    user_input13_df = user_input13_df.reindex(columns=X_train.columns)
                                    prediction = selected_model.predict(user_input13_df)
                                    st.write(f"the prediction value{target_column1}:{prediction[0]:,.2f}")
                                    st.success("🚀 Boom! Here's your final prediction!")
