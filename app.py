
# ============================================================
# BANK MARKETING PREDICTION - STREAMLIT APPLICATION
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🏦 Bank Marketing Prediction System")

st.markdown(
    """
    This application predicts whether a bank customer will
    subscribe to a **Bank Term Deposit** using Machine Learning.

    Upload a test CSV file, select a model, and generate
    predictions along with evaluation metrics.
    """
)

st.divider()


# ============================================================
#  LOAD TRAINED MODELS AND PREPROCESSING OBJECTS
# ============================================================

log_model = joblib.load(
    "model/logistic_regression.pkl"
)

dt_model = joblib.load(
    "model/decision_tree.pkl"
)

knn_model = joblib.load(
    "model/knn.pkl"
)

nb_model = joblib.load(
    "model/naive_bayes.pkl"
)

rf_model = joblib.load(
    "model/random_forest.pkl"
)

scaler = joblib.load(
    "model/scaler.pkl"
)

label_encoders = joblib.load(
    "model/label_encoders.pkl"
)


# ============================================================
# EXPECTED INPUT FEATURES
# ============================================================

expected_columns = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]


# ============================================================
#  MODEL SELECTION
# ============================================================

st.sidebar.title("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "k-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest"
    ]
)

st.sidebar.success(
    f"Selected Model : {selected_model}"
)

st.sidebar.markdown("---")


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)


# ============================================================
# PROCESS UPLOADED FILE
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # READ THE UPLOADED CSV
        # ----------------------------------------------------

        data = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python"
        )


        # ----------------------------------------------------
        # KEEP ORIGINAL DATA UNCHANGED
        # ----------------------------------------------------

        original_data = data.copy()


        # ----------------------------------------------------
        # DISPLAY ORIGINAL DATA
        # ----------------------------------------------------

        st.subheader("Uploaded Dataset")

        st.info(
            "The table below shows the original uploaded data. "
            "Categorical values are kept in their original form."
        )

        st.dataframe(
            original_data.head(10),
            use_container_width=True
        )


        # ----------------------------------------------------
        # SHOW DATASET INFORMATION
        # ----------------------------------------------------

        st.write(
            f"Dataset shape: **{original_data.shape[0]} rows × "
            f"{original_data.shape[1]} columns**"
        )


        # ====================================================
        #  KEEP TARGET FOR EVALUATION IF AVAILABLE
        # ====================================================

        if "y" in original_data.columns:

            y_original = original_data["y"].copy()

        else:

            y_original = None


        # ====================================================
        #  CREATE SEPARATE DATA FOR ML PROCESSING
        # ====================================================

        prediction_data = original_data.copy()


        # ----------------------------------------------------
        # REMOVE TARGET COLUMN
        # ----------------------------------------------------

        if "y" in prediction_data.columns:

            prediction_data = prediction_data.drop(
                columns=["y"]
            )


        # ====================================================
        #  CHECK REQUIRED FEATURES
        # ====================================================

        missing_columns = [
            column
            for column in expected_columns
            if column not in prediction_data.columns
        ]

        if len(missing_columns) > 0:

            st.error(
                "The uploaded CSV is missing the following "
                "required columns:"
            )

            st.write(missing_columns)

            st.info(
                "Please upload a Bank Marketing test CSV "
                "containing the 20 input features."
            )

            st.stop()


       

        extra_columns = [
            column
            for column in prediction_data.columns
            if column not in expected_columns
        ]

        if len(extra_columns) > 0:

            st.warning(
                "Extra columns were detected. "
                "They will not be used for prediction:"
            )

            st.write(extra_columns)


        # ====================================================
        # SELECT FEATURES IN TRAINING ORDER
        # ====================================================

        prediction_data = prediction_data[
            expected_columns
        ].copy()


        # ====================================================
        # APPLY THE SAME LABEL ENCODING USED DURING TRAINING
        # ====================================================


        for column, encoder in label_encoders.items():

            if column not in prediction_data.columns:
                continue


            # Convert values to strings because the encoders
            # were trained on string categorical values.

            column_values = (
                prediction_data[column]
                .astype(str)
                .str.strip()
            )


            # Classes learned during training

            known_classes = set(
                encoder.classes_
            )


            # Find categories not seen during training

            unknown_values = sorted(
                set(column_values) - known_classes
            )


            if len(unknown_values) > 0:

                st.warning(
                    f"Unknown value(s) found in '{column}': "
                    f"{unknown_values[:5]}. "
                    "These values will be replaced by the "
                    "first known training category."
                )

                replacement_value = (
                    encoder.classes_[0]
                )

                column_values = column_values.apply(
                    lambda value:
                    value
                    if value in known_classes
                    else replacement_value
                )


            # Apply the SAME encoder used during training

            prediction_data[column] = (
                encoder.transform(
                    column_values
                )
            )


        # ====================================================
        # CONVERT NUMERIC FEATURES TO NUMERIC TYPE
        # ====================================================

        for column in expected_columns:

            if column not in label_encoders:

                prediction_data[column] = pd.to_numeric(
                    prediction_data[column],
                    errors="coerce"
                )


        # ====================================================
        # HANDLE MISSING / INVALID VALUES
        # ====================================================

        if prediction_data.isnull().sum().sum() > 0:

            st.warning(
                "Missing or invalid numeric values were detected. "
                "They will be replaced with 0 for prediction."
            )

            prediction_data = prediction_data.fillna(0)


        # ====================================================
        # SCALE USING THE SAVED TRAINING SCALER
        # ====================================================

        prediction_data_scaled = scaler.transform(
            prediction_data
        )


        # ====================================================
        #  SELECT TRAINED MODEL
        # ====================================================

        if selected_model == "Logistic Regression":

            model = log_model

        elif selected_model == "Decision Tree":

            model = dt_model

        elif selected_model == "k-Nearest Neighbors":

            model = knn_model

        elif selected_model == "Naive Bayes":

            model = nb_model

        else:

            model = rf_model


        # ====================================================
        # PREDICTION BUTTON
        # ====================================================

        if st.button(
            "Predict",
            type="primary"
        ):

            # ------------------------------------------------
            # Generate predictions
            # ------------------------------------------------

            # Models trained with scaled features
            if selected_model in [
                "Logistic Regression",
                "k-Nearest Neighbors",
                "Naive Bayes"
            ]:
                model_input = prediction_data_scaled

            # Models trained with unscaled encoded features
            else:
                model_input = prediction_data_scaled

            predictions = model.predict(model_input)

            # ------------------------------------------------
            # Convert predictions to Yes / No
            # ------------------------------------------------

            prediction_labels = np.where(
                predictions == 1,
                "Yes",
                "No"
            )


            # =================================================
            #  CREATE RESULT USING ORIGINAL DATA
            # =================================================

            result = original_data.copy()

            result["Prediction"] = (
                prediction_labels
            )


            # =================================================
            #  DISPLAY PREDICTION RESULTS
            # =================================================

            st.subheader(
                "Prediction Results"
            )

            st.dataframe(
                result,
                use_container_width=True
            )


            # =================================================
            #  PREDICTION SUMMARY
            # =================================================

            yes_count = np.sum(
                prediction_labels == "Yes"
            )

            no_count = np.sum(
                prediction_labels == "No"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Predicted Yes",
                    int(yes_count)
                )


            with col2:

                st.metric(
                    "Predicted No",
                    int(no_count)
                )


            # =================================================
            # EVALUATION
            # =================================================

            if y_original is not None:

                # ------------------------------------------------
                # Convert actual target to 0 / 1
                # ------------------------------------------------

                y_text = (
                    y_original
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )


                y_true = y_text.map({
                    "no": 0,
                    "yes": 1
                })


                # ------------------------------------------------
                # Handle numeric 0/1 target if supplied
                # ------------------------------------------------

                if y_true.isnull().any():

                    y_true = pd.to_numeric(
                        y_original,
                        errors="coerce"
                    )


                if y_true.isnull().any():

                    st.error(
                        "The target column 'y' contains values "
                        "that could not be interpreted as No/Yes "
                        "or 0/1."
                    )

                    st.stop()


                y_true = y_true.astype(int).values


                # =================================================
                # CALCULATE METRICS
                # =================================================

                accuracy = accuracy_score(
                    y_true,
                    predictions
                )

                precision = precision_score(
                    y_true,
                    predictions,
                    zero_division=0
                )

                recall = recall_score(
                    y_true,
                    predictions,
                    zero_division=0
                )

                f1 = f1_score(
                    y_true,
                    predictions,
                    zero_division=0
                )

                mcc = matthews_corrcoef(
                    y_true,
                    predictions
                )


                # ------------------------------------------------
                # AUC
                # ------------------------------------------------

                try:

                    if hasattr(
                        model,
                        "predict_proba"
                    ):

                        probabilities = (
                            model
                            .predict_proba(
                                model_input
                            )[:, 1]
                        )

                    else:

                        probabilities = (
                            model
                            .decision_function(
                                model_input
                            )
                        )


                    auc = roc_auc_score(
                        y_true,
                        probabilities
                    )

                except Exception:

                    auc = np.nan


                # =================================================
                #DISPLAY METRICS
                # =================================================

                st.subheader(
                    "Model Evaluation Metrics"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Accuracy",
                        f"{accuracy:.4f}"
                    )


                with col2:

                    if np.isnan(auc):

                        st.metric(
                            "AUC",
                            "N/A"
                        )

                    else:

                        st.metric(
                            "AUC",
                            f"{auc:.4f}"
                        )


                with col3:

                    st.metric(
                        "Precision",
                        f"{precision:.4f}"
                    )


                col4, col5, col6 = st.columns(3)


                with col4:

                    st.metric(
                        "Recall",
                        f"{recall:.4f}"
                    )


                with col5:

                    st.metric(
                        "F1 Score",
                        f"{f1:.4f}"
                    )


                with col6:

                    st.metric(
                        "MCC",
                        f"{mcc:.4f}"
                    )


                # =================================================
                #  CONFUSION MATRIX
                # =================================================

                st.subheader(
                    "Confusion Matrix"
                )


                cm = confusion_matrix(
                    y_true,
                    predictions
                )


                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        "Actual No",
                        "Actual Yes"
                    ],
                    columns=[
                        "Predicted No",
                        "Predicted Yes"
                    ]
                )


                st.dataframe(
                    cm_df,
                    use_container_width=True
                )


                # =================================================
                # CLASSIFICATION REPORT
                # =================================================

                st.subheader(
                    "Classification Report"
                )


                report = classification_report(
                    y_true,
                    predictions,
                    target_names=[
                        "No",
                        "Yes"
                    ],
                    output_dict=True,
                    zero_division=0
                )


                report_df = (
                    pd.DataFrame(report)
                    .transpose()
                    .round(4)
                )


                st.dataframe(
                    report_df,
                    use_container_width=True
                )


            else:

                st.info(
                    "The uploaded file does not contain a 'y' "
                    "column. Predictions are available, but "
                    "evaluation metrics cannot be calculated."
                )


            # =================================================
            #  DOWNLOAD ORIGINAL DATA + PREDICTION
            # =================================================

            csv = result.to_csv(
                index=False,
                sep=";"
            ).encode("utf-8")


            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="bank_marketing_predictions.csv",
                mime="text/csv"
            )


            # =================================================
            #  SUCCESS MESSAGE
            # =================================================

            st.success(
                f"Prediction completed successfully using "
                f"{selected_model}!"
            )


    except Exception as e:

        st.error(
            "An error occurred while processing the uploaded file."
        )

        st.exception(e)
