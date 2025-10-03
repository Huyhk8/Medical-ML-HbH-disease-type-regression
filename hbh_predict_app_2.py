import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

HbH = pd.DataFrame({
    "age": [30, 74, 21, 68, 33, 33, 69, 63, 18, 31, 
63, 44, 49, 58, 68, 65, 56, 5, 68, 30, 17, 8, 40, 8, 67, 82, 
71, 64, 29, 51, 10, 18, 44, 57, 36, 16, 19, 7, 35, 37, 58, 30, 
60, 24, 57, 30, 71, 30, 76, 80, 50, 40, 43, 57, 35, 11, 15, 80, 
63, 31, 65, 57, 64, 64, 69, 9, 38, 41, 45, 31, 36, 19],
    "sex": [2, 
2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 
2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 
1, 1, 2, 1, 2, 2, 2, 2],
    "transfusion": [7, 12, 7, 3, 8, 1, 10, 7, 1, 1, 1, 2, 4, 
    1, 4, 6, 3, 3, 4, 3, 1, 1, 3, 3, 4, 4, 7, 9, 2, 0, 4, 4, 
    1, 3, 2, 5, 4, 2, 4, 1, 3, 1, 1, 5, 5, 3, 4, 3, 1, 3, 1, 
    0, 4, 2, 6, 1, 0, 4, 2, 3, 0, 2, 1, 0, 1, 3, 2, 0, 1, 2, 
    1, 4],
    "RBC": [3.48, 3.81, 3, 3.89, 2.99, 3.59, 3.11, 4.16, 
    4.93, 2.83, 3.33, 5.12, 3.13, 3.68, 3.61, 3.72, 4.2, 2.94, 
    3.27, 3.13, 3.44, 4.39, 2.61, 3.9, 4.1, 3.67, 3.7, 4.2, 3.31, 
    5.25, 3.97, 3.98, 3.35, 4.98, 4.22, 3.03, 3.81, 5.04, 4.3, 
    4.44, 3.42, 3.42, 5.07, 3.82, 2.62, 3.96, 3.83, 3.97, 4.63, 
    4.05, 4.46, 6.14, 3.64, 4.01, 3.39, 5.05, 5.52, 4.3, 3.97, 
    3.53, 4.14, 4.68, 5.16, 4.65, 4.17, 3.83, 5.44, 5.63, 4.69, 
    4.64, 3.34, 3.71],
    "Hb": [82, 86, 69, 82, 74, 71, 69, 76, 
    98, 73, 62, 101, 59, 72, 75, 77, 88, 65, 65, 67, 72, 88, 
    52, 73, 80, 75, 80, 86, 67, 93, 85, 75, 79, 89, 71, 56, 61, 
    74, 92, 83, 72, 78, 83, 74, 58, 83, 83, 77, 77, 89, 84, 106, 
    73, 83, 68, 91, 90, 87, 79, 71, 87, 88, 86, 93, 83, 78, 88, 
    100, 89, 87, 72, 78],
    "MCV": [84.2, 78.7, 82.3, 77.4, 83.9, 
    81.6, 79.1, 66.1, 77.5, 86.2, 69.1, 70.5, 73.8, 75, 74.5, 
    73.4, 69.4, 74.1, 73.7, 81.2, 86.9, 75.6, 87, 77.9, 66.3, 
    70.6, 75.9, 73.8, 79.8, 63, 78.3, 65.8, 85.4, 62.4, 58.3, 
    76.2, 68.8, 51.4, 73, 70.7, 74.3, 90.4, 56.4, 78, 79.4, 73, 
    76.2, 78.3, 56.8, 79.8, 65.5, 62.2, 81.9, 75.8, 80.2, 70.5, 
    56.3, 67.9, 69.3, 79.6, 74.9, 60.7, 58.5, 73.5, 72.2, 75.7, 
    55.7, 62.7, 70.8, 65.3, 82.9, 78.7],
    "MCH": [23.6, 22.6, 
    23, 20.6, 24.7, 19.8, 22.2, 18.3, 19.9, 25.8, 18.6, 19.7, 
    18.8, 19.6, 20.8, 20.7, 20.7, 22.1, 19.9, 21.4, 20.9, 20, 
    19.9, 18.7, 19.5, 20.4, 21.6, 20.5, 20.2, 17.1, 21.4, 18.8, 
    23.5, 17.9, 16.8, 18.5, 16, 14.7, 21.4, 18.7, 21.1, 22.8, 
    16.4, 19.4, 22.1, 21, 21.7, 19.4, 16.6, 22, 18.8, 17.3, 20.1, 
    20.7, 20.1, 18, 16.3, 20.2, 19.9, 20.1, 21, 18.8, 16.7, 20, 
    19.9, 20.4, 16.2, 17.8, 19, 18.8, 21.6, 21],
    "RDWCV": [18.6, 
    21.3, 19, 23.8, 20.8, 23.2, 20.5, 29.3, 22.1, 25.9, 24.3, 
    22.1, 24.4, 24.1, 23.5, 26.9, 30.5, 26.9, 26.3, 30.4, 29, 
    23.8, 21.6, 23.9, 27.4, 25.6, 23, 25.4, 25.3, 23.7, 20.2, 
    30.9, 21.3, 27, 25, 31.9, 30.3, 25.4, 25.9, 25.5, 23.1, 24.8, 
    23.4, 29.2, 20.8, 23.9, 28.2, 23.8, 24.9, 20.9, 31.2, 22.9, 
    20.1, 22.3, 22.1, 23.1, 26.5, 27.9, 29.1, 23.7, 21.3, 28.9, 
    25, 23.9, 23.2, 24, 28.2, 26.4, 22.6, 26.9, 22.5, 24],
    "Ferritin": [534.3, 
    11931, 2444, 1878, 1222, 590, 10691, 3939, 1553, 532, 2135, 
    1128, 6795, 1228, 2119, 3915, 1644, 1617, 4621, 1162, 2228, 
    1896, 1197, 140, 1063, 2365, 1530, 11104, 2591, 547.7, 992, 
    1392, 7041, 786, 2455, 244, 57028, 204.1, 849, 3251, 1533, 
    343.6, 651.7, 1516, 3240, 2615, 1127, 5537, 1219, 3444, 919, 
    1402, 4903, 1079, 7672, 349, 35, 1158, 1053, 1397, 620, 833.6, 
    435, 2604, 1353, 315, 3627, 1320, 1510, 578, 1424, 200],
    "HbHa": [30, 10.1, 12.1, 13.7, 21.7, 26.1, 9.7, 0, 28.8, 
    4.2, 27, 9.6, 10.3, 27.1, 27.6, 7.6, 3.1, 0, 8.8, 9, 7.1, 
    10.3, 33.2, 18.3, 2.7, 7, 7.5, 3.7, 18.3, 9.9, 13, 0, 32.5, 
    2.2, 0, 9.6, 0, 0, 10, 15.7, 9.6, 12.1, 2.8, 8.1, 7, 9.4, 
    11.9, 22.7, 3.4, 7.4, 10.9, 6.9, 7.7, 14.2, 9.1, 24.8, 4.4, 
    9.2, 8.3, 12.2, 8.2, 1.9, 3.2, 9.8, 14.8, 15.5, 3.6, 10, 
    14.1, 10.6, 30.6, 12.3],
    "Bart": [1.8, 0.5, 0.8, 0, 2, 2.7, 
    0, 1.1, 3, 0, 0, 2.9, 0.6, 0.9, 5.3, 0.8, 0, 0.9, 0, 2.6, 
    2.8, 1.7, 1.5, 4.6, 0.7, 0.4, 0, 0, 1.5, 0.7, 0, 2.8, 3.4, 
    1.2, 4.3, 1.6, 1.4, 3.1, 1.3, 1.3, 0.7, 1.5, 0.9, 2.1, 1.3, 
    0, 3.7, 1.1, 0, 0.8, 2.1, 0, 1.3, 1, 0, 1.2, 0, 1.4, 1.1, 
    2.3, 1, 0, 0, 1.1, 0, 0.9, 0, 0, 0, 2, 1.5, 5.6],
    "HbA": [67.7, 
    87.6, 83.7, 85, 72.6, 70.7, 87.6, 88, 67.7, 93.8, 72.5, 86.6, 
    87.2, 70.8, 66.7, 89.5, 95.5, 86.4, 89.9, 85.1, 86.7, 84.4, 
    64.9, 72.7, 94.6, 91.8, 91.3, 89.1, 76.4, 88.5, 81.8, 77.2, 
    63.7, 95.5, 77.2, 80.5, 79.8, 79.5, 87.9, 80.4, 88.9, 85.5, 
    95, 86.8, 86.7, 89.7, 83.1, 75.3, 95.4, 91, 86.6, 92.2, 88.1, 
    84.1, 87.5, 73.5, 94.6, 88.5, 89.6, 81.7, 89.6, 96.7, 95.8, 
    88.4, 84.6, 80.2, 95.1, 88.8, 85.1, 86.5, 67.3, 78],
    "HbF": [0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0, 4.3, 0, 
    0, 0, 0, 0, 0.3, 0.6, 0, 0, 0, 0, 0, 0, 5.9, 0, 0, 2, 0, 
    2.5, 2.1, 0, 0, 0, 0, 0, 0, 0.6, 0, 0.6, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6],
    "HbE": [0, 0, 0, 0, 0, 0, 0.8, 7.3, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.4, 0, 0, 0, 10.6, 
    0, 0, 12.4, 5, 12.1, 12, 0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0],
    "HbA2": [0.5, 1.8, 1.4, 1.3, 0.7, 0.5, 1.9, 
    2.1, 0.5, 2, 0.5, 0.9, 1.3, 0.6, 0.4, 1.8, 1.4, 5.5, 1.3, 
    1.3, 1.3, 1, 0.4, 0.6, 1.1, 0.8, 1.2, 1.8, 0.7, 0.9, 2.9, 
    1.7, 0.4, 1.1, 2, 1.3, 2.4, 2.1, 0.8, 0.8, 1.2, 0.9, 1.3, 
    1.5, 2.2, 0.9, 0.7, 0.9, 1.2, 0.8, 0.4, 0, 0.9, 0.7, 1.6, 
    0.5, 1, 0.9, 1, 0.4, 1.2, 1.4, 1, 0.7, 0.6, 0.7, 1.3, 1, 
    0.8, 0.9, 0.6, 0.7],
    "HbC": [0, 0, 2, 0, 3, 0, 0, 1.5, 0, 
    0, 0, 0, 0.6, 0, 0, 0, 0, 2.9, 0, 2, 2.1, 2.3, 0, 3.5, 0, 
    0, 0, 0, 3.1, 0, 0, 0, 0, 0, 2.1, 2, 1.8, 1.2, 0, 1.8, 0, 
    0, 0, 1.5, 1.4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1.8, 0, 0, 0, 
    0, 3.4, 0, 0, 0, 0, 0, 2.7, 0, 0, 0, 0, 0, 2.8],
    "class2": [1, 
    1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 
    1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 
    1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 
    0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]  # biến mục tiêu
})

# Tách X (đặc trưng đầu vào) và y (biến mục tiêu)
X_data = HbH.drop(columns=["class2"])
y_data = HbH["class2"]

# 📌 Giao diện nhập liệu (thay cho vòng for cũ)
st.set_page_config(layout="wide")
st.title("🧬 HbH Disease Prediction")


col1, col2 = st.columns([1, 2])
with col1:
    st.header("🔢 Inpiut Data")

    labels = {
        "age": "Age (year)",
        "sex": "Sex (1=Male, 2=Female)",
        "transfusion": "Number of transfusion",
        "RBC": "RBC (10^12/L)",
        "Hb": "Hemoglobin (g/L)",
        "MCV": "MCV (fL)",
        "MCH": "MCH (pg)",
        "RDWCV": "RDW-CV (%)",
        "Ferritin": "Ferritin (ng/mL)",
        "HbHa": "HbH (%)",
        "Bart": "Hb Bart (%)",
        "HbA": "HbA (%)",
        "HbF": "HbF (%)",
        "HbE": "HbE (%)",
        "HbA2": "HbA2 (%)",
        "HbC": "HbC (%)"
    }

    user_input = {}
    for col in X_data.columns:
        label = labels.get(col, col)
        val = st.number_input(label, value=np.nan)
        if not np.isnan(val):
            user_input[col] = val

    run_analysis = st.button("▶️ Run anslysis", key="run_analysis_btn")

with col2:
    if run_analysis:
        if len(user_input) >= 1:
            selected_features = list(user_input.keys())
            X_selected = X_data[selected_features].copy()
            y = y_data

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_selected)
            user_df = pd.DataFrame([user_input])[selected_features]
            user_scaled = scaler.transform(user_df)

            models = {
                "Logistic Regression": LogisticRegression(),
                "SVM": SVC(probability=True),
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes": GaussianNB()
            }

            rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
            results = []
            fitted_models = {}

            st.subheader("📊 Model Results")
            for name, model in models.items():
                accs, kappas, ses, sps = [], [], [], []

                for train_idx, test_idx in rkf.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accs.append(accuracy_score(y_test, y_pred))
                    kappas.append(cohen_kappa_score(y_test, y_pred))
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                    se = tp / (tp + fn) if (tp + fn) else 0
                    sp = tn / (tn + fp) if (tn + fp) else 0
                    ses.append(se)
                    sps.append(sp)

                model.fit(X_scaled, y)
                fitted_models[name] = model
                pred_user = model.predict(user_scaled)[0]
                proba_user = model.predict_proba(user_scaled)[0][1]

                results.append({
                    "Model": name,
                    "Accuracy": np.mean(accs),
                    "Kappa": np.mean(kappas),
                    "Sensitivity (Se)": np.mean(ses),
                    "Specificity (Sp)": np.mean(sps),
                    "Prediction": int(pred_user),
                    "Probability": proba_user
                })

            results_df = pd.DataFrame(results)
            float_cols = ['Accuracy', 'Kappa', 'Sensitivity (Se)', 'Specificity (Sp)', 'Probability']
            styled_df = results_df.style.format({col: "{:.2f}" for col in float_cols})
            st.dataframe(styled_df, use_container_width=True)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results CSV", csv, "results.csv", "text/csv")

            st.subheader("📈 ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            for name, model in fitted_models.items():
                y_score = model.predict_proba(X_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_score)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

            st.subheader("📉 Prediction Difference Plot")
            fig_ba, ax_ba = plt.subplots()
            for name, model in fitted_models.items():
                y_pred_all = model.predict(X_scaled)
                avg = (y + y_pred_all) / 2
                diff = y_pred_all - y
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)

                ax_ba.scatter(avg, diff, alpha=0.5, label=name)
                ax_ba.axhline(mean_diff, color='gray', linestyle='--')
                ax_ba.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--')
                ax_ba.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--')

            ax_ba.set_title("Prediction Difference Plot")
            ax_ba.set_xlabel("Mean of Prediction and True")
            ax_ba.set_ylabel("Prediction - True")
            ax_ba.legend()
            st.pyplot(fig_ba)
        else:
            st.warning("⚠️ Please enter at least 1 variable before running.")
    else:
        st.info("💡 Enter data on the left and click **Start Analysis** to see results.")
