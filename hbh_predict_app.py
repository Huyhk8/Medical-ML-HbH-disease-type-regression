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

HbH = pd.read_csv("HbH.csv") 

# Tách X (đặc trưng đầu vào) và y (biến mục tiêu)
X_data = HbH.drop(columns=["class2"])
y_data = HbH["class2"]

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🧬 HbH Disease Prediction")

col1, col2 = st.columns([1, 2])
with col1:
    st.header("🔢 Input data")
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
    run_analysis = st.button("▶️ Run analysis")

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

            st.subheader("📊 Kết quả mô hình")
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
            st.download_button("📥 Tải bảng kết quả CSV", csv, "results.csv", "text/csv")

            st.subheader("📈 Đường cong ROC")
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

            st.subheader("📉 Biểu đồ khác biệt dự đoán")
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
            st.warning("⚠️ Vui lòng nhập ít nhất 1 biến trước khi chạy.")
    else:
        st.info("💡 Nhập dữ liệu bên trái và nhấn nút **Bắt đầu phân tích** để xem kết quả.")
HbH.to_csv("HbH.csv", index=False)
