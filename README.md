# 🛡️ Unbiased AI: Compliance Terminal

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)

## 🌐 Live Access
**Access the live auditor terminal here:** https://ai-fairness-auditor-6dn7twadahkdgx2wxwbact.streamlit.app/

## 📌 Project Overview
The Unbiased AI Compliance Terminal is an interactive auditing and mitigation engine designed to enforce algorithmic fairness in machine learning models. This application mathematically detects demographic prejudices and applies post-processing threshold optimization to ensure legal compliance—specifically the 4/5ths Rule / Disparate Impact threshold.

## 🚨 The Problem
Machine learning models often inherit historical human biases from their training data. If a model predicting income or loan approval approves male applicants at a significantly higher rate than female applicants, it creates a "Disparate Impact," which is both unethical and legally non-compliant.

## 💡 The Solution
This terminal intercepts biased AI predictions:
1. **Baseline Audit:** Calculates the Disparate Impact score using the formula:
   $$DI = \frac{\text{Approval Rate of Unprivileged Group}}{\text{Approval Rate of Privileged Group}}$$
2. **Mitigation:** Dynamically adjusts decision thresholds for each demographic to neutralize historical skew.
3. **Compliance Verification:** Re-audits the model to ensure the DI score is above the legal **0.80** requirement.

## ⚙️ Technology Stack
* **Cloud & AI:** Google Cloud (Colab), Google Gemini API
* **Frontend:** Streamlit Cloud
* **Core Engine:** Scikit-Learn, Pandas, NumPy
* **Data:** UCI Adult Income Dataset
