# 🛡️ Unbiased AI: Compliance Terminal

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)

## 🌐 Live Access
**Access the live auditor terminal here:** [https://ai-fairness-auditor-6dn7twadahkdgx2wxwbact.streamlit.app/](https://ai-fairness-auditor-6dn7twadahkdgx2wxwbact.streamlit.app/)

## 📌 Project Overview
[cite_start]The Unbiased AI Compliance Terminal is an interactive auditing and mitigation engine designed to enforce algorithmic fairness in machine learning models. [cite: 6] [cite_start]This application mathematically detects demographic prejudices and applies post-processing threshold optimization to ensure legal compliance (specifically the 4/5ths Rule / Disparate Impact threshold). [cite: 8]

## 🚨 The Problem
[cite_start]Machine learning models often inherit historical human biases from their training data. [cite: 12] [cite_start]If a model predicting income or loan approval approves male applicants at a significantly higher rate than female applicants, it creates a "Disparate Impact," which is both unethical and legally non-compliant. [cite: 13]

## 💡 The Solution
This terminal intercepts biased AI predictions:
1. **Baseline Audit:** Calculates the Disparate Impact score using the formula:
   $$DI = \frac{\text{Approval Rate of Unprivileged Group}}{\text{Approval Rate of Privileged Group}}$$
2. [cite_start]**Mitigation:** Dynamically adjusts decision thresholds for each demographic to neutralize historical skew. [cite: 16]
3. [cite_start]**Compliance Verification:** Re-audits the model to ensure the DI score is above the legal **0.80** requirement. [cite: 17]

## ⚙️ Technology Stack
* [cite_start]**Cloud & AI:** Google Cloud (Colab), Google Gemini API [cite: 8, 22]
* [cite_start]**Frontend:** Streamlit Cloud [cite: 22]
* [cite_start]**Core Engine:** Scikit-Learn, Pandas, NumPy [cite: 22]
* [cite_start]**Data:** UCI Adult Income Dataset [cite: 22]