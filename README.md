# 🛡️ Unbiased AI: Compliance Terminal

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange)

**Live Web Application:** [Insert your Streamlit Cloud Link Here]
**Demo Video:** [Insert your Google Drive Video Link Here]

## 📌 Project Overview
The Unbiased AI Compliance Terminal is an interactive auditing and mitigation engine designed to enforce algorithmic fairness in machine learning models. Built for the Solution Challenge 2026, this application mathematically detects demographic prejudices in AI predictions and applies post-processing threshold optimization to ensure legal compliance (specifically the 4/5ths Rule / Disparate Impact threshold).

## 🚨 The Problem
Machine learning models often optimize solely for raw accuracy, inadvertently inheriting historical human biases from their training data. For example, a naive model predicting high income may approve male applicants at a vastly higher rate than female applicants, resulting in algorithmic discrimination and violating legal fairness standards.

## 💡 The Solution
This terminal intercepts biased AI predictions before deployment. 
1. **Baseline Audit:** Ingests the model and mathematically exposes the bias (calculating the Disparate Impact score).
2. **Mitigation Engine:** Dynamically adjusts decision thresholds for privileged and unprivileged groups.
3. **Compliance:** Equalizes approval rates to push the Disparate Impact score above the 0.80 legal requirement, prioritizing human equity alongside statistical accuracy.

## ⚙️ Technology Stack
* **Frontend/Deployment:** Streamlit (Community Cloud)
* **Core Logic:** Python, Scikit-Learn, Pandas, NumPy
* **Visualization:** Matplotlib
* **Data Source:** UCI Adult Income Dataset

## 🚀 How to Run Locally

1. Clone the repository:
git clone https://github.com/BhaskarNova/ai-fairness-auditor.git