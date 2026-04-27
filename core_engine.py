import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def build_baseline_model():
    print("[SYSTEM] Fetching Dataset via Direct Secure Link...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    
    # Safely load and strip whitespace
    data = pd.read_csv(url, names=columns, sep=r',\s*', engine='python')
    
    # Isolate Target (>50K Income) and Features
    y = (data['income'] == '>50K') * 1 
    X_raw = data.drop(columns=['income'])
    
    # Isolate Sensitive Attribute (Gender) for the Audit
    sensitive_feature = X_raw['sex']
    
    # Convert text to machine-readable numbers
    X = pd.get_dummies(X_raw)

    X_train, X_test, y_train, y_test, _, sensitive_test = train_test_split(
        X, y, sensitive_feature, test_size=0.2, random_state=42
    )

    print("[SYSTEM] Training Core AI Model...")
    model = RandomForestClassifier(random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)

    print("[SYSTEM] Executing Bias Audit...")
    predictions = model.predict(X_test)
    
    results = pd.DataFrame({'Gender': sensitive_test.values, 'Approval': predictions})
    rates = results.groupby('Gender')['Approval'].mean()
    
    di_score = rates['Female'] / rates['Male']
    
    print("\n" + "="*40)
    print("      BASELINE AUDIT RESULTS      ")
    print("="*40)
    print(f"Male Approval Rate   : {rates['Male']:.1%}")
    print(f"Female Approval Rate : {rates['Female']:.1%}")
    print(f"Disparate Impact     : {di_score:.3f}")
    
    if di_score < 0.80:
        print("🚨 AUDIT FAILED: Model violates the 80% Rule.")
    else:
        print("✅ AUDIT PASSED: Model is statistically fair.")

if __name__ == "__main__":
    build_baseline_model()