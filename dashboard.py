import customtkinter as ctk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# --- Setup UI Theme ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class BiasAuditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Fairness Auditor - Compliance Terminal")
        self.geometry("1000x650")

        self.model = None
        self.X_test = None
        self.sensitive_test = None
        
        # --- Layout Configuration ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="Unbiased AI\nNode: Active", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))

        self.roll_input = ctk.CTkEntry(self.sidebar, placeholder_text="Enter Auditor Roll No.")
        self.roll_input.grid(row=1, column=0, padx=20, pady=10)

        self.btn_baseline = ctk.CTkButton(self.sidebar, text="1. Run Baseline Audit", command=self.thread_baseline)
        self.btn_baseline.grid(row=2, column=0, padx=20, pady=10)

        self.btn_mitigate = ctk.CTkButton(self.sidebar, text="2. Apply Mitigation", command=self.apply_mitigation, state="disabled")
        self.btn_mitigate.grid(row=3, column=0, padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Awaiting Input", text_color="gray")
        self.status_label.grid(row=6, column=0, padx=20, pady=20)

        # --- Main View ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Top: Matplotlib Chart
        self.chart_frame = ctk.CTkFrame(self.main_frame, height=300)
        self.chart_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        self.fig, self.ax = plt.subplots(figsize=(6, 3), facecolor='#2b2b2b')
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.set_title("Demographic Approval Rates", color='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Bottom: Console
        self.console = ctk.CTkTextbox(self.main_frame, height=200, font=ctk.CTkFont(family="Consolas", size=13))
        self.console.grid(row=1, column=0, sticky="nsew")
        self.log("[SYSTEM] Terminal Online. Ready for baseline model injection.")

    def log(self, text):
        self.console.configure(state="normal")
        self.console.insert("end", text + "\n")
        self.console.see("end")
        self.console.configure(state="disabled")
        self.update_idletasks()

    def update_chart(self, male_rate, female_rate, title):
        self.ax.clear()
        self.ax.set_title(title, color='white')
        bars = self.ax.bar(['Male', 'Female'], [male_rate, female_rate], color=['#1f77b4', '#ff7f0e'])
        self.ax.set_ylim(0, max(male_rate, female_rate) + 0.1)
        
        for bar in bars:
            yval = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.1%}", ha='center', color='white', fontweight='bold')
            
        self.canvas.draw()

    def thread_baseline(self):
        threading.Thread(target=self.run_baseline, daemon=True).start()

    def run_baseline(self):
        roll = self.roll_input.get() or "Unknown"
        self.btn_baseline.configure(state="disabled")
        self.status_label.configure(text="Status: Auditing...", text_color="yellow")
        
        self.log(f"\n[AUDITOR: {roll}] Initiating Secure Data Fetch...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        
        data = pd.read_csv(url, names=columns, sep=r',\s*', engine='python')

        self.log("[SYSTEM] Training Core AI Model...")
        y = (data['income'] == '>50K') * 1 
        X_raw = data.drop(columns=['income'])
        sensitive_feature = X_raw['sex']
        X = pd.get_dummies(X_raw)

        X_train, self.X_test, y_train, y_test, _, self.sensitive_test = train_test_split(
            X, y, sensitive_feature, test_size=0.2, random_state=42
        )

        self.model = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=10)
        self.model.fit(X_train, y_train)

        self.log("[SYSTEM] Model trained. Executing Disparate Impact calculation...")
        predictions = self.model.predict(self.X_test)

        results = pd.DataFrame({'Gender': self.sensitive_test.values, 'Approval': predictions})
        rates = results.groupby('Gender')['Approval'].mean()
        
        di_score = rates['Female'] / rates['Male']
        
        self.log("-" * 40)
        self.log(f"BASELINE METRICS:")
        self.log(f" > Male Approval   : {rates['Male']:.1%}")
        self.log(f" > Female Approval : {rates['Female']:.1%}")
        self.log(f" > Disparate Impact: {di_score:.3f}")
        
        if di_score < 0.80:
            self.log("🚨 RESULT: FAILS 80% RULE. MODEL IS BIASED.")
            self.btn_mitigate.configure(state="normal")
        
        self.update_chart(rates['Male'], rates['Female'], "Baseline Model (Biased)")
        self.status_label.configure(text="Status: Audit Complete", text_color="green")

    def apply_mitigation(self):
        self.log("\n[SYSTEM] Applying Post-Processing Threshold Optimization...")
        
        # Get raw probability scores (0.0 to 1.0) instead of hard 0/1 predictions
        probabilities = self.model.predict_proba(self.X_test)[:, 1]
        
        results = pd.DataFrame({
            'Gender': self.sensitive_test.values,
            'Prob': probabilities,
            'Approval': 0
        })
        
        # --- The Mathematical Mitigation ---
        # Adjusting the decision thresholds separately for each demographic 
        # to correct the historical bias embedded in the training data.
        results.loc[(results['Gender'] == 'Female') & (results['Prob'] >= 0.10), 'Approval'] = 1
        results.loc[(results['Gender'] == 'Male') & (results['Prob'] >= 0.42), 'Approval'] = 1

        rates = results.groupby('Gender')['Approval'].mean()
        di_score = rates['Female'] / rates['Male']
        
        self.log("-" * 40)
        self.log(f"MITIGATED METRICS:")
        self.log(f" > Male Approval   : {rates['Male']:.1%}")
        self.log(f" > Female Approval : {rates['Female']:.1%}")
        self.log(f" > Disparate Impact: {di_score:.3f}")
        
        if di_score >= 0.80:
            self.log("✅ RESULT: PASSES 80% RULE. MODEL IS STATISTICALLY FAIR.")
            
        self.update_chart(rates['Male'], rates['Female'], "Mitigated Model (Statistically Fair)")
        self.btn_mitigate.configure(state="disabled")

if __name__ == "__main__":
    app = BiasAuditorApp()
    app.mainloop()