# 📊 Customer Churn Analysis Dashboard

An interactive **Streamlit dashboard** to analyze, predict, and visualize customer churn.  
Built with **Python, Scikit-Learn, and Streamlit**, this project transforms raw CSV data into **business insights** and **predictive analytics**.

---

## 🚀 Features

- 📂 **Flexible CSV Upload** – works with any structured dataset
- 🧹 **Smart Preprocessing** – automatic numeric & categorical column detection
- ⚡ **Multi-Model Training**
  - Logistic Regression
  - Random Forest
  - XGBoost
- 📊 **Interactive Visualizations**
  - Churn distribution
  - Feature importance
  - Correlation heatmaps
  - SHAP explainability plots
- 🧾 **Downloadable Reports**
  - Excel report (metrics, predictions, feature importance)
- 🎨 **Modern UI**
  - Clean sidebar controls
  - Responsive layout
  - Dark/Light mode supported

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) – UI & Dashboard  
- [Pandas](https://pandas.pydata.org/) – Data manipulation  
- [Scikit-Learn](https://scikit-learn.org/) – ML models  
- [XGBoost](https://xgboost.ai/) – Advanced boosting model  
- [SHAP](https://shap.readthedocs.io/) – Explainable AI  
- [Matplotlib / Seaborn / Plotly](https://plotly.com/) – Visualizations  

---

## 📂 Project Structure

```
Customer_Churn_Analysis/
│
├── app.py                # Streamlit dashboard
├── requirements.txt      # Dependencies
├── .streamlit/
│   └── config.toml       # UI theme
├── scripts/              # Modularized logic
│   ├── preprocessing.py  # Data cleaning
│   ├── modeling.py       # Model training
│   ├── evaluation.py     # Metrics & explainability
│   ├── io_utils.py       # File handling
│   └── utils.py          # Helpers
└── README.md             # Project documentation
```

---

## ⚡ Usage

1. **Clone the repo**
   ```bash
   git clone https://github.com/VIGNESH54/Customer_Churn_Analysis.git
   cd Customer_Churn_Analysis
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app locally**
   ```bash
   streamlit run app.py
   ```

5. **Deploy to Streamlit Cloud**
   - Connect GitHub repo
   - Select `app.py` as entrypoint

---

## 🎯 Example Insights

- Identify high-risk churn segments  
- Discover top predictors of churn (e.g., contract type, tenure, charges)  
- Generate explainable ML predictions with SHAP  

---

## 📸 Screenshots

*(Add app screenshots here once deployed!)*

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss your ideas.

---

## 📜 License

This project is licensed under the MIT License.

---

👨‍💻 Developed by **Vignesh P**  
✨ Showcasing advanced **Data Science & ML Engineering skills**
