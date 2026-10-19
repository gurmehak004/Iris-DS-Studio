# Iris Data Studio 🌸

Iris Data Studio is a premium, AI-powered automated data science workspace. It transforms raw datasets into actionable insights, cleaned data, and professional board-ready deliverables in minutes.

## 🚀 Features

- **Automated Data Quality Audit:** Instantly detects missing values, duplicates, outliers, and data type inconsistencies.
- **AI-Powered Feature Engineering:** Get smart recommendations for scaling, encoding, and date-time extractions.
- **Interactive EDA:** High-quality, dynamic visual distributions, and Pearson correlation matrices.
- **Machine Learning Recommendations:** Context-aware suggestions for the best ML algorithms to solve your business problem.
- **Professional Exports:** Download an interactive HTML dashboard, a high-contrast executive PDF report, or your freshly cleaned CSV dataset.
- **"Iris" AI Chat:** Ask questions directly about your dataset's findings.

## 💻 Tech Stack
- **Frontend:** Streamlit
- **Visualizations:** Plotly, Seaborn/Matplotlib
- **AI Integration:** Google Gemini
- **Exporting:** FPDF (Custom Dark Theme), HTML/CSS

## 🛠️ Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Iris-DS-Studio.git
   cd Iris-DS-Studio
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional, for app operators) Configure a Google Gemini API key in the app environment (PowerShell):
   ```powershell
   $env:GEMINI_API_KEY = "your-api-key-here"
   ```
   Alternatively, add `GEMINI_API_KEY = "your-api-key-here"` to `.streamlit\secrets.toml`.
   For Streamlit Community Cloud, open the app's **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
   Save the settings and reboot the app. In the deployed app, use **Test Gemini connection**
   in the Iris chat section to verify an actual Gemini API request. The app never displays
   the key.
   Do not ask end users to provide or paste API keys, and never commit keys. Basic dataset
   questions are answered by Iris without Gemini; the key only enables Gemini-enhanced
   open-ended responses. Failed Gemini requests fall back to built-in answers.
4. Run the app:
   ```bash
   streamlit run app.py
   ```
