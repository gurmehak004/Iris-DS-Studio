# Iris Data Studio 🌸

Iris Data Studio is a premium, AI-powered automated data science workspace. It transforms raw datasets into actionable insights, cleaned data, and professional board-ready deliverables in minutes.

## 🚀 Features

- **Automated Data Quality Audit:** Instantly detects missing values, duplicates, outliers, and data type inconsistencies.
- **AI-Powered Feature Engineering:** Get smart recommendations for scaling, encoding, and date-time extractions.
- **Interactive EDA:** High-quality, dynamic visual distributions, and Pearson correlation matrices.
- **Machine Learning Recommendations:** Context-aware suggestions for the best ML algorithms to solve your business problem.
- **Professional Exports:** Prepare and download an interactive HTML dashboard, a high-contrast executive PDF report, or your freshly cleaned CSV dataset from the final stage.
- **"Iris" AI Chat:** Ask questions directly about your dataset's findings.

## 💻 Tech Stack
- **Frontend:** Streamlit
- **Visualizations:** Plotly, Seaborn/Matplotlib
- **AI Integration:** Iris AI (via Google GenAI)
- **Exporting:** FPDF (Custom Dark Theme), HTML/CSS

## 🛠️ Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/gurmehak004/Iris-DS-Studio.git
   cd Iris-DS-Studio
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional, for app operators) Configure an API key in the app environment (PowerShell):
   ```powershell
   $env:GEMINI_API_KEY = "your-api-key-here"
   ```
   Alternatively, add `GEMINI_API_KEY = "your-api-key-here"` to `.streamlit\secrets.toml`.
   For Streamlit Community Cloud, open the app's **Settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   ```
   Save the settings and reboot the app. Use **Ask Iris AI to review these models** in the
   Communication stage to get extended AI insights. Iris AI explanations and the expanded
   executive narrative run only when requested, and API calls have a fast 6-second timeout.
   The app never displays the key to end users.
   Do not ask end users to provide or paste API keys, and never commit keys. All statistical
   analyses, cleaning, and model cards run natively in Python without needing a key.
4. Run the app:
   ```bash
   streamlit run app.py
   ```
