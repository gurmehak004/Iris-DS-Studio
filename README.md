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
3. Set your Google Gemini API Key in your environment:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
