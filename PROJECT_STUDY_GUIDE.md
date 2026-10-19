# Iris Data Science Studio: Project Study Guide

Use this document as a learning and viva-preparation reference. It describes the current source code, not every feature proposed in the original synopsis. If the application changes, update the relevant sections and distinguish implemented features from planned work.

## 1. Project in One Sentence

Iris Data Science Studio is a Streamlit application that guides users through data collection, cleaning, exploratory data analysis (EDA), feature engineering, model recommendations, and report export.

Its aim is to bring common data science tasks together and make complex analysis easier to approach without requiring users to write code.

## 2. Project Structure

### Root files and folders

- `app.py`: Main application. Defines the five stages, manages Streamlit session state, receives user actions, calls engine functions, and displays results.
- `README.md`: Project description, features, dependencies, and local run instructions.
- `requirements.txt`: Python dependencies, including Streamlit, Pandas, NumPy, SciPy, Plotly, scikit-learn, XGBoost, Chardet, Gemini, FPDF, and Excel readers.
- `.streamlit/config.toml`: Streamlit theme and upload-size configuration. The configured maximum upload size is 1,000 MB; this does not guarantee that a machine has enough memory to analyse a file of that size.
- `.devcontainer/devcontainer.json`: Development-container configuration. It specifies Python 3.11, installs project requirements, and runs Streamlit on port 8501.
- `assets/logo.jpg`: Logo used in the interface.
- `samples/`: Demo datasets: `customer_churn.csv`, `ecommerce_sales.csv`, `heart_disease.csv`, `housing.csv`, `iris.csv`, and `titanic.csv`.
- `.git/`: Git version-control metadata.
- `__pycache__/`: Generated Python bytecode cache; not application logic.

### `engine/` modules

- `parser.py`: Ingests CSV, Excel, and Google Sheets data.
- `type_inference.py`: Infers column types from data values.
- `quality.py`: Computes quality summaries such as missing values, duplicates, and numeric outliers.
- `cleaning.py`: Implements missing-value handling, duplicate removal, outlier treatment, structural cleanup, type conversion, encoding, scaling, reduction, recommendations, and health scoring.
- `eda.py`: Computes column statistics, correlations, class-imbalance results, and EDA findings.
- `eda_automation.py`: Selects EDA charts using rules, provides chart rationales, and builds a standalone HTML EDA dashboard.
- `viz.py`: Reusable Plotly chart-building helpers. Some app charts are instead built directly in `app.py` or `eda_automation.py`.
- `feature_engineering.py`: Suggests and applies feature transformations, creates interactions, plans transformations, and computes a limited feature-importance preview.
- `business.py`: Stores domain/objective templates and suggests a target column using names and domain keywords.
- `ml_recommender.py`: Recommends machine-learning approaches and evaluation metrics. This is called by the current app.
- `ml.py`: Contains separate model-training and clustering functions. The current app does not call `train_model()` or `run_clustering()`.
- `gemini_client.py`: Calls the Gemini API for selected explanations and executive summaries, with fallback text if the API is unavailable or unconfigured.
- `exporter.py`: Builds the cleaned CSV and PDF report.
- `html_exporter.py`: Builds the full HTML dashboard.
- `__init__.py`: Marks `engine` as a Python package.

### `ui/` modules

- `components.py`: Reusable Streamlit display components such as KPI cards, previews, quality scorecards, and section headers.
- `styles.py`: CSS styles and a style-injection function. The main app also defines a large inline CSS block.
- `__init__.py`: Marks `ui` as a Python package.

## 3. Application Pipeline

The five visible stages are:

1. **Business & Data Collection:** Choose a domain/objective, upload a file or load a Google Sheet, inspect the preview, inferred types, and health summary.
2. **Data Cleaning & Preprocessing:** Review quality findings and optionally apply cleaning actions.
3. **Exploratory Data Analysis:** Inspect statistics and correlations, findings, and automatically recommended charts.
4. **Feature Engineering:** Preview and apply transformations, create interactions, and optionally view the feature-importance preview.
5. **Communication & Export:** Review the immediate executive summary and model recommendations; prepare reports or cleaned data only when needed.

`app.py` uses `st.session_state` to retain the active DataFrame, metadata, inferred column types, quality results, statistics, correlations, target suggestion, and action logs as Streamlit reruns after interactions. The application has no persistent project database or saved analysis history.

### End-to-end flow

`Upload / select data -> parse into DataFrame -> infer types -> suggest target -> assess quality -> clean data -> compute EDA -> recommend charts -> engineer features -> show model advice and narratives -> export`

Cleaning and feature engineering mutate the active DataFrame. Results should be interpreted in light of the current state of that DataFrame.

## 4. Ingestion and Large Files

The main parser is in `engine/parser.py`.

### CSV path

1. The app passes uploaded file bytes to `load_csv()`.
2. Chardet guesses the file encoding. If decoding raises an exception, the parser falls back to UTF-8 with replacement characters.
3. The first 5,000 characters are inspected. The parser counts comma, semicolon, tab, and pipe characters and chooses the most frequent as the delimiter.
4. Pandas reads the decoded text using its Python CSV engine. `on_bad_lines="skip"` means malformed lines can be skipped rather than causing the whole file to fail.
5. The parser trims header whitespace, drops entirely empty rows and columns, and resets the row index.
6. It records metadata such as encoding, delimiter, dimensions, and sampling status.

The delimiter detection is a simple character-count heuristic, not a complete CSV dialect detector.

### Excel and Google Sheets

- Excel files are read with Pandas. The app lets users choose a worksheet, removes fully empty rows/columns, then applies the same row-sampling threshold.
- Google Sheets are retrieved through a CSV export link. The sheet must be accessible through its sharing settings; its contents are passed through the CSV parser.

### Sampling details

If the parsed CSV has more than 50,000 rows, `load_csv()` uses `DataFrame.sample(n=50000, random_state=42)`. Excel ingestion has the same threshold. The fixed seed makes sampling reproducible for the same input and row order.

**Important:** Sampling occurs after Pandas has read the file into a DataFrame. It reduces the data used by later analysis, but it does not prevent the full input from being loaded into memory first. The parser is not streaming or chunk-based. Rare patterns can also be absent from a sample.

## 5. Type Inference and Target Suggestion

`engine/type_inference.py` uses a rule order rather than a supplied schema. It checks for boolean-like values, date-like values, numeric dtypes, numeric-looking strings, and then category/text/ID patterns. Types include `numeric`, `categorical`, `datetime`, `boolean`, `text`, `id`, and `empty`.

Type inference can be wrong. For example, an integer-like identifier may be treated as numeric in some cases. The cleaning UI allows a user to recast a column, but the initial inference remains heuristic.

`engine/business.py` suggests a target by looking for common target names, then domain keywords, then falling back to the last column. This is a suggestion, not a guarantee that the chosen column is the correct prediction target.

## 6. Data Cleaning: Seven Steps

The cleaning stage reports issues and presents actions. The user should check whether each action makes sense for the dataset; an automatically detected issue is not always a data error.

### Step 1: Missing values

A missing value is an empty or unavailable cell. The report gives missing counts and percentages by column.

- **Mean imputation:** Fill numeric gaps with the arithmetic average. Extreme values can pull the mean.
- **Median imputation:** Fill numeric gaps with the middle value. It is less sensitive to extreme values.
- **Mode imputation:** Fill gaps with the most common value; often used for categorical data.
- **Constant imputation:** Fill gaps with a chosen value such as `0` or `Unknown`.
- **Forward/backward fill:** Use a preceding or following value. This is most sensible when row order represents a sequence or time.
- **Drop rows / columns:** Remove incomplete observations or features, at the cost of losing data.

The quick sparse-column action uses a greater-than-40% missingness threshold. This is a configurable decision in principle, not a universal statistical rule.

### Step 2: Duplicate rows

A duplicate row repeats the same values across all columns. Duplicates can overweight repeated observations in summaries and later analysis. The app previews duplicates and can remove them. Identical rows may still be legitimate records, so confirm their meaning before removal.

### Step 3: Outliers

An outlier is unusually far from most values. It can be an error, a rare valid case, or a meaningful event; it should not automatically be deleted.

- **IQR method:** Let `Q1` be the 25th percentile and `Q3` the 75th percentile. `IQR = Q3 - Q1`. Values below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR` are flagged.
- **Z-score method:** Measures how many standard deviations a value is from the mean. The app flags absolute Z-scores above a threshold, normally 3.
- **Capping/clipping:** Replace values outside the calculated boundary with the boundary. This retains the row but limits the extreme value.
- **Dropping:** Remove rows containing detected outlier values. This can discard valid rare observations.

Example: if the IQR upper fence is 100 and a value is 500, capping changes it to 100; dropping removes the row.

### Step 4: Structural errors and type correction

Structural issues make equivalent values appear different or store values in an inconvenient format. Examples include leading/trailing whitespace (`" Male "`), inconsistent casing (`"male"` and `"Male"`), and numeric values stored as strings.

The app can trim whitespace, change case, remove punctuation, and convert a column to a selected type. Check that standardization does not erase meaningful distinctions.

### Step 5: Categorical encoding

Many mathematical models need numeric inputs, so categories may need encoding.

- **Binary encoding:** Map two categories to 0 and 1.
- **One-hot encoding:** Create a 0/1 indicator column for each category; useful for low-cardinality features.
- **Label encoding:** Map categories to integer codes; the numbers may imply an order that the categories do not actually have.

The app recommends binary encoding for two categories, one-hot encoding for up to ten, and label encoding for higher cardinality. Applying an encoding changes/adds columns in the active data.

### Step 6: Feature scaling

Features measured in different units or ranges may need to be rescaled. Distance-based methods are especially sensitive to scale.

- **Standard scaling:** `z = (x - mean) / standard_deviation`; centers values around zero and expresses distance in standard deviations.
- **Min-max scaling:** Rescales values to the `[0, 1]` range.

The recommendation logic leaves values already in `[0, 1]` unchanged, recommends standard scaling for some high-skew/high-range cases, and otherwise recommends min-max scaling. Scaling does not remove outliers. Tree models are generally less scale-sensitive than distance-based or linear methods.

### Step 7: Data reduction

- **Constant-column removal:** Drop columns with one or fewer distinct values; they cannot distinguish rows.
- **High-correlation reduction:** The app can remove numeric columns when an absolute correlation is above `0.90`. High correlation can indicate overlapping information, but does not alone prove a feature is useless.
- **Manual removal:** The user selects columns to drop.

Review the data meaning before reduction; dropping a column is a data-loss decision.

## 7. EDA Statistics and Charts

`engine/eda.py` computes per-column summaries and Pearson/Spearman correlations. It sorts numeric pairs by the absolute Pearson correlation and retains the strongest pairs for display/recommendations. Correlation measures association, not causation.

The rule-based chart recommender is in `engine/eda_automation.py`. The same dataset in the same state usually produces the same recommendations; the selection is not random.

| Chart | Main use |
| --- | --- |
| Histogram with rug | Numeric distribution shape, concentration, spread, and possible skew |
| Box plot | Median, quartiles, spread, potential outliers, and group comparisons |
| Horizontal bar chart | Compare category frequencies and identify dominant categories |
| Donut chart | Show proportions for a small number of categories |
| Correlation heatmap | Scan many numeric pairwise correlations at once |
| Scatter plot with trend line | Inspect direction and approximate form of a numeric relationship |
| Grouped box plot | Compare category medians and value spreads at a glance |
| Grouped bars for two categorical variables | Compare combinations of category values |
| Colour-coded multivariate scatter | View two numeric dimensions split by a category |
| 2D density heatmap | Locate concentrations of paired numeric values |
| Parallel coordinates | Compare several numeric variables across observations |
| Datetime/numeric line chart | Show a numeric value over time; does not itself forecast or aggregate counts |

Some recommendations are intentionally limited, such as using the strongest correlation pairs or only a few columns/categories. Grouped category charts use box plots, ordered by median, for a quick comparison of centers and spread. The feature explorer also has numeric histogram/violin/box/outlier-strip charts, category bar/pie charts, and the cleaning stage has a missing-value heatmap.

## 8. Feature Engineering

Feature engineering creates or transforms columns to make data more useful for analysis or possible later modelling. The app offers a plan, automatic transformations, and custom actions.

- Numeric: a log-transformed feature may be added when absolute skewness is above 1; standardized features can be added.
- Binary categorical: a 0/1 encoded feature can be created.
- Low-cardinality categorical: one-hot indicator columns can be created.
- Datetime: year, month, day, weekday, and weekend indicators can be extracted.
- Custom transformations include min-max scaling, quantile binning, frequency encoding, text length/word-count features, and dropping a column.
- **Interaction features:** Multiply two numeric columns or divide one by the other. Division by zero is treated as missing.

Automatic transforms are general rules; inspect the preview and outputs before using them. Repeated transformations can add more columns and make interpretation harder.

### Feature-importance preview

`quick_feature_importance()` can fit a 15-tree Random Forest for a preview. It uses numeric columns only and drops rows with missing values. It chooses a classifier when the target has at most ten unique values and otherwise a regressor. It does not create a train/test split or report evaluation metrics, so the result is exploratory and not validated model performance.

## 9. Machine-Learning Recommendations vs Training

The app calls `recommend_ml_models()` from `engine/ml_recommender.py` in the final stage. It uses the target type and unique-value count to decide whether to recommend classification, regression, or unsupervised approaches. These baseline model cards are generated by Python rules. The user can click **Ask Gemini to review these models** to send the candidate list and a dataset summary (not raw rows) to Gemini for a personalized ranking and rationale. The UI labels whether that review actually succeeded; a configured key alone does not mean a recommendation used Gemini.

**Neither the recommendation cards nor the Gemini review train the listed models or show actual scores from those models.** Iris suggests a target from the column names, and the Stage 1 target dropdown lets the user confirm or change the value to predict. For example, the housing sample should suggest `price`; generic labels such as `y` must not match substrings such as `age_years`.

There is a separate `engine/ml.py` module containing `train_model()` and `run_clustering()`. `train_model()` defines preprocessing and classifiers/regressors, then uses an 80/20 train/test split; it does not perform cross-validation. Pylance reference search found no caller for `train_model()` in the application. Explain this as code present in the repository but not integrated into the UI.

## 10. AI Explanations

`engine/gemini_client.py` uses the Google GenAI SDK and the `gemini-3.8-flash` model. It reads a Gemini key from `GEMINI_API_KEY` or Streamlit secrets, configured by the app operator rather than end users. Iris explanations and the expanded executive narrative run only when requested; API calls use a 15-second timeout and a single attempt. The final stage shows a local summary immediately, while its HTML, PDF, and CSV exports are prepared only on request. Gemini also supports open-ended chat and an explicitly requested model-recommendation review. The recommendation cards themselves remain rule-generated; the review result is separately labeled as Gemini-generated only after a successful response. Basic target-feature and column questions are answered locally without a key. Gemini is optional; when it is unavailable, built-in answers remain available.

Stage 2 and EDA tab selections are persisted through Streamlit reruns. The outlier summary distinguishes columns with IQR flags from columns with Z-score flags. The one-click capping action only caps columns with detected IQR outliers; Z-score-only flags are called out separately so they are not mistaken for uncapped IQR issues.

Interactive chat receives the active dataset's row count, column names and inferred types, target feature and its statistics, and business goal. It does not receive raw dataset rows. Other inspected explanation prompts pass summaries/statistics rather than raw dataset rows, but avoid making broader privacy guarantees without checking every external call and deployment setting.

## 11. Export and Operational Notes

- `engine/exporter.py`: Cleaned CSV and PDF report.
- `engine/html_exporter.py`: Full HTML dashboard.
- `engine/eda_automation.py`: Standalone EDA charts HTML dashboard.
- Data is kept in the active Streamlit session; there is no persistent user project database.
- Files over 50,000 rows are sampled after parsing. Results may omit patterns in the full dataset.
- Gemini-generated content requires an API key and network access; core non-AI analyses have fallback/independent paths.
- The selected Python 3.13 interpreter in the workspace was missing `chardet` during an import check, while `requirements.txt` includes it. Use an environment with project requirements installed before a live demo. The Dev Container is configured for Python 3.11.
- No project test files were found in the workspace listing. Do not claim a test suite was run unless you run and document one.

## 12. Presentation Answers

**What is the project?**

“Iris is a guided data science workflow that combines ingestion, cleaning, EDA, feature engineering, model recommendations, and reporting in a Streamlit application.”

**Where is AI used?**

“Gemini is an optional natural-language explanation layer and can review the app's rule-generated model candidates when asked. The model candidate list, statistics, cleaning checks, and chart recommendations are computed by Python logic. Model recommendations are advisory; models are not trained or scored.”

**Does it train models?**

“The current user-facing workflow recommends models and metrics. It also has a limited numeric-only Random Forest feature-importance preview. Separate training code exists but is not integrated into the app.”

**How does it handle outliers?**

“It flags numeric values with IQR fences or a Z-score threshold. The user can cap them to a boundary or drop affected rows. An outlier may be valid, so it should be reviewed before treatment.”

**Why the name Iris?**

“Iris refers to the classic Iris flower dataset many learners meet when beginning data science. ‘Studio’ reflects that the app brings several analysis stages into one workspace.”

**What is the project’s main contribution?**

“It integrates common data preparation, exploration, feature-engineering, and reporting tasks into one accessible guided interface; its contribution is workflow integration and usability, not a new statistical algorithm.”