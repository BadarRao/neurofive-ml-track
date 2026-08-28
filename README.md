NeuroFive ML Track

Machine learning internship track with NeuroFive Solutions. Ten foundation tasks covering the full workflow from exploratory data analysis through model evaluation, ensembles, and imbalanced data handling, followed by a self directed capstone taken from raw data to a deployed product.

Every result in this README comes from actually running the notebooks. No metric is estimated.

Certification: Machine Learning Fundamentals Internship, NeuroFive Solutions, July 2026 Cohort. Certificate ID 26071033, verifiable at neurofivesolutions.com/verify.

Live Apps
App	Description	Link
Air Quality Forecaster	Capstone. Predicts PM2.5 concentration from weather conditions	deployment in progress
Titanic Survival Predictor	Task 10. Predicts survival from passenger details	https://badarrao-titanic-predictor.streamlit.app/

Both are hosted on Streamlit Community Cloud's free tier, which sleeps after 12 hours without traffic. If you see a sleep screen, click "Yes, get this app back up!" and it loads in about 30 seconds.

Repository Structure
text
neurofive-ml-track/
│
├── Capstone_Air_Quality_Prediction.ipynb
│
├── Task_01_Titanic_EDA.ipynb
├── Task_02_Titanic_Data_Cleaning_Visualization.ipynb
├── Task_03_Titanic_Classification.ipynb
├── Task_04_House_Price_Regression.ipynb
├── Task_05_Model_Evaluation_Tuning.ipynb
├── Task_06_Telco_Churn_Prediction.ipynb
├── Task_07_ML_Pipeline_Feature_Engineering.ipynb
├── Task_08_Ensemble_RandomForest_XGBoost.ipynb
├── Task_09_Imbalanced_Data_Fraud_Detection.ipynb
├── Task_10_Model_Deployment_Streamlit.ipynb
│
├── capstone_app.py           # Capstone Streamlit application
├── aqi_model.joblib          # Capstone trained model
├── app.py                    # Task 10 Streamlit application
├── titanic_pipeline.joblib   # Task 10 trained model
├── requirements.txt          # Pinned dependencies, shared by both apps
│
├── train.csv                                 # Titanic dataset
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Telco churn dataset
│
├── CASE_STUDY.md             # Capstone case study writeup
└── README.md
Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, Matplotlib, seaborn, Streamlit, joblib. All notebooks are written for Google Colab.

Capstone: Air Quality Forecaster

Notebook: Capstone_Air_Quality_Prediction.ipynb | App: capstone_app.py | Case study: CASE_STUDY.md

Live app: deployment in progress

A self directed project taken from raw data to a deployed product, chosen independently rather than assigned.

Problem Statement

Fine particulate matter (PM2.5) is the air pollutant most strongly linked to human harm. The particles measure under 2.5 micrometres across, small enough to bypass the body's normal filtering, reach deep into the lungs, and pass into the bloodstream. Long term exposure is associated with heart disease, stroke, reduced lung development in children, and shortened life expectancy.

Cities across Asia experience severe seasonal smog, and residents want to know one practical thing: is tomorrow going to be bad?

Pollution monitoring stations are expensive and sparse. Weather forecasts are free, universal, and available days ahead. That gap motivates the question this project answers:

Can PM2.5 concentration be predicted from weather conditions alone?

If it can, any location with a weather forecast could issue an air quality warning without installing a single sensor.

This is a regression problem. The target is a continuous concentration in µg/m³.

Dataset

Hourly measurements from Beijing, 2010 to 2014, pairing PM2.5 readings taken at the US Embassy with meteorological data from Beijing Capital International Airport.

Property	Value
Raw records	43,824 hourly readings
Records after cleaning	41,757
Features	9 raw, expanded to 13 after engineering
Target	pm2.5, concentration in µg/m³
Missing values	2,067, all in the target column

Beijing was chosen because few cities publish five continuous years of paired pollution and weather data openly. The method transfers to anywhere with comparable records.

A framing statistic: 88.1% of hours in this five year record exceeded the WHO 24 hour guideline of 15 µg/m³.

Cleaning

All 2,067 missing values sat in the target column. These rows were dropped rather than imputed, a different decision from the foundation tasks where missing features were filled with a median or mode. Imputing a target means inventing the answer and then training the model to reproduce that invention, so every metric would partly measure how well the model copied a fabricated number. Dropping cost 4.7% of the data.

Feature engineering

Four features were added, each motivated by a physical mechanism rather than by trying combinations at random.

Feature	Definition	Reasoning
heating_season	15 Nov to 15 Mar	Beijing runs coal fired district heating on a fixed winter schedule. The raw month column captures this only indirectly, since the window splits mid month.
temp_dewp_spread	TEMP - DEWP	Indicates how close air is to saturation. A small spread signals humid, stagnant conditions that trap particulates near ground level.
is_night	Before 06:00 or from 20:00	Overnight cooling forms temperature inversions, a lid of warm air that stops pollution dispersing upward.
season	Meteorological season	A coarse grouping for tree models to split on.

Deliberate omission: no lagged PM2.5 features, despite yesterday's pollution being the single strongest available predictor. The project's purpose is forecasting from a weather forecast, and a user checking the app in advance has no recent pollution reading to supply. Including lags would have inflated the metrics while producing a model that cannot serve its stated purpose.

Evaluation strategy

This is time series data, so train_test_split with a random shuffle is the wrong tool. Pollution at 3pm closely resembles pollution at 2pm and 4pm. A random split scatters neighbouring hours across train and test, letting the model interpolate between hours it has effectively already seen. No row appears twice, yet the test set is not genuinely unseen.

A chronological split was used instead: train on 2010 to 2013, test on all of 2014. Both were computed to make the difference visible:

Split strategy	R²
Random shuffle	0.6207
Chronological (honest)	0.5071

Random splitting overstated performance by 0.11 R² points, a larger gap than the difference between the best and worst of the three models tested. All results below use the chronological split.

Results
Model	RMSE (µg/m³)	MAE (µg/m³)	R²
Linear Regression	76.93	54.41	0.3234
Random Forest	67.78	45.48	0.4748
XGBoost	65.66	44.54	0.5071

Baseline (always predicting the training mean) gives RMSE 93.53 µg/m³, so the selected model reduces error by 29.8%.

Top features

Rank	Feature	Importance
1	heating_season	0.247
2	season_Winter	0.135
3	temp_dewp_spread	0.107
4	cbwd_NW (north west wind)	0.070

Two of the four engineered features rank in the top three, including first place. Domain reasoning about Beijing's heating schedule mattered more than any raw weather reading.

Sanity check

Scenario	Predicted PM2.5	Band
Calm winter night in heating season	265.7 µg/m³	Hazardous
Mild autumn day, moderate wind	50.8 µg/m³	Unhealthy for Sensitive Groups
Windy summer afternoon with rain	0.9 µg/m³	Good
Limitations
Weather does not create pollution, it disperses or traps it. The model cannot see traffic, industry, construction, or agricultural burning. The unexplained variance is largely emissions, and no volume of weather data recovers it.
Severe episodes are under predicted. The residual plot shows the extremes sitting below the diagonal. Multi day smog builds up under sustained stagnation, and a model seeing only the current hour cannot represent that accumulation. These are exactly the days a warning system exists to flag.
No accumulation history, by the deliberate design choice explained above.
Beijing specific. The model learned one city's emission profile and terrain. The method transfers; these coefficients do not.
Data ends in 2014. Air quality policy has changed substantially since.
Next steps
Add lagged pollution features for a same day nowcasting variant, keeping the weather only model for genuine forecasting.
Reframe as classification over air quality bands, since users act on "should I go outside" rather than on a precise number.
Model log PM2.5 or weight training toward high pollution hours, to reduce under prediction at the extremes.
Retrain on recent data for a specific city using the same pipeline.

A longer discussion of the real world value and limits of this project is in CASE_STUDY.md.

Foundation Tasks 1 to 10

The ten assigned tasks that built up to the capstone.

Task 1: Exploratory Data Analysis

Notebook: Task_01_Titanic_EDA.ipynb | Dataset: Titanic

First look at the data before any modelling.

Loaded the dataset with pandas and inspected it using head(), info(), and describe()
Identified dataset shape and separated numerical from categorical features
Located missing values
Wrote a short data story summarising the findings

Dataset characteristics: 891 passenger records, 12 columns, target variable Survived. Missing values concentrated in Cabin, Age, and Embarked.

Task 2: Data Cleaning and Visualization

Notebook: Task_02_Titanic_Data_Cleaning_Visualization.ipynb | Dataset: Titanic

Cleaning decisions

Column	Action	Reasoning
Age	Filled with median	Skewed distribution, so median is more robust than mean
Embarked	Filled with mode	Only two values missing, and one port dominates
Cabin	Dropped	A very large proportion of values missing

Visualizations: age histogram, fare boxplot for outlier detection, survival rate bar chart by gender, and a correlation heatmap.

Key finding: Sex has the strongest relationship with survival. Female passengers survived at a substantially higher rate than male passengers.

Task 3: Titanic Survival Classification

Notebook: Task_03_Titanic_Classification.ipynb | Dataset: Titanic

First predictive model. Logistic Regression built inside a scikit-learn Pipeline with a ColumnTransformer and OneHotEncoder.

Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked Split: 80/20, stratified, random_state=42, giving 712 training and 179 test samples

Result: 80.45% accuracy

Confusion matrix:

text
[[98 12]
 [23 46]]

Reading it: 98 correctly predicted as not surviving, 46 correctly predicted as surviving, 12 false positives, and 23 false negatives. The model is noticeably better at identifying deaths than survivals, a bias that plain accuracy conceals. Task 5 addresses this directly.

Task 4: House Price Prediction with Linear Regression

Notebook: Task_04_House_Price_Regression.ipynb | Dataset: California Housing (built into scikit-learn)

First regression problem, predicting a continuous value rather than a yes/no outcome.

Features: MedInc, HouseAge, AveRooms, AveBedrms, Population Target: MedHouseVal, median house value in hundreds of thousands of dollars

Results

Metric	Value
RMSE	0.8021 (about $80,213)
R²	0.5090
Baseline RMSE (predicting the mean)	1.1449
Improvement over baseline	29.94%

What R² means here, in plain English: house prices vary a lot, and the model explains about 51% of that variation using its five features. A score of 1.0 would mean perfect prediction; 0.0 would mean no better than always quoting the average price. The remaining 49% is driven by factors the model was never given, most importantly location, since a coastal house is worth far more than an identical inland one.

Latitude and Longitude were deliberately excluded to stay within the 3 to 5 feature limit the task specified. Including them would likely improve the score considerably.

Task 5: Model Evaluation and Tuning, Beyond Accuracy

Notebook: Task_05_Model_Evaluation_Tuning.ipynb | Dataset: Titanic

Why accuracy alone is misleading

A DummyClassifier that ignores every feature and always predicts "did not survive" scores 61.45% accuracy on this dataset, while achieving 0.00 recall on survivors. It identifies no survivors at all. Accuracy hides that completely, because guessing the majority class is right whenever that class is the majority.

On a fraud dataset that is 99% legitimate, the same trick scores 99% accuracy and catches zero fraud. Task 9 demonstrates exactly that.

Hyperparameter tuning

GridSearchCV with 5-fold cross-validation across three hyperparameters (C, class_weight, solver), optimising F1 on the survived class rather than accuracy, since tuning for accuracy would simply reward the majority-class bias being corrected.

Best parameters: C=0.01, class_weight='balanced', solver='liblinear'

Before and after
Metric	Before (Task 3)	After (Tuned)	Change
Accuracy	0.8045	0.7765	-0.0279
Precision	0.7931	0.6883	-0.1048
Recall	0.6667	0.7681	+0.1014
F1-Score	0.7244	0.7260	+0.0016

Honest reading: accuracy went down and F1 improved by only 0.0016, which on a 179-row test set is noise rather than genuine improvement. What tuning actually achieved was a meaningful shift in the precision and recall balance: false negatives fell from 23 to 16, meaning seven more real survivors were identified, while false positives rose from 12 to 24. The defensible claim is that the model was rebalanced toward recall at roughly equal F1, not that it was made better.

Task 6: Customer Churn Prediction

Notebook: Task_06_Telco_Churn_Prediction.ipynb | Dataset: Telco Customer Churn (7,043 customers, 21 columns)

The data cleaning catch

TotalCharges arrives stored as text rather than a number, and eleven rows contain a blank space instead of a value. Every one of those rows has tenure = 0: these are brand-new customers who have not been billed yet.

They were therefore filled with 0, not the median. Using the median would have invented a billing history for customers who have none, telling the model that a brand-new customer had already spent an average amount. Looking at which rows are missing before choosing a strategy matters more than the strategy itself.

Class imbalance

Roughly 73% of customers stayed and 27% churned. Both models were trained with and without class_weight='balanced'.

Results

Metrics are for the churned class.

Model	Accuracy	Precision	Recall	F1
Logistic Regression	0.8055	0.6572	0.5588	0.6040
Decision Tree (max_depth=5)	0.7984	0.6347	0.5668	0.5989
Logistic Regression (balanced)	0.7381	0.5043	0.7834	0.6136
Decision Tree (balanced)	0.7544	0.5260	0.7567	0.6206
Overfitting demonstration

An unrestricted Decision Tree scores 99.8% on training data and 72.5% on test data. That 27-point gap is the signature of memorisation, and it is why the reported models cap depth at 5.

Top 3 churn drivers
Rank	Feature	Importance
1	Contract: Month-to-month	0.5140
2	Internet Service: Fiber optic	0.1633
3	Tenure	0.1567

Contract type alone accounts for over half of the tree's decision making.

Business summary

About one in four customers is leaving, and we can now identify which ones before they go. The clearest warning sign is contract type: month-to-month customers leave at a dramatically higher rate than those on annual contracts. Tenure is the next strongest signal, with most churn concentrated in the first months, meaning risk sits with new customers rather than long-standing ones. Customers on higher monthly charges, particularly fiber-optic subscribers without support add-ons, also leave more often.

The recommendation is to focus retention spending on new, month-to-month, high-bill customers within their first year. Offering an incentive to move onto an annual contract, or bundling a support add-on, targets the two factors the data flags most strongly. One caveat: the model identifies who is at risk, not why any individual is unhappy.

Task 7: Building an ML Pipeline

Notebook: Task_07_ML_Pipeline_Feature_Engineering.ipynb | Dataset: Titanic

The data leakage fix

Tasks 2, 3, and 5 filled missing ages using df['Age'].median() computed before the train/test split, meaning the median was calculated across all 891 passengers including the 179 that later became the test set. Test set information flowed into training preparation.

Moving SimpleImputer inside the pipeline makes this structurally impossible: the imputer learns its median from training data only, then applies that learned value to the test set. On Titanic the impact is small, and the pipeline reproduced the Task 3 accuracy of 0.8045 exactly, but the pattern matters enormously on problems where scaling or target-based encoding is involved.

Engineered features
Feature	Formula	Reasoning
FamilySize	SibSp + Parch + 1	Total group size is more meaningful than the two raw columns separately
IsAlone	FamilySize == 1	Travelling alone is qualitatively different from travelling in a group
Title	Extracted from Name	Earlier tasks discarded Name as unusable text, but Mr, Mrs, Miss, and Master encode age, sex, and status in one token
FarePerPerson	Fare / FamilySize	Fare is recorded per ticket, so a large family sharing one ticket looks wealthier than they are
Results
Approach	CV Mean Accuracy	CV Std Dev
Manual (Task 3 style)	0.7924	0.0174
Pipeline, original features	0.7935	0.0172
Pipeline, engineered features	0.8283	0.0078

Test set accuracy improved from 0.8045 to 0.8436.

Unlike Task 5, this is a real improvement: roughly 3.5 percentage points, about four standard deviations. The standard deviation also more than halved, meaning the model became more stable across folds as well as more accurate.

The final pipeline was saved with joblib, reloaded as a separate object, and confirmed to produce identical predictions.

Task 8: Ensemble Learning, Random Forest vs XGBoost

Notebook: Task_08_Ensemble_RandomForest_XGBoost.ipynb | Dataset: Telco Customer Churn

Results
Model	Accuracy	Precision	Recall	F1	ROC AUC
Logistic Regression (Task 6)	0.8055	0.6572	0.5588	0.6040	0.8421
Decision Tree (Task 6)	0.7984	0.6347	0.5668	0.5989	0.8297
Random Forest	0.8048	0.6725	0.5160	0.5840	0.8409
XGBoost	0.7935	0.6317	0.5321	0.5776	0.8397
Random Forest (balanced)	0.7644	0.5407	0.7460	0.6270	0.8413
XGBoost (balanced)	0.7601	0.5336	0.7647	0.6286	0.8365

Cross-validated ROC AUC: Random Forest 0.8456, Logistic Regression 0.8451, XGBoost 0.8384, Decision Tree 0.8300, with standard deviations around 0.012.

How the two ensembles differ

Random Forest builds all its trees in parallel and independently, each trained on a random sample of rows and choosing from a random subset of features at every split, with the final prediction a vote across the forest. XGBoost builds trees sequentially, each new tree trained specifically on the errors the current ensemble still makes. Random Forest reduces variance by averaging away the instability of individual deep trees; XGBoost reduces bias by having many shallow trees chip away at remaining error. In practice Random Forest is hard to break and works well at defaults, while XGBoost reaches a higher ceiling but is more sensitive to learning rate and depth.

The honest finding

The ensembles did not beat Logistic Regression on this dataset. The top three models sit within noise of each other on cross-validated ROC AUC. Churn here has substantial near-linear structure, and a well-specified linear model captures it about as well as a forest does.

Ensembles clearly beat the single Decision Tree, which is the claim that holds. But reaching for the most complex available model is not the same as good modelling, and a simpler model that matches on performance is usually the better production choice: faster to train, easier to explain, fewer ways to fail quietly.

Feature importance disagreement

The two ensembles rank features very differently despite scoring similarly. Random Forest leads with tenure (0.145), TotalCharges (0.115), and Contract_Month-to-month (0.111). XGBoost leads with Contract_Month-to-month (0.246). Random feature subsetting spreads credit across correlated features in a forest, while boosting concentrates it on whatever previous trees have not yet fixed. Importance describes the model, not reality.

Task 9: Handling Imbalanced and Messy Real-World Data

Notebook: Task_09_Imbalanced_Data_Fraud_Detection.ipynb | Dataset: Credit Card Fraud Detection (284,807 transactions)

The imbalance

492 frauds among 284,807 transactions, or 0.17%, which is one fraud per 578 legitimate transactions. Plotted on a linear axis the fraud bar is effectively invisible, so the notebook also shows a log-scale version.

Why accuracy is the wrong metric

A model that flags nothing as fraud scores 99.828% accuracy and catches 0 of 98 frauds in the test set.

Accuracy asks what fraction of all predictions were correct. When 99.83% of transactions are legitimate, answering "legitimate" every time is correct 99.83% of the time. The metric is dominated by the majority class, so the minority class, which is the entire reason the model exists, contributes almost nothing. Worse, an uncorrected model is pulled toward the same behaviour during training, because predicting the majority class also minimises its loss. The imbalance distorts both how the model is measured and what it learns.

Results
Model	Accuracy	Precision	Recall	F1	PR AUC
Baseline (no balancing)	0.9992	0.8289	0.6429	0.7241	0.7432
Class weighted	0.9755	0.0610	0.9184	0.1144	0.7159
SMOTE	0.9742	0.0580	0.9184	0.1092	0.7249
Undersampling	0.9603	0.0384	0.9184	0.0738	0.6778

The percentages hide the real story, so here are the counts:

Model	Frauds Caught	Frauds Missed	False Alarms
Baseline	63	35	13
Class weighted	90	8	1,386
SMOTE	90	8	1,461
Undersampling	90	8	2,252

Recall rose from 0.64 to 0.92, but F1 fell by a factor of six. At a fixed 0.5 threshold, rebalancing overcorrects badly on data this skewed: it catches 27 more frauds at the cost of over 1,300 additional false alarms. Reporting only the recall improvement would be technically true and genuinely misleading.

Threshold selection

Rebalancing and threshold selection are two separate levers, and rebalancing is the blunter one. Using explicit cost assumptions (a missed fraud costing 100 units, a false alarm costing 2), the optimal threshold for the class-weighted model is 0.97, not the default 0.50:

87 of 98 frauds caught
95 false alarms, down from 1,386
F1 recovers to 0.6214
Estimated cost falls from 3,572 to 1,290
A note on ROC AUC

ROC AUC exceeds 0.95 for every model here, which flatters them. The false positive rate divides false alarms by 284,000 legitimate transactions, so even thousands of false alarms barely move it. PR AUC never involves that denominator and stays honest, which is why it is reported alongside.

Avoiding SMOTE leakage

SMOTE was applied inside an imblearn pipeline so it runs during fitting only. Applying it before the train/test split would place synthetic frauds derived from real ones into both halves, producing impressive recall figures that mean nothing. This is one of the most common serious errors in imbalanced learning writeups.

Task 10: Deploying the Model as a Live Web App

Notebook: Task_10_Model_Deployment_Streamlit.ipynb | Files: app.py, titanic_pipeline.joblib, requirements.txt

Live app: https://badarrao-titanic-predictor.streamlit.app/

The Task 7 Titanic pipeline was selected for deployment, and not only on performance. It accepts raw passenger details and handles imputation, scaling, and encoding internally, so the app passes user input straight in without reproducing any preprocessing logic.

The alternatives were poor fits for a web interface: the Task 9 fraud model requires 28 anonymized PCA components as input, and the Task 6 churn model requires 19 fields. Titanic requires 8, all of them things a person can understand and enter.

Model performance: 0.8436 test accuracy, 0.8283 cross-validated (std 0.0078). For deployment the model was refitted on all 891 passengers, since the held-out estimate had already been taken.

On version pinning: requirements.txt is generated from the exact library versions in the training session. Streamlit Cloud builds a fresh environment from that file, and if it installs a different scikit-learn than the one that pickled the model, loading either fails outright or succeeds while behaving incorrectly. This is the most common reason a deployment works locally and dies online.

Results Summary
Task	Problem	Model	Key Metric
3	Titanic survival	Logistic Regression	80.45% accuracy
4	California house prices	Linear Regression	R² 0.5090, RMSE $80,213
5	Titanic, tuned	Logistic Regression + GridSearchCV	Recall 0.6667 to 0.7681
6	Telco churn	Logistic Regression	F1 0.6040 (churned class)
7	Titanic + pipeline	Logistic Regression + engineered features	0.8436 accuracy, CV 0.8283
8	Telco churn	Random Forest / XGBoost	ROC AUC 0.8409 / 0.8397
9	Credit card fraud	Logistic Regression + threshold tuning	87/98 frauds caught, F1 0.6214
10	Deployment	Task 7 pipeline	Live Streamlit app
Capstone	Beijing PM2.5 from weather	XGBoost	R² 0.5071, RMSE 65.66 µg/m³
What This Track Taught

Accuracy is often the wrong metric. A model that detects nothing scored 61% on Titanic and 99.8% on fraud detection. Precision, recall, F1, and PR AUC were needed to see what was actually happening.

Leakage is subtle and easy to miss. Filling missing values before splitting the data leaks test information into training. Pipelines prevent this structurally rather than relying on remembering to do it correctly.

Complexity is not automatically better. Ensembles tied with Logistic Regression on churn. The simpler model that matches on performance is usually the right production choice.

Feature engineering can beat algorithm selection. Four engineered Titanic features moved cross-validated accuracy by 3.5 points, more than switching model families achieved on other tasks.

Improvements must be weighed against noise. A +0.0016 F1 change on 179 test rows is not an improvement. Cross-validation standard deviations are what make that judgement possible.

How you evaluate can matter more than what you train. In the capstone, switching from a random split to a chronological one moved R² by 0.11, a larger swing than the gap between the best and worst model tested. On time series data the split strategy is not a detail.

Domain knowledge can outrank the data itself. The capstone's most important feature was a hand built flag for Beijing's coal heating season, which beat every raw weather measurement available.

Metric changes hide costs. Recall rising from 0.64 to 0.92 in fraud detection came with 1,373 additional false alarms. Reporting the counts alongside the percentages is what keeps the claim honest.

How to Run

In Google Colab (recommended)

Open any notebook and run all cells. Most load their datasets directly from public URLs, so no manual download is required.

Locally

bash
git clone https://github.com/BadarRao/neurofive-ml-track.git
cd neurofive-ml-track
pip install -r requirements.txt
jupyter notebook

Running either Streamlit app locally

bash
streamlit run capstone_app.py   # Air Quality Forecaster
streamlit run app.py            # Titanic Survival Predictor

Each opens at http://localhost:8501.

Note that requirements.txt is shared by both apps. It is a superset covering the capstone's XGBoost dependency as well as everything the Titanic app needs.
