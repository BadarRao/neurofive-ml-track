# Air Quality Forecaster: Predicting PM2.5 from Weather Conditions

**NeuroFive ML Track, Capstone Project**

**Live app:** _add your Streamlit URL here after deploying_

**Notebook:** `Capstone_Air_Quality_Prediction.ipynb`

---

## Problem Statement

Fine particulate matter (PM2.5) is the air pollutant most strongly linked to human harm. The particles measure under 2.5 micrometres across, small enough to bypass the body's normal filtering, reach deep into the lungs, and pass into the bloodstream. Long term exposure is associated with heart disease, stroke, reduced lung development in children, and shortened life expectancy.

Cities across Asia experience severe seasonal smog, and residents want to know one practical thing: is tomorrow going to be bad?

Pollution monitoring stations are expensive and sparse. Weather forecasts are free, universal, and available days ahead. That gap motivates the question this project answers:

> **Can PM2.5 concentration be predicted from weather conditions alone?**

If it can, any location with a weather forecast could issue an air quality warning without installing a single sensor.

This is a **regression** problem. The target is a continuous concentration in µg/m³.

---

## Dataset

Hourly measurements from Beijing, 2010 to 2014, pairing PM2.5 readings taken at the US Embassy with meteorological data from Beijing Capital International Airport.

| Property | Value |
|---|---|
| Raw records | 43,824 hourly readings |
| Records after cleaning | 41,757 |
| Features | 9 raw, expanded to 13 after engineering |
| Target | `pm2.5`, concentration in µg/m³ |
| Missing values | 2,067, all in the target column |

Beijing was chosen because few cities publish five continuous years of paired pollution and weather data openly. The method transfers to anywhere with comparable records.

**A framing statistic:** 88.1% of hours in this five year record exceeded the WHO 24 hour guideline of 15 µg/m³.

---

## Approach

### 1. Cleaning

All 2,067 missing values sat in the target column. These rows were **dropped rather than imputed**, which is a different decision from earlier tasks in this repository where missing *features* were filled with a median or mode. Imputing a target means inventing the answer and then training the model to reproduce that invention, so every metric would partly measure how well the model copied a fabricated number. Dropping cost 4.7% of the data.

### 2. Exploratory analysis

- Wind speed shows the strongest and most nearly monotonic relationship with pollution: sustained wind disperses particulates, calm air lets them accumulate.
- Winter months are far more polluted than summer.
- No single weather variable correlates strongly with PM2.5 on its own, which correctly predicted that tree based models would beat Linear Regression.

### 3. Feature engineering

Four features were added, each motivated by a physical mechanism rather than by trying combinations at random.

| Feature | Definition | Reasoning |
|---|---|---|
| `heating_season` | 15 Nov to 15 Mar | Beijing runs coal fired district heating on a fixed winter schedule. The raw `month` column captures this only indirectly, since the window splits mid month. |
| `temp_dewp_spread` | `TEMP - DEWP` | Indicates how close air is to saturation. A small spread signals humid, stagnant conditions that trap particulates near ground level. |
| `is_night` | Before 06:00 or from 20:00 | Overnight cooling forms temperature inversions, a lid of warm air that stops pollution dispersing upward. |
| `season` | Meteorological season | A coarse grouping for tree models to split on. |

**Deliberate omission:** no lagged PM2.5 features, despite yesterday's pollution being the single strongest available predictor. The project's purpose is forecasting from a *weather forecast*, and a user checking the app in advance has no recent pollution reading to supply. Including lags would have inflated the metrics while producing a model that cannot serve its stated purpose.

### 4. Evaluation strategy

This is time series data, so `train_test_split` with a random shuffle is the wrong tool. Pollution at 3pm closely resembles pollution at 2pm and 4pm. A random split scatters neighbouring hours across train and test, letting the model interpolate between hours it has effectively already seen. No row appears twice, yet the test set is not genuinely unseen.

A **chronological split** was used instead: train on 2010 to 2013, test on all of 2014.

Both were computed to make the difference visible:

| Split strategy | R² |
|---|---|
| Random shuffle | 0.6207 |
| Chronological (honest) | **0.5071** |

Random splitting overstated performance by **0.11 R² points**, a larger gap than the difference between the best and worst of the three models tested. All results reported here use the chronological split.

---

## Results

Three models trained and evaluated on 2014, a year never seen during training.

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² |
|---|---|---|---|
| Linear Regression | 76.93 | 54.41 | 0.3234 |
| Random Forest | 67.78 | 45.48 | 0.4748 |
| **XGBoost** | **65.66** | **44.54** | **0.5071** |

Baseline (always predicting the training mean) gives RMSE 93.53 µg/m³, so the selected model reduces error by **29.8%**.

### Top features

| Rank | Feature | Importance |
|---|---|---|
| 1 | `heating_season` | 0.247 |
| 2 | `season_Winter` | 0.135 |
| 3 | `temp_dewp_spread` | 0.107 |
| 4 | `cbwd_NW` (north west wind) | 0.070 |

Two of the four engineered features rank in the top three, including first place. The domain reasoning about Beijing's heating schedule turned out to matter more than any raw weather reading.

### Sanity check

| Scenario | Predicted PM2.5 | Band |
|---|---|---|
| Calm winter night in heating season | 265.7 µg/m³ | Hazardous |
| Mild autumn day, moderate wind | 50.8 µg/m³ | Unhealthy for Sensitive Groups |
| Windy summer afternoon with rain | 0.9 µg/m³ | Good |

---

## Limitations

- **Weather does not create pollution, it disperses or traps it.** The model cannot see traffic, industry, construction, or agricultural burning. The unexplained variance is largely emissions, and no volume of weather data recovers it.
- **Severe episodes are under predicted.** The residual plot shows the extremes sitting below the diagonal. Multi day smog builds up under sustained stagnation, and a model seeing only the current hour cannot represent that accumulation. These are exactly the days a warning system exists to flag.
- **No accumulation history**, by the deliberate design choice explained above.
- **Beijing specific.** The model learned one city's emission profile and terrain. The method transfers; these coefficients do not.
- **Data ends in 2014.** Air quality policy has changed substantially since.

---

## Next Steps

1. Add lagged pollution features for a same day nowcasting variant, keeping the weather only model for genuine forecasting.
2. Reframe as classification over air quality bands, since users act on "should I go outside" rather than on a precise number.
3. Model log PM2.5 or weight training toward high pollution hours, to reduce under prediction at the extremes.
4. Retrain on recent data for a specific city using the same pipeline.

---

## How to Run This Project

### Option 1: Use the live app

Open the Streamlit link at the top of this README. No installation needed.

The app is hosted on Streamlit Community Cloud's free tier, which sleeps after 12 hours without traffic. If you see a sleep screen, click "Yes, get this app back up!" and it loads in about 30 seconds.

### Option 2: Run the notebook

Open `Capstone_Air_Quality_Prediction.ipynb` in Google Colab and run all cells. The dataset loads from a public URL, so there is nothing to download. The notebook generates `aqi_model.joblib`, `capstone_app.py`, and `requirements.txt`.

### Option 3: Run the app locally

```bash
git clone https://github.com/BadarRao/neurofive-ml-track.git
cd neurofive-ml-track
pip install -r requirements.txt
streamlit run capstone_app.py
```

Opens at `http://localhost:8501`.

---

## Project Files

| File | Purpose |
|---|---|
| `Capstone_Air_Quality_Prediction.ipynb` | Full analysis, from raw data to saved model |
| `capstone_app.py` | Streamlit web application |
| `aqi_model.joblib` | Trained XGBoost pipeline |
| `requirements.txt` | Pinned dependencies |

## Tech Stack

Python, pandas, NumPy, scikit-learn, XGBoost, Matplotlib, seaborn, Streamlit, joblib
