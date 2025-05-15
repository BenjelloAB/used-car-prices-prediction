# Car Price Prediction App 🚗

This project is a complete pipeline to scrape, clean, analyze, and predict used car prices. It includes data scraping notebooks, preprocessing tools, machine learning training and optimization scripts, and a web frontend powered by a Python server.

---

## 📁 Project Structure

### 🔍 Data Collection & Scraping
- `Scrapping_brands_models_collecting_selenium.ipynb`  
  Scrapes car brands and models using Selenium from Avito.

- `Scrapping_listings_data.ipynb`  
  Scrapes individual car listings and exports them for further processing.

---

### 🧼 Data Cleaning & Processing
- `data_cleaning.ipynb`  
  Handles cleaning of raw scraped data including column filtering, handling missing values, etc.

- `clean_data.csv`  
  The cleaned dataset ready for feature engineering and training.

---

### 📊 Data Analysis & Visualization
- `data_exploration_visualization.ipynb`  
  Exploratory data analysis and visualization of distributions, correlations, and trends.

---

### 🧠 Model Training & Optimization
- `training_optimization.ipynb`  
  Trains several regression models, tunes hyperparameters, and evaluates performance.

- `best_model_manual.pkl`  
  The final trained regression model serialized for inference.

---

### 🧪 Preprocessing Artifacts
- `brand_encoder.json` / `brand_smoothed_encoder.json`  
  Encoders for brand column (target encoded and smoothed).

- `model_encoder.json` / `model_smoothed_encoder.json`  
  Encoders for model column (target encoded and smoothed).

- `mean_log_price.json`  
  Stores the global mean of the log prices used during target encoding.

- `onehot_encoder.pkl`  
  Fitted OneHotEncoder for categorical features.

- `scaler.pkl`  
  Fitted StandardScaler used to scale numerical features.

- `feature_order.json`  
  Stores the expected input feature order for model prediction.

- `transmission_target_encoder.pkl`  
  Target encoder for the transmission column.

---

### 🌐 Web Interface
- `testing.html`  
  A frontend UI form for user input to predict car prices.

- `server.py`  
  A lightweight Python backend server that handles incoming requests and returns price predictions using the trained model.

---

### 📦 Other
- `merged_output_cp.csv`  
  An older or combined dataset used during development.

- `requirements.txt`  
  Lists the required Python libraries to run the project.

---

## 🔧 Setup Instructions

**Install dependencies:**

   ```bash
   pip install -r requirements.txt```
   
  ## Run the Server :
  ```bash
  python server.py```
