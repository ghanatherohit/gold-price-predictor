from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
import os
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
IS_PRODUCTION = os.getenv("APP_ENV", "").lower() == "production"


def configure_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"),
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    app.logger.handlers.clear()
    app.logger.propagate = True


configure_logging()

# --- Load all models ---
best_models = {
    'United_StatesUSD': 'LinearRegression',
    'EuropeEUR': 'Stacking',
    'JapanJPY': 'Stacking',
    'United_KingdomGBP': 'LinearRegression',
    'CanadaCAD': 'Stacking',
    'SwitzerlandCHF': 'LinearRegression',
    'IndiaINR': 'Stacking',
    'ChinaCNY': 'LinearRegression',
    'TurkeyTRY': 'Stacking',
    'Saudi_ArabiaSAR': 'LinearRegression',
    'IndonesiaIDR': 'Stacking',
    'United_Arab_EmiratesAED': 'LinearRegression',
    'ThailandTHB': 'LinearRegression',
    'VietnamVND': 'LinearRegression',
    'EgyptEGP': 'Stacking',
    'South_KoreanKRW': 'LinearRegression',
    'AustraliaAUD': 'LinearRegression',
    'South_AfricaZAR': 'Stacking'
}


# Load models with error handling
models = {}
for c in best_models.keys():
    try:
        models[c] = joblib.load(f"saved_models/{c}.joblib")
        app.logger.info(f"Model loaded: {c}")
    except Exception as e:
        app.logger.error(f"Failed to load model {c}: {e}")
        # Optionally, raise or continue depending on criticality
        raise
app.logger.info("✅ All models loaded!")

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/gold', methods=['GET', 'POST'])
def gold():

    countries = list(best_models.keys())
    if request.method == 'POST':
        country = request.form.get('country')
        date_str = request.form.get('date')  # YYYY-MM-DD
        # Input validation
        if not country or not date_str:
            app.logger.warning("Country or date not provided in form.")
            return "Please select both country and date", 400
        if country not in models:
            app.logger.warning(f"Invalid country selected: {country}")
            return "Invalid country selected", 400
        try:
            dt = pd.to_datetime(date_str, errors='raise')
        except Exception as e:
            app.logger.warning(f"Invalid date format: {date_str} | {e}")
            return "Invalid date format", 400
        try:
            # Prediction
            X_predict = pd.DataFrame([[dt.month, dt.year]], columns=['month','year'])
            pred = models[country].predict(X_predict)[0]

            # Historical 12-month graph
            months = pd.date_range(end=dt, periods=12, freq='ME')
            X_hist = pd.DataFrame([[m.month,m.year] for m in months], columns=['month','year'])
            y_hist = models[country].predict(X_hist)

            # Plot
            plt.figure(figsize=(6,4))
            plt.plot(months, y_hist, marker='o', color='gold')
            plt.title(f"{country} Gold Price Last 12 Months")
            plt.xlabel("Date")
            plt.ylabel("Gold Price")
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Convert plot to base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            return render_template('result.html', country=country, date=dt.date(), pred=pred, plot_url=plot_url)
        except Exception as e:
            app.logger.error(f"Prediction or plotting error: {e}")
            return f"Error: {e}", 400

    return render_template('gold.html', countries=countries)

# Note: For production, use a WSGI server like Gunicorn or Waitress.
if __name__ == '__main__':
    app.run(debug=not IS_PRODUCTION)
