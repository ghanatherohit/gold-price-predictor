from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

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

models = {c: joblib.load(f"saved_models/{c}.joblib") for c in best_models.keys()}
print("✅ All models loaded!")

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
        if not country or not date_str:
            return "Please select both country and date", 400
        try:
            dt = pd.to_datetime(date_str)

            # Prediction
            X_predict = pd.DataFrame([[dt.month, dt.year]], columns=['month','year'])
            pred = models[country].predict(X_predict)[0]

            # Historical 12-month graph
            months = pd.date_range(end=dt, periods=12, freq='M')
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
            print("Error:", e)
            return f"Error: {e}", 400

    return render_template('gold.html', countries=countries)

if __name__ == '__main__':
    app.run(debug=True)
