# Production Error Check (Windows + Waitress)

## 1) Install dependencies
```powershell
cd "C:\Users\dell\Desktop\Rohit\CODES\gold"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2) Run in production-like mode
```powershell
$env:APP_ENV="production"
python -m waitress --host=0.0.0.0 --port=8000 app:app
```

## 3) Capture logs to file
```powershell
New-Item -ItemType Directory -Force logs | Out-Null
$env:APP_ENV="production"
python -m waitress --host=0.0.0.0 --port=8000 app:app *> logs\\prod.log
```

## 4) Trigger checks
- Open `http://127.0.0.1:8000/`
- Open `http://127.0.0.1:8000/gold`
- Submit:
  - Missing fields
  - Invalid date
  - Country without a model file

## 5) Inspect latest errors
```powershell
Get-Content logs\prod.log -Tail 100
Get-Content logs\app.log -Tail 100
```

## Notes
- User-facing production errors are generic (`500`) to avoid leaking tracebacks.
- Full exception details are written to `logs\app.log`.
- Monthly frequency uses `freq='ME'` to avoid pandas `M` offset errors.
