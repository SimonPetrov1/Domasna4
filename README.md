# Domasna3 – Microservices Refactor (Flask)

This folder contains a reference microservices split of the original monolithic Flask app.

## Services / Ports
- ui-service (main/orchestrator + UI): http://localhost:5000
- auth-service (users/login): http://localhost:5001
- technical-service (coins + technical indicators + signals): http://localhost:5002
- prediction-service (LSTM forecast API): http://localhost:5003

## Quick start
Open 4 terminals and run (each service uses its own venv):

### auth-service
```bash
cd auth-service
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### technical-service
```bash
cd technical-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### prediction-service
```bash
cd prediction-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### ui-service
```bash
cd ui-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Demo (curl)
Register + login via UI (which calls auth-service):
```bash
curl -X POST http://localhost:5000/register -H "Content-Type: application/json" -d '{"username":"demo","password":"demo"}'
curl -X POST http://localhost:5000/login -H "Content-Type: application/json" -d '{"username":"demo","password":"demo"}'
```

Coins + analysis (UI calls technical-service):
```bash
curl "http://localhost:5000/coins?limit=20"
curl "http://localhost:5000/analysis/BTC"
```

Prediction (UI calls prediction-service, which fetches history from technical-service):
```bash
curl "http://localhost:5000/predict/BTC?horizon=7"
```
