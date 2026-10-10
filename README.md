# E-Learning Student Behavioral Segmentation Engine

## Overview
A machine learning application that classifies student behavioral
patterns into four segments using a trained K-Means model.

## Technology
- Python
- Scikit-learn
- Pandas
- Gradio
- Pytest
- GitHub Actions

## Model artifacts
- `models/final_kmeans_model.pkl`
- `models/final_scaler.pkl`
- `models/project_config.json`

## Run locally
Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
python app.py
```

## Run tests

```bash
pytest -v test_model.py
```

## CI pipeline
GitHub Actions installs the dependencies and runs automated model
smoke tests whenever code is pushed or a pull request is opened.
