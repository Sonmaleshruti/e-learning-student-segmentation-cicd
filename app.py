import os
import json
from pathlib import Path

import gradio as gr
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "project_config.json", "r") as file:
    CONFIG = json.load(file)

FEATURES = CONFIG["features"]
SEGMENTS = CONFIG["behavior_segments"]

model = joblib.load(MODEL_DIR / "final_kmeans_model.pkl")
scaler = joblib.load(MODEL_DIR / "final_scaler.pkl")


def predict_student_segment(student_data):
    if student_data is None:
        raise gr.Error("Please enter student feature values.")

    if not isinstance(student_data, pd.DataFrame):
        student_data = pd.DataFrame(student_data, columns=FEATURES)

    missing = [feature for feature in FEATURES
               if feature not in student_data.columns]
    if missing:
        raise gr.Error("Missing features: " + ", ".join(missing))

    ordered = student_data[FEATURES].copy()

    for feature in FEATURES:
        ordered[feature] = pd.to_numeric(
            ordered[feature], errors="raise"
        )

    if ordered.empty or ordered.isnull().any().any():
        raise gr.Error("Enter valid numeric values for every feature.")

    scaled = scaler.transform(ordered)
    clusters = model.predict(scaled)

    result = student_data.copy()
    result["predicted_cluster"] = clusters
    result["predicted_behavior_segment"] = [
        SEGMENTS[str(int(cluster))] for cluster in clusters
    ]

    return result


with gr.Blocks(title=CONFIG["project_name"]) as demo:
    gr.Markdown("# E-Learning Student Behavioral Segmentation Engine")
    gr.Markdown(
        "Enter the 20 behavioral features for each student. "
        "Each row represents one student."
    )

    input_table = gr.Dataframe(
        headers=FEATURES,
        datatype=["number"] * len(FEATURES),
        value=[[0] * len(FEATURES)],
        row_count=(1, "dynamic"),
        col_count=(len(FEATURES), "fixed"),
        type="pandas",
        label="Student behavioral features",
        interactive=True
    )

    predict_button = gr.Button("Predict Student Segments")
    output_table = gr.Dataframe(
        label="Prediction Results",
        type="pandas"
    )

    predict_button.click(
        fn=predict_student_segment,
        inputs=input_table,
        outputs=output_table
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
