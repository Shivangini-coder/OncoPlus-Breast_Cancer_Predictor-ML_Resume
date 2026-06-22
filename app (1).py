import gradio as gr
import numpy as np
import json
import joblib

# ----------------------------
# Load Model, Scaler, Features
# ----------------------------

model = joblib.load("best_model_LR.pkl")
scaler = joblib.load("scaler.pkl")

with open("assets/features.json") as f:
    selected_features = json.load(f)

# --------------------------------------------------
# FIX: Normalize feature names (remove accidental “ ”)
# --------------------------------------------------

selected_features = [f.replace(" ", "_") for f in selected_features]

# -------------------------------------
# Human-readable Feature Name Dictionary
# -------------------------------------

feature_labels = {
    "radius_mean": "Mean Radius (Size of Nucleus)",
    "compactness_mean": "Compactness (Smoothness of Edges)",
    "concavity_mean": "Concavity (Indentation Level)",
    "concave_points_mean": "Concave Points (Sharp Indentations)",
    "symmetry_mean": "Mean Symmetry Score",
    "texture_se": "Texture Standard Error",
    "perimeter_se": "Perimeter Standard Error",
    "radius_worst": "Worst Radius Measurement",
    "texture_worst": "Worst Texture Measurement",
    "smoothness_worst": "Worst Smoothness",
    "compactness_worst": "Worst Compactness",
    "concavity_worst": "Worst Concavity",
    "concave_points_worst": "Worst Concave Points",
    "symmetry_worst": "Worst Symmetry"
}

# ------------------------------------------------
# Reasonable Value Ranges for Each Selected Feature
# ------------------------------------------------

feature_ranges = {
    "radius_mean": (6, 28),
    "compactness_mean": (0, 1),
    "concavity_mean": (0, 1),
    "concave_points_mean": (0, 1),
    "symmetry_mean": (0.1, 0.4),
    "texture_se": (0, 5),
    "perimeter_se": (0, 5),
    "radius_worst": (7, 36),
    "texture_worst": (10, 50),
    "smoothness_worst": (0.05, 0.25),
    "compactness_worst": (0, 1.2),
    "concavity_worst": (0, 1.5),
    "concave_points_worst": (0, 0.3),
    "symmetry_worst": (0.1, 1.0)
}

# ---------------------
# Prediction Function
# ---------------------

def predict(*values):
    x = np.array(values).reshape(1, -1)
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)[0]
    prob = model.predict_proba(x_scaled)[0][1]

    label = "🩺 Malignant" if pred == 1 else "🌸 Benign"
    percent = str(round(prob * 100, 2)) + "%"

    return label, prob, percent

# ---------------------
# Slider Reset Function
# ---------------------

def reset_slider(min_val, max_val):
    return (min_val + max_val) / 4


# ---------------------
# BUILD USER INTERFACE
# ---------------------

def build_ui():

    inputs = []

    with gr.Blocks(theme=gr.themes.Soft()) as demo:

        # Title
        gr.Markdown(
            "<h1 style='text-align:center;color:#c2185b;'>🎗 BREAST CANCER PREDICTION</h1>"
        )

        gr.Markdown(
            "<p style='text-align:center;'>Provide the diagnostic measurements below.<br>"
            "This ML model predicts whether the tumor is <b>Benign</b> or <b>Malignant</b>.</p>"
        )

        # Medical Disclaimer
        gr.Markdown(
            "<p style='text-align:center;color:#b00020;font-size:16px;'>"
            "⚠ This is an AI-based prediction tool. Always consult qualified medical professionals."
            "</p>"
        )

        # Feature sliders
        '''
        gr.Markdown("### 🩺 Input Features (14 selected)")

        for feature in selected_features:
            min_val, max_val = feature_ranges[feature]
            inputs.append(
                gr.Slider(
                    label=feature_labels[feature],
                    minimum=min_val,
                    maximum=max_val,
                    step=0.01,
                    interactive=True
                )
            )
        '''

                # Feature sliders (2-column layout)
        gr.Markdown("### 🩺 Input Features (14 selected features)")

        with gr.Column():
            for i in range(0, len(selected_features), 2):
                with gr.Row():

                    for feature in selected_features[i:i+2]:

                        min_val, max_val = feature_ranges[feature]
                        default_value = (min_val + max_val) / 4

                        with gr.Column():

                            # Create slider
                            slider = gr.Slider(
                                label=feature_labels[feature],
                                minimum=min_val,
                                maximum=max_val,
                                value=default_value,
                                step=0.01,
                                interactive=True,
                            )

                            # Refresh Button
                            refresh_btn = gr.Button("↻", size="sm")

                            # Attach refresh reset
                            refresh_btn.click(
                                fn=reset_slider,
                                inputs=[gr.Number(min_val, visible=False),
                                        gr.Number(max_val, visible=False)],
                                outputs=slider
                            )

                            inputs.append(slider)

        # Predict Button
        with gr.Row():
            predict_btn = gr.Button(
                "🔍 Predict",
                variant="primary",
                elem_id="predict-btn"
            )

        # Outputs
        output_label = gr.Textbox(label="Predicted Label")
        output_prob = gr.Textbox(label="Malignant Probability (0–1)")
        output_percent = gr.Textbox(label="Probability (%)")

        # Click action
        predict_btn.click(
            fn=predict,
            inputs=inputs,
            outputs=[output_label, output_prob, output_percent]
        )

        # Custom CSS for styling
        demo.css = """
        #predict-btn {
            background: linear-gradient(to right, #ff4081, #f50057);
            color: white;
            font-weight: bold;
            transition: 0.2s ease;
            border-radius: 12px;
            padding: 12px;
        }
        #predict-btn:hover {
            transform: scale(1.05);
            background: linear-gradient(to right, #f50057, #c51162);
        }
        """

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
