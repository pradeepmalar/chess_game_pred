import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler 

# ======================
# CONFIG
# ======================
MODEL_PATH = "models/chess_model_with_artifacts.joblib"
CLASS_ORDER = ["white_win", "draw", "black_win"]
DECISIVENESS_BIAS_C = 0.3 # Use the optimal factor found during tuning

# ======================
# Load model & setup
# ======================
@st.cache_resource
def load_resources():
    """Loads the trained models and necessary feature artifacts."""
    try:
        resources = joblib.load(MODEL_PATH)
        return (resources["step1"], 
                resources["step2"], 
                resources["scaler"],
                resources["freq_maps"],
                resources["encoding_maps"],
                resources["features_ordered"],
                resources["decisiveness_c"]
                )
    except FileNotFoundError:
        st.error(f"Model file not found. Please run train.py first to create {MODEL_PATH}")
        return None, None, None, None, None, None, None

# Load all artifacts
(step1_model, step2_model, scaler, freq_maps, encoding_maps, 
 features_ordered, decisiveness_c) = load_resources()

# ======================
# Prediction function (UPDATED)
# ======================
def predict_outcome(white_elo, black_elo, eco, opening):
    """
    Predicts the chess game outcome using the two-step model and artifacts.
    """
    if not all([step1_model, scaler, freq_maps, encoding_maps, features_ordered]):
        return {"error": "Prediction artifacts missing. Please check the model file path and run train.py."}
    
    # 1. Base Feature Engineering
    game_data = pd.DataFrame({
        "white_elo": [white_elo],
        "black_elo": [black_elo],
        "eco": [eco],
        "opening": [opening]
    })
    
    game_data["elo_diff"] = game_data["white_elo"] - game_data["black_elo"]
    game_data["elo_diff_sq"] = game_data["elo_diff"]**2
    game_data["elo_ratio"] = game_data["white_elo"] / game_data["black_elo"]
    game_data["elo_ratio"] = game_data["elo_ratio"].clip(0.5, 2.0)
    
    # 2. Log Frequency Encoding (using maps saved from training)
    for col in ["eco", "opening"]:
        if col in freq_maps:
            freq_map = freq_maps[col]
            # Fillna(0) for unseen categories
            game_data[f'{col}_log_freq'] = np.log1p(
                game_data[col].map(freq_map).fillna(0)
            )
            
    # 3. Target Encoding (using maps saved from training)
    for col in ["eco", "opening"]:
        if col in encoding_maps:
            map_data = encoding_maps[col]
            for cls in CLASS_ORDER:
                mapping = map_data[cls].to_dict()
                global_mean = map_data[cls].mean()
                # Map and fill unseen categories with the global mean
                game_data[f"{col}_prob_{cls}"] = game_data[col].map(mapping).fillna(global_mean)

    # 4. Final Feature Vector Construction and Scaling
    # Use the features_ordered list to ensure the correct column order
    try:
        X = game_data[features_ordered].values
        X_scaled = scaler.transform(X)
    except KeyError as e:
        return {"error": f"Feature mismatch: Missing column {e}. Check feature names."}
    
    # 5. Model Prediction
    # Step 1: Draw vs Non-draw
    step1_proba = step1_model.predict_proba(X_scaled)[0]
    # Note: In XGBoost binary classifier (Draw=1 vs Non-Draw=0), index 0 is Non-Draw.
    p_nondraw = step1_proba[0] 
    p_draw = step1_proba[1] 

    # Step 2: White vs Black win (index 1 is White Win)
    p_white_cond = 0.5
    p_black_cond = 0.5
    if step2_model is not None:
        step2_proba = step2_model.predict_proba(X_scaled)[0]
        p_white_cond = step2_proba[1] 
        p_black_cond = step2_proba[0] 

    # 6. Combine probabilities (Unbiased)
    p_white = p_white_cond * p_nondraw
    p_black = p_black_cond * p_nondraw

    # 7. Apply Final Decisiveness Bias Factor (C)
    C = decisiveness_c 
    
    biased_proba_vector = np.array([p_white, p_draw, p_black])
    
    # Reduce Draw Probability (index 1)
    biased_proba_vector[1] = np.clip(biased_proba_vector[1] - C, 0.0, 1.0)
    
    # Re-normalize
    total = biased_proba_vector.sum()
    if total > 0:
        final_probas = biased_proba_vector / total
    else:
        # Should not happen, but safe fallback
        final_probas = np.array([1/3, 1/3, 1/3]) 

    return {
        "white_win": final_probas[0],
        "draw": final_probas[1],
        "black_win": final_probas[2],
    }


# ======================
# UI
# ======================
st.set_page_config(page_title="♟️ Chess Outcome Predictor", layout="centered")

st.title("♟️ Chess Outcome Prediction Dashboard")
st.markdown("Enter player ratings and opening to predict the game result.")

if step1_model is None:
    st.warning("Prediction models could not be loaded. Please ensure 'train.py' has been run successfully.")
else:
    col1, col2 = st.columns(2)

    with col1:
        # Use default values that were shown in the prompt images
        white_elo = st.number_input("White Elo", min_value=600, max_value=3000, value=2738, step=1) 

    with col2:
        black_elo = st.number_input("Black Elo", min_value=600, max_value=3000, value=2712, step=1)

    opening_name = st.text_input("Opening Name (optional)", value="Queen's Gambit")
    # ECO must be provided for the model to work
    opening_eco = st.text_input("Opening ECO (Required for accurate prediction)", value="D06") 

    if st.button("🔮 Predict"):
        if not opening_eco:
             st.error("Opening ECO is required for feature encoding.")
        else:
            proba = predict_outcome(white_elo, black_elo, opening_eco, opening_name)
            
            if "error" in proba:
                st.error(proba["error"])
            else:
                st.subheader("Prediction Results")
                st.write(f"**White Win Probability:** {proba['white_win']:.2%}")
                st.write(f"**Draw Probability:** {proba['draw']:.2%}")
                st.write(f"**Black Win Probability:** {proba['black_win']:.2%}")

                # Prepare data for a stacked bar chart visualization
                chart_data = pd.DataFrame({
                    "Prediction": [proba['white_win'], proba['draw'], proba['black_win']]
                }, index=["White Win", "Draw", "Black Win"])
                
                st.bar_chart(chart_data)