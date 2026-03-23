import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, f1_score
import xgboost as xgb
import joblib
import os 
from collections import Counter 

# =====================
# CONFIG (Using best performing parameters)
# =====================
DATA_PATH = r"C:\Users\chall\Documents\chess_game_pred\data\parsed\chess_games.feather"
# CHANGE PATH TO INCLUDE ALL ARTIFACTS
MODEL_PATH = "models/chess_model_with_artifacts.joblib"
PRED_CSV = "models/validation_preds_final_optimal.csv"

WHITE_ELO = "white_elo"
BLACK_ELO = "black_elo"
RESULT_COL = "result"
OPENING_COLS = ["eco", "opening"] 

CLASS_ORDER = ["white_win", "draw", "black_win"]

# FINAL OPTIMAL HYPERPARAMETERS (Derived from tuning runs)
TUNING_PARAMS_STEP1 = { 'max_depth': 8, 'n_estimators': 700, 'learning_rate': 0.01, 'gamma': 0.1 }
TUNING_PARAMS_STEP2 = { 'max_depth': 10, 'n_estimators': 850, 'learning_rate': 0.1, 'gamma': 0 }
DECISIVENESS_BIAS_C = 0.3 

# =====================
# Feature Engineering Functions
# =====================
def add_log_frequency_features(df_train, df_test, col):
    """Calculates log frequency based on training set and applies to both."""
    freq_map = df_train[col].value_counts().to_dict()
    df_train[f'{col}_freq'] = df_train[col].map(freq_map).fillna(0)
    df_test[f'{col}_freq'] = df_test[col].map(freq_map).fillna(0)
    df_train[f'{col}_log_freq'] = np.log1p(df_train[f'{col}_freq'])
    df_test[f'{col}_log_freq'] = np.log1p(df_test[f'{col}_freq'])
    return df_train.drop(columns=[f'{col}_freq']), df_test.drop(columns=[f'{col}_freq']), freq_map

def load_data(path):
    df = pd.read_feather(path)
    def to_result_label(x):
        if pd.isna(x): return np.nan
        if isinstance(x, (int,float)):
            if x==1.0: return "white_win"
            if x==0.0: return "black_win"
            if x==0.5: return "draw"
        key = str(x).strip().lower().replace("½","1/2")
        if key in ["1-0","w","white","white win"]: return "white_win"
        if key in ["0-1","b","black","black win"]: return "black_win"
        if key in ["1/2-1/2","draw","d","1/2"]: return "draw"
        return np.nan

    df["__result__"] = df[RESULT_COL].apply(to_result_label)
    df = df.dropna(subset=["__result__"])
    df = df.assign(
        white_elo = df["white_elo"].replace(0, np.nan),
        black_elo = df["black_elo"].replace(0, np.nan)
    )
    df.dropna(subset=["white_elo","black_elo"], inplace=True)
    df = df.assign(
        elo_diff = df["white_elo"] - df["black_elo"],
        elo_diff_sq = (df["white_elo"] - df["black_elo"])**2,
        elo_ratio = (df["white_elo"] / df["black_elo"]).replace([np.inf,-np.inf], np.nan)
    )
    df.dropna(subset=["elo_ratio"], inplace=True)
    df["elo_ratio"] = df["elo_ratio"].clip(0.5, 2.0)
    return df.reset_index(drop=True)

def target_encode_openings(df, cat_cols, y_col="__result__", n_splits=5):
    df_encoded = df.copy()
    encoding_maps = {} 
    for col in cat_cols:
        if col not in df.columns: continue
        for cls in CLASS_ORDER:
            df_encoded[f"{col}_prob_{cls}"] = 0.0

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        global_map = df.groupby(col)[y_col].value_counts(normalize=True).unstack().reindex(columns=CLASS_ORDER).fillna(0)
        encoding_maps[col] = global_map
        
        for train_idx, val_idx in skf.split(df, df[y_col]):
            train, val = df.iloc[train_idx], df.iloc[val_idx]
            mapping = train.groupby(col)[y_col].value_counts(normalize=True).unstack().reindex(columns=CLASS_ORDER).fillna(0)
            for cls in CLASS_ORDER:
                df_encoded.loc[val_idx, f"{col}_prob_{cls}"] = val[col].map(mapping[cls]).fillna(mapping[cls].mean())
    return df_encoded, encoding_maps

def prepare_features(df, scaler=None):
    cat_cols = [c for c in OPENING_COLS if c in df.columns]
    
    log_freq_cols = [c for c in df.columns if c.endswith('_log_freq')]
    numeric_cols = [WHITE_ELO, BLACK_ELO, "elo_diff", "elo_diff_sq", "elo_ratio"] + log_freq_cols
    
    te_cols = [c for c in df.columns if any(c.startswith(col+"_prob_") for col in cat_cols)]
    
    X_num = df[numeric_cols].values
    X_te = df[te_cols].values if te_cols else np.empty((len(df),0))

    X = np.hstack([X_num, X_te])
    
    # If scaler is None, we are in the training phase and must fit a new one
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, scaler
    # If scaler is provided, we are in the deployment/validation phase and transform only
    else:
        X = scaler.transform(X)
        return X

def train_binary_xgb(X_train, y_train, X_valid, y_valid, 
                     scale_pos_weight=1.0, 
                     max_depth=6, 
                     n_estimators=300, 
                     learning_rate=0.05,
                     gamma=1): 
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        tree_method="hist",
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42,
        scale_pos_weight=scale_pos_weight, 
        gamma=gamma 
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_valid)[:,1] 
    y_pred = (y_proba >= 0.5).astype(int)
    return model, y_pred, y_proba

# =====================
# MAIN TRAINING PIPELINE
# =====================
def main(params_step1, params_step2):
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} games from {DATA_PATH}")

    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42, stratify=df["__result__"])
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    # 1. ADD LOG FREQUENCY FEATURES and SAVE MAPS
    df_train, df_valid, eco_freq_map = add_log_frequency_features(df_train, df_valid, "eco")
    df_train, df_valid, opening_freq_map = add_log_frequency_features(df_train, df_valid, "opening")
    freq_maps = {"eco": eco_freq_map, "opening": opening_freq_map}

    # 2. TARGET ENCODE and SAVE MAPS
    df_train_enc, encoding_maps_train = target_encode_openings(df_train, OPENING_COLS)
    # Apply saved encoding maps (from training) to validation set for testing
    df_valid_enc = df_valid.copy()
    for col in OPENING_COLS:
        for cls in CLASS_ORDER:
            mapping = encoding_maps_train[col][cls].to_dict()
            global_mean = encoding_maps_train[col][cls].mean()
            df_valid_enc[f"{col}_prob_{cls}"] = df_valid_enc[col].map(mapping).fillna(global_mean)
    
    # 3. PREPARE FEATURES and SAVE SCALER
    X_train, scaler = prepare_features(df_train_enc)
    X_valid = prepare_features(df_valid_enc, scaler=scaler)
    
    # 4. CALCULATE CLASS WEIGHTS
    train_counts = Counter(df_train_enc["__result__"])
    weight_step1 = train_counts["draw"] / (train_counts["white_win"] + train_counts["black_win"])
    weight_step2 = train_counts["black_win"] / max(train_counts["white_win"], 1)
    
    print(f"\nCalculated Step 1 scale_pos_weight (Non-Draw): {weight_step1:.2f}")
    print(f"Calculated Step 2 scale_pos_weight (White Win): {weight_step2:.2f}")

    # 5. TRAIN MODELS
    y_train_step1 = (df_train_enc["__result__"]=="draw").astype(int)
    y_valid_step1 = (df_valid_enc["__result__"]=="draw").astype(int)
    
    print("\n=== STEP 1: Draw vs Non-Draw ===")
    model_step1, pred_step1, proba_step1 = train_binary_xgb(X_train, y_train_step1, X_valid, y_valid_step1, 
                                                   scale_pos_weight=weight_step1, **params_step1)
    print(f"Step 1 Accuracy (Draw/Non-Draw): {accuracy_score(y_valid_step1, pred_step1):.4f}")
    print(f"Step 1 F1 (Draw/Non-Draw): {f1_score(y_valid_step1, pred_step1):.4f}")
    
    P_DRAW = proba_step1
    P_NON_DRAW = 1.0 - P_DRAW

    non_draw_train_mask = df_train_enc["__result__"]!="draw"
    non_draw_valid_mask = df_valid_enc["__result__"]!="draw"
    X_train_step2 = X_train[non_draw_train_mask]
    y_train_step2 = (df_train_enc.loc[non_draw_train_mask,"__result__"]=="white_win").astype(int)
    X_valid_step2 = X_valid[non_draw_valid_mask]
    y_valid_step2 = (df_valid_enc.loc[non_draw_valid_mask,"__result__"]=="white_win").astype(int) # Defined here
    
    P_WHITE_GIVEN_NON_DRAW = np.full(len(df_valid_enc), 0.5)
    
    if len(np.unique(y_train_step2)) >= 2:
        print("\n=== STEP 2: White vs Black (non-draw only) ===")
        model_step2, pred_step2, proba_step2 = train_binary_xgb(
            X_train_step2, y_train_step2, X_valid_step2, y_valid_step2, 
            scale_pos_weight=weight_step2, **params_step2)
        P_WHITE_GIVEN_NON_DRAW[non_draw_valid_mask] = proba_step2
        
        print(f"Step 2 Accuracy (White/Black only): {accuracy_score(y_valid_step2, pred_step2):.4f}")
        print(f"Step 2 F1 (White/Black only): {f1_score(y_valid_step2, pred_step2):.4f}")
    else:
        model_step2 = None

    P_BLACK_GIVEN_NON_DRAW = 1.0 - P_WHITE_GIVEN_NON_DRAW

    # 6. COMBINE PROBABILITIES
    proba_white_win = P_WHITE_GIVEN_NON_DRAW * P_NON_DRAW
    proba_draw = P_DRAW
    proba_black_win = P_BLACK_GIVEN_NON_DRAW * P_NON_DRAW
    final_proba = np.column_stack([proba_white_win, proba_draw, proba_black_win])
    
    # Apply Bias Factor C
    C = DECISIVENESS_BIAS_C
    print(f"\n=== COMBINING PROBABILITIES (C={C}) ===")
    
    biased_proba = final_proba.copy()
    biased_proba[:, 1] = np.clip(biased_proba[:, 1] - C, 0.0, 1.0)
    row_sums = biased_proba.sum(axis=1, keepdims=True)
    biased_proba = np.divide(biased_proba, row_sums, out=np.zeros_like(biased_proba), where=row_sums!=0)
    final_pred = np.argmax(biased_proba, axis=1)

    # 7. Evaluation and Saving
    y_true = df_valid_enc["__result__"].map({"white_win":0,"draw":1,"black_win":2}).values
    loss = log_loss(y_true, final_proba)
    acc = accuracy_score(y_true, final_pred)
    macro_f1 = f1_score(y_true, final_pred, average="macro")

    print("\n=== FINAL RESULTS (Probability-Combined) ===")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Log Loss (Unbiased Proba): {loss:.4f}")
    print(f"Validation Macro-F1: {macro_f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, final_pred, target_names=CLASS_ORDER))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, final_pred))
    
    # Save Artifacts
    os.makedirs("models", exist_ok=True)
    trained_features_list = [
        "white_elo", "black_elo", "elo_diff", "elo_diff_sq", "elo_ratio",
        "eco_log_freq", "opening_log_freq", 
        "eco_prob_white_win", "eco_prob_draw", "eco_prob_black_win", 
        "opening_prob_white_win", "opening_prob_draw", "opening_prob_black_win" 
    ]
    
    joblib.dump({
        "step1": model_step1, "step2": model_step2, "scaler": scaler, 
        "freq_maps": freq_maps, "encoding_maps": encoding_maps_train,
        "features_ordered": trained_features_list, "decisiveness_c": DECISIVENESS_BIAS_C 
    }, MODEL_PATH)
    print(f"\nModels and Artifacts saved to: {MODEL_PATH}")
    
    valid_out = pd.DataFrame({
        "y_true": y_true, "y_pred": final_pred, 
        "proba_white_win": final_proba[:,0], "proba_draw": final_proba[:,1], 
        "proba_black_win": final_proba[:,2]
    })
    valid_out.to_csv(PRED_CSV, index=False)
    
    return model_step1, model_step2, valid_out

if __name__ == "__main__":
    print("Starting Training Pipeline (Final Optimized Configuration)...")
    # Execute with the parameters provided in the prompt's final attempt
    model_step1, model_step2, predictions_df = main(
        params_step1=TUNING_PARAMS_STEP1,
        params_step2=TUNING_PARAMS_STEP2
    )
    print("\nTraining finished. Artifacts saved for deployment.")
    print("\nFinal Prediction Sample (Bias Applied):")
    print(predictions_df.head(10))