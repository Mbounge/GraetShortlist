# main.py
import json
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
# No optim needed for inference
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional
from torch.utils.data import TensorDataset, DataLoader
import logging
import warnings
import traceback
import sys
import copy # Keep copy for archetype context if needed
from datetime import datetime, timedelta, timezone
import os
from google.cloud import storage # Import GCS client library

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.simplefilter('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# --- Constants & Settings ---
# --- Define GCS Bucket/Blob Names and Local Paths ---
BUCKET_NAME = "cz_players_reference_data" # <<< REPLACE WITH YOUR BUCKET NAME

ML_READY_BLOB = "cz_players_ml_ready_data.pkl"
REFERENCE_BLOB = "cz_players_reference_data.pkl"
SCALER_BLOB = "cz_scaler_pipeline.pkl"
FEATURE_INFO_BLOB = "cz_feature_info.json"
ENCODER_MODEL_BLOB = "player_encoder_model.pth"

# Define local paths where files will be saved inside the container
# Using /tmp is generally safe and writable in Cloud Run environments
LOCAL_DATA_DIR = "/tmp/player_data"
ML_READY_DF_PATH = os.path.join(LOCAL_DATA_DIR, ML_READY_BLOB)
REFERENCE_DF_PATH = os.path.join(LOCAL_DATA_DIR, REFERENCE_BLOB)
SCALER_PIPELINE_PATH = os.path.join(LOCAL_DATA_DIR, SCALER_BLOB)
FEATURE_INFO_PATH = os.path.join(LOCAL_DATA_DIR, FEATURE_INFO_BLOB)
ENCODER_MODEL_PATH = os.path.join(LOCAL_DATA_DIR, ENCODER_MODEL_BLOB)
# --- End GCS Bucket/Blob Names and Local Paths ---

# Scoring Weights
WEIGHT_SIMILARITY = 0.3
WEIGHT_PERFORMANCE = 0.3
WEIGHT_RECENT = 0.3
WEIGHT_FRESH = 0.1
ARCHETYPE_PERCENTILE = 0.85

# System Settings
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_DATE = datetime(2025, 4, 26, tzinfo=timezone.utc) # Ensure timezone aware for age calc

# Best Hyperparameters (from tuning) - needed for model definition
EMBEDDING_DIM = 32
BEST_HIDDEN1_SIZE = 512
BEST_HIDDEN2_SIZE = 64
BEST_DROPOUT1 = 0.4940863172925382
BEST_DROPOUT2 = 0.17214600618183104

# --- FastAPI & CORS Setup ---
# !!! IMPORTANT: Replace with your actual frontend URL(s) for production !!!
allowed_origins = [
    "http://localhost:3000",        # Local dev frontend
    "http://127.0.0.1:3000",       # Local dev frontend
    "https://graet.ai", # Your deployed Vercel frontend URL
    # Add any other origins (e.g., custom domains) if needed
]

app = FastAPI(title="Player Shortlist API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, # Use the list
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Global Variables for Loaded Data ---
ml_ready_df: Optional[pd.DataFrame] = None
reference_df: Optional[pd.DataFrame] = None
scaler_pipeline = None
feature_info: Dict[str, Any] = {}
feature_cols: List[str] = []
scaled_numeric_cols: List[str] = []
player_id_col: Optional[str] = None
encoder: Optional[nn.Module] = None
all_embeddings: Optional[np.ndarray] = None
# Keep device defined globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Global device set to: {device}")


# --- Model Definition (Encoder Only) ---
class PlayerEmbedder(nn.Module):
    def __init__(self, input_size, embedding_dim=EMBEDDING_DIM,
                 hidden1_size=BEST_HIDDEN1_SIZE, hidden2_size=BEST_HIDDEN2_SIZE,
                 dropout1=BEST_DROPOUT1, dropout2=BEST_DROPOUT2):
        super(PlayerEmbedder, self).__init__()
        if input_size <= 0: raise ValueError(f"Input size must be positive, got {input_size}")
        embedding_dim = max(1, int(embedding_dim)); hidden1_size = max(1, int(hidden1_size)); hidden2_size = max(1, int(hidden2_size))
        dropout1 = float(dropout1); dropout2 = float(dropout2)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden1_size), nn.ReLU(), nn.BatchNorm1d(hidden1_size), nn.Dropout(dropout1),
            nn.Linear(hidden1_size, hidden2_size), nn.ReLU(), nn.BatchNorm1d(hidden2_size), nn.Dropout(dropout2),
            nn.Linear(hidden2_size, embedding_dim)
        )
    def forward(self, x): return self.net(x)

# --- GCS Download Function ---
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    logger.info(f"Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_name}...")
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        # The client automatically authenticates using the Cloud Run service account
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        # Check if blob exists before downloading
        if not blob.exists():
             logger.error(f"Blob gs://{bucket_name}/{source_blob_name} does not exist!")
             raise FileNotFoundError(f"Blob gs://{bucket_name}/{source_blob_name} not found in GCS.")

        blob.download_to_filename(destination_file_name)
        logger.info(f"Successfully downloaded {source_blob_name}.")
    except FileNotFoundError as fnf_error:
         raise fnf_error # Re-raise specific error
    except Exception as e:
        logger.error(f"Failed to download gs://{bucket_name}/{source_blob_name}: {e}")
        raise # Re-raise other exceptions


# --- Load Data & Model on Startup ---
@app.on_event("startup")
def load_model_and_data():
    global ml_ready_df, reference_df, scaler_pipeline, feature_info, feature_cols
    global scaled_numeric_cols, player_id_col, encoder, all_embeddings, device

    # --- Create data directory ---
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    # --- Download files using GCS Client ---
    try:
        logger.info("--- Starting File Downloads from GCS ---")
        download_blob(BUCKET_NAME, ML_READY_BLOB, ML_READY_DF_PATH)
        download_blob(BUCKET_NAME, REFERENCE_BLOB, REFERENCE_DF_PATH)
        download_blob(BUCKET_NAME, SCALER_BLOB, SCALER_PIPELINE_PATH)
        download_blob(BUCKET_NAME, FEATURE_INFO_BLOB, FEATURE_INFO_PATH)
        download_blob(BUCKET_NAME, ENCODER_MODEL_BLOB, ENCODER_MODEL_PATH)
        logger.info("--- File Downloads Completed ---")
    except Exception as download_err:
         logger.error(f"FATAL ERROR during file download: {download_err}", exc_info=True)
         # This will cause the Cloud Run instance startup to fail, which is intended
         raise RuntimeError(f"Failed to download necessary files: {download_err}") from download_err

    # --- Load data and model (using LOCAL paths) ---
    try:
        logger.info("--- Loading Preprocessed Data and Metadata from Downloaded Files ---")
        ml_ready_df = pd.read_pickle(ML_READY_DF_PATH)
        reference_df = pd.read_pickle(REFERENCE_DF_PATH)
        with open(SCALER_PIPELINE_PATH, 'rb') as f: scaler_pipeline = pickle.load(f)
        with open(FEATURE_INFO_PATH, 'r') as f: feature_info = json.load(f)
        feature_cols = feature_info['feature_columns']
        scaled_numeric_cols = feature_info['scaled_numeric_columns']
        player_id_col = feature_info['player_id_column']
        logger.info("Data and metadata loaded from local files.")

        # --- Data Validation ---
        if ml_ready_df.empty or reference_df.empty or not feature_cols or not player_id_col:
            raise ValueError("Loaded data incomplete after download.")
        if player_id_col not in ml_ready_df.columns or player_id_col not in reference_df.columns:
            raise ValueError(f"Player ID column '{player_id_col}' missing after download.")
        missing_features = [col for col in feature_cols if col not in ml_ready_df.columns]
        if missing_features:
            raise ValueError(f"Missing features in ML-Ready DF after download: {missing_features}")

        if ml_ready_df[player_id_col].nunique() == len(ml_ready_df):
            ml_ready_df = ml_ready_df.set_index(player_id_col)
        else:
            raise ValueError("Player ID not unique in ML-Ready DF after download.")
        if reference_df[player_id_col].nunique() == len(reference_df):
            reference_df = reference_df.set_index(player_id_col)
        else:
            raise ValueError("Player ID not unique in Reference DF after download.")
        logger.info("Data validated and index set.")

        # --- Load Encoder Model ---
        input_size = len(feature_cols)
        if input_size <= 0:
             raise ValueError("Feature columns list is empty, cannot determine model input size.")
        encoder = PlayerEmbedder(input_size).to(device)
        logger.info(f"Loading trained encoder model state from local path: {ENCODER_MODEL_PATH}")
        encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
        encoder.eval() # Set to evaluation mode
        logger.info("Encoder loaded and set to eval mode.")

        # --- Generate Embeddings ---
        logger.info("Generating player embeddings from loaded data...")
        # Ensure data exists before accessing .loc
        if ml_ready_df is None or ml_ready_df.empty:
             raise ValueError("ml_ready_df is not loaded or empty before embedding generation.")
        # Use .loc with the full feature list, handle potential NaNs generated during preprocessing
        X = ml_ready_df.loc[:, feature_cols].fillna(0).values.astype(np.float32)
        if np.isinf(X).any():
             logger.warning("Infinity values found in feature matrix X before embedding. Replacing with 0.")
             X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) # Replace inf as well

        ds = TensorDataset(torch.from_numpy(X))
        # Adjust batch size based on available memory if needed, 512 is often reasonable
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        embeddings_list = []
        with torch.no_grad():
            for (batch,) in loader:
                if batch.shape[0] == 0: continue
                # Handle potential batchnorm issue with batch size 1 during inference
                if batch.shape[0] <= 1 and any(isinstance(m, nn.BatchNorm1d) for m in encoder.modules()):
                     logger.warning(f"Encountered batch size <= 1 during embedding generation with BatchNorm active. Skipping batch or handle differently if needed.")
                     # Option: could pad the batch temporarily, or skip
                     continue # Skipping for simplicity
                embeddings_list.append(encoder(batch.to(device)).cpu().numpy())

        if not embeddings_list:
             raise RuntimeError("No embeddings were generated, possibly due to empty batches or BatchNorm issues.")

        all_embeddings = np.vstack(embeddings_list)
        if all_embeddings.shape[0] != len(ml_ready_df):
            raise ValueError(f"Embeddings shape mismatch after generation. Expected {len(ml_ready_df)}, Got {all_embeddings.shape[0]}.")
        logger.info(f"Embeddings generated with shape: {all_embeddings.shape}")
        logger.info("--- Startup data and model loading complete ---")

    except Exception as e:
        logger.error(f"FATAL ERROR during startup after download: {e}", exc_info=True)
        # Re-raise to ensure Cloud Run knows the startup failed
        raise RuntimeError(f"Failed to initialize model and data after download: {e}") from e


# --- Archetype Creation Function (Copied from Inference Script) ---
def create_archetype_input_vector(position_group_target, age_group, percentile,
                                  feature_cols, scaled_numeric_cols, scaler_pipeline,
                                  reference_data_df):
    # Check if required data is loaded
    if not feature_cols or reference_data_df is None or reference_data_df.empty:
        logger.error("Missing data for archetype creation (features/reference_df).")
        return None

    archetype_input = pd.Series(0.0, index=feature_cols)
    original_target_values = {}
    scaled_target_values = {}
    age_min, age_max = age_group
    required_context_cols = ['position_group', 'age_orig', 'gender']
    target_gender = 'MEN' # Assuming MEN context for now
    gender_col_name = 'gender'

    # --- Context Filtering ---
    context_df = pd.DataFrame() # Initialize
    if all(col in reference_data_df.columns for col in required_context_cols):
        # Create boolean masks safely
        pos_mask = reference_data_df['position_group'] == position_group_target
        age_mask = (reference_data_df['age_orig'] >= age_min) & (reference_data_df['age_orig'] <= age_max)
        gender_mask = reference_data_df[gender_col_name].fillna('Unknown').astype(str) == target_gender
        # Combine masks
        context_df = reference_data_df[pos_mask & age_mask & gender_mask].copy()

        if context_df.empty:
            logger.warning(f"No players found for context: PosGroup={position_group_target}, Age={age_group}, Gender={target_gender}. Using global context (filtered by gender).")
            # Fallback: Use global reference data, filtered only by gender if possible
            context_df = reference_data_df.copy()
            if gender_col_name in context_df.columns:
                context_df = context_df[context_df[gender_col_name].fillna('Unknown').astype(str) == target_gender].copy()
                logger.info("Applied gender filter to global context fallback.")
            else:
                logger.warning("Gender column missing, cannot apply gender filter to global context fallback.")
        else:
            logger.info(f"Found {len(context_df)} players for archetype context: PosGroup={position_group_target}, Age={age_group}, Gender={target_gender}.")
    else:
        missing_cols_str = ", ".join([col for col in required_context_cols if col not in reference_data_df.columns])
        logger.error(f"Missing required context columns: [{missing_cols_str}]. Using global context (filtered by gender if possible).")
        # Fallback: Use global reference data, filtered only by gender if possible
        context_df = reference_data_df.copy()
        if gender_col_name in context_df.columns:
            context_df = context_df[context_df[gender_col_name].fillna('Unknown').astype(str) == target_gender].copy()
            logger.info("Applied gender filter to global context fallback.")
        else:
            logger.warning("Gender column missing, cannot apply gender filter to global context fallback.")

    # Handle case where even fallback context is empty
    if context_df.empty:
        logger.error("Context DataFrame is empty even after fallback. Cannot create archetype.")
        # Return a default vector (e.g., all 0.5s) or handle as needed
        archetype_input[:] = 0.5 # Example default
        # Set categorical defaults (ensure these columns exist in feature_cols)
        pos_col_name = f'pos_{position_group_target}'
        if pos_col_name in archetype_input.index: archetype_input[pos_col_name] = 1.0
        if 'nat_CZ' in archetype_input.index: archetype_input['nat_CZ'] = 1.0
        gender_ohe_col = f'gender_{target_gender}'
        if gender_ohe_col in archetype_input.index: archetype_input[gender_ohe_col] = 1.0
        logger.warning("Returning default archetype vector (0.5 numeric, default categorical).")
        return archetype_input.values.astype(np.float32)


    # --- Calculate Original Target Values ---
    original_col_map = {}
    for scaled_col in scaled_numeric_cols:
        potential_orig = f"{scaled_col}_orig"
        if potential_orig in reference_data_df.columns:
            original_col_map[scaled_col] = potential_orig
        elif scaled_col in reference_data_df.columns: # Fallback to scaled name if _orig missing
            original_col_map[scaled_col] = scaled_col
            logger.debug(f"Using non-orig column '{scaled_col}' as source for archetype calculation.")
        else:
            logger.debug(f"Cannot find original source for scaled column '{scaled_col}'.")

    # Define which columns use percentile vs median
    percentile_cols_scaled = [
        col for col in scaled_numeric_cols if
        '_trend_' in col or 'recent_P' in col or 'recent_save' in col or
        'game_freshness' in col or 'season_pointsPerGame' in col or
        'season_svp' in col or 'adj_season' in col or 'adj_user_career' in col # Include adjusted rates
    ]
    inverse_percentile_cols = [col for col in percentile_cols_scaled if 'GAA' in col or 'gaa' in col] # Include adjusted GAA

    numeric_cols_to_process = [col for col in scaled_numeric_cols if col in feature_cols]

    for scaled_col in numeric_cols_to_process:
        target_value_orig = 0.0
        original_context_col = original_col_map.get(scaled_col)
        source_df_for_stats = context_df # Start with filtered context

        # Check if context has the column and valid data
        if not original_context_col or original_context_col not in context_df.columns or not context_df[original_context_col].notna().any():
             original_context_col = original_col_map.get(scaled_col) # Get name again for global check
             source_df_for_stats = reference_data_df # Switch to global reference
             logger.debug(f"Falling back to global reference for archetype column '{scaled_col}' (orig: {original_context_col})")

        # Check if the chosen source (context or global) has the column and valid data
        if original_context_col and original_context_col in source_df_for_stats.columns and source_df_for_stats[original_context_col].notna().any():
            context_series = source_df_for_stats[original_context_col].dropna()
            if not context_series.empty:
                try:
                    # Calculate target value based on percentile or median
                    if scaled_col in percentile_cols_scaled:
                        target_percentile_val = percentile
                        if scaled_col in inverse_percentile_cols:
                            target_percentile_val = 1.0 - percentile
                        target_value_orig = context_series.quantile(target_percentile_val)
                    elif scaled_col == 'age':
                        # Use midpoint of the requested age group, not the context ages
                        target_value_orig = (age_min + age_max) / 2.0
                    else:
                        # Use median for other stats (like raw GP, PIM etc.)
                        target_value_orig = context_series.median()
                except Exception as e_stat:
                    logger.warning(f"Error calculating quantile/median for {original_context_col}: {e_stat}. Using 0 as fallback.")
                    target_value_orig = 0.0 # Fallback on calculation error
            else:
                # Series was empty after dropna
                target_value_orig = 0.0
                logger.warning(f"Context series for {original_context_col} was empty after dropna. Using 0.")
        else:
            # Column not found or all NaN in the chosen source
            logger.warning(f"Cannot find valid data for archetype column '{scaled_col}' (orig: {original_context_col}). Using 0.")
            target_value_orig = 0.0

        # Final check for NaN result from quantile/median
        if pd.isna(target_value_orig):
            logger.warning(f"Target original value for '{scaled_col}' is NaN after quantile/median. Using 0.")
            target_value_orig = 0.0

        original_target_values[scaled_col] = target_value_orig

    # --- Scale Target Values ---
    if scaler_pipeline and numeric_cols_to_process:
        # Create DataFrame with only the columns to be scaled
        target_df_for_scaling = pd.DataFrame([original_target_values])[numeric_cols_to_process]

        # Replace inf/-inf and fill NaNs (should ideally not happen after previous step)
        target_df_for_scaling = target_df_for_scaling.replace([np.inf, -np.inf], np.nan)
        if target_df_for_scaling.isnull().any().any():
             logger.warning("NaNs found in archetype vector before scaling. Filling with 0.")
             target_df_for_scaling = target_df_for_scaling.fillna(0.0)

        try:
            # Get expected feature names from the pipeline
            pipeline_features = []
            if hasattr(scaler_pipeline, 'feature_names_in_'):
                pipeline_features = scaler_pipeline.feature_names_in_
            elif hasattr(scaler_pipeline, 'steps') and hasattr(scaler_pipeline.steps[0][1], 'feature_names_in_'):
                 pipeline_features = scaler_pipeline.steps[0][1].feature_names_in_
            else:
                 logger.warning("Cannot determine feature names from scaler pipeline. Assuming order matches numeric_cols_to_process.")
                 pipeline_features = numeric_cols_to_process # Fallback

            # Ensure DataFrame columns match pipeline's expected features
            cols_in_vector = target_df_for_scaling.columns
            missing_cols = set(pipeline_features) - set(cols_in_vector)
            extra_cols = set(cols_in_vector) - set(pipeline_features)

            if missing_cols:
                logger.warning(f"Archetype vector missing columns expected by scaler: {missing_cols}. Filling with 0.")
                for col in missing_cols: target_df_for_scaling[col] = 0.0
            if extra_cols:
                 logger.warning(f"Archetype vector has extra columns not seen by scaler: {extra_cols}. Dropping them.")
                 target_df_for_scaling = target_df_for_scaling.drop(columns=list(extra_cols), errors='ignore')

            # Reorder to match pipeline
            target_df_for_scaling = target_df_for_scaling[pipeline_features]

            # Transform using the loaded pipeline
            scaled_targets_array = scaler_pipeline.transform(target_df_for_scaling)
            scaled_target_values = dict(zip(pipeline_features, scaled_targets_array[0]))

        except Exception as e:
            logger.error(f"Error scaling archetype vector for PosGroup={position_group_target}, Age={age_group}: {e}. Using 0.5 default for numeric features.", exc_info=True)
            # Use pipeline_features if available for default keys
            keys_for_default = pipeline_features if 'pipeline_features' in locals() and pipeline_features else numeric_cols_to_process
            scaled_target_values = {col: 0.5 for col in keys_for_default}
    else:
        logger.warning("Scaler pipeline not available or no numeric columns to process. Using 0.5 default for archetype numeric features.")
        scaled_target_values = {col: 0.5 for col in numeric_cols_to_process}

    # --- Assemble Final Archetype Vector ---
    # Apply scaled numeric values
    for col, scaled_value in scaled_target_values.items():
        if col in archetype_input.index:
            archetype_input[col] = scaled_value

    # Set OHE categorical features (ensure columns exist in `feature_cols`)
    unique_positions = ['G', 'D', 'F', 'Unknown'] # Expected positions
    unique_nationalities = feature_info.get('unique_nationalities', ['CZ']) # Get from loaded info or default
    unique_genders = ['MEN', 'WOMEN', 'UNKNOWN'] # Expected genders

    # Position
    for p in unique_positions:
        p_col = f'pos_{p}'
        if p_col in archetype_input.index:
            archetype_input[p_col] = 1.0 if p == position_group_target else 0.0
    # Verify if the target position column actually exists
    target_pos_col = f'pos_{position_group_target}'
    if target_pos_col not in archetype_input.index:
         logger.warning(f"Target position column '{target_pos_col}' not found in features. Setting pos_Unknown.")
         if 'pos_Unknown' in archetype_input.index: archetype_input['pos_Unknown'] = 1.0


    # Nationality (Assume CZ for Czech context)
    for n in unique_nationalities:
        n_col = f'nat_{n}'
        if n_col in archetype_input.index:
            archetype_input[n_col] = 1.0 if n == 'CZ' else 0.0
    # Fallback if nat_CZ doesn't exist
    if 'nat_CZ' not in archetype_input.index and unique_nationalities:
         fallback_nat_col = f'nat_{unique_nationalities[0]}'
         if fallback_nat_col in archetype_input.index:
             archetype_input[fallback_nat_col] = 1.0
             logger.warning(f"nat_CZ not found, setting {fallback_nat_col}=1.0 as default.")

    # Gender
    for g in unique_genders:
        g_col = f'gender_{g}'
        if g_col in archetype_input.index:
            archetype_input[g_col] = 1.0 if g == target_gender else 0.0
    # Fallback if target gender column missing
    target_gender_col = f'gender_{target_gender}'
    if target_gender_col not in archetype_input.index:
         logger.warning(f"Target gender column '{target_gender_col}' not found. Setting gender_UNKNOWN.")
         if 'gender_UNKNOWN' in archetype_input.index: archetype_input['gender_UNKNOWN'] = 1.0

    # Final check for NaNs
    if archetype_input.isnull().any():
        logger.warning("NaNs found in final archetype vector. Filling with 0.")
        archetype_input = archetype_input.fillna(0.0)

    return archetype_input.values.astype(np.float32)


# --- Shortlist Generation Function (Copied from Inference Script) ---
def generate_birth_year_shortlist(birth_year: int,
                                  ml_ready_player_df: pd.DataFrame,
                                  all_player_embeddings: np.ndarray,
                                  reference_df: pd.DataFrame,
                                  scaler_pipeline,
                                  feature_cols,
                                  scaled_numeric_cols,
                                  player_id_col,
                                  top_n=50,
                                  archetype_percentile=ARCHETYPE_PERCENTILE,
                                  w_similarity=WEIGHT_SIMILARITY,
                                  w_perf=WEIGHT_PERFORMANCE,
                                  w_recent=WEIGHT_RECENT,
                                  w_fresh=WEIGHT_FRESH) -> dict:
    logger.info(f"\n--- Generating Shortlist: Birth Year {birth_year} (Filtering by Age & Gender='MEN') (Top {top_n}) ---")

    # --- Initial checks ---
    if all_player_embeddings is None or reference_df is None or ml_ready_player_df is None or encoder is None or scaler_pipeline is None:
        logger.error("Required global components not set for shortlist generation.")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    if len(ml_ready_player_df) != all_player_embeddings.shape[0]:
        logger.error(f"Mismatch between ml_ready_df length ({len(ml_ready_player_df)}) and embeddings length ({all_player_embeddings.shape[0]}).")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    if ml_ready_player_df.index.name != player_id_col:
         logger.error("ML Ready DF index is not player_id. Cannot map embeddings correctly.")
         return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}

    # --- Filtering by Age and Gender ---
    age_col_name = 'age_orig'; gender_col_name = 'gender'; target_gender = 'MEN'
    required_cols = [age_col_name, gender_col_name, 'position_group']
    if not all(col in reference_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in reference_df.columns]
        logger.error(f"Required columns missing from Reference DF for filtering: {missing}.")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}

    current_year = CURRENT_DATE.year
    target_age = current_year - birth_year
    logger.info(f"Filtering for players with {age_col_name} == {target_age} AND {gender_col_name} == '{target_gender}'")

    # Use .copy() to avoid SettingWithCopyWarning if modifying ref_local later
    ref_local = reference_df.copy()
    # Convert age to numeric, coercing errors to NaN
    ref_local[age_col_name] = pd.to_numeric(ref_local[age_col_name], errors='coerce')
    valid_age_mask = ref_local[age_col_name].notna()
    ref_local[gender_col_name] = ref_local[gender_col_name].fillna('Unknown').astype(str)

    # Apply filters
    year_filtered_ref_df = ref_local[
        valid_age_mask &
        (ref_local[age_col_name] == target_age) &
        (ref_local[gender_col_name] == target_gender)
    ]

    if year_filtered_ref_df.empty:
        logger.warning(f"No players found in Reference DF for age {target_age}, gender '{target_gender}' (Birth Year: {birth_year}).")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    logger.info(f"Found {len(year_filtered_ref_df)} players in Reference DF for age {target_age}, gender '{target_gender}'.")

    # --- Get Embeddings for Filtered Players ---
    target_player_ids = year_filtered_ref_df.index.tolist()

    # Create mapping from player_id (index of ml_ready_df) to embedding row index
    ml_df_index_to_embedding_idx = {idx: i for i, idx in enumerate(ml_ready_player_df.index)}

    # Get embedding indices for the filtered players
    year_embedding_indices = [ml_df_index_to_embedding_idx.get(pid) for pid in target_player_ids]

    # Filter out players whose IDs weren't found in the mapping (shouldn't happen if data is consistent)
    # Also filter the corresponding embedding indices
    valid_map = [(pid, idx) for pid, idx in zip(target_player_ids, year_embedding_indices) if idx is not None]

    if len(valid_map) != len(target_player_ids):
        logger.warning(f"Could not map all {len(target_player_ids)} filtered players to embeddings. Found {len(valid_map)}.")

    if not valid_map:
        logger.error("Could not map any filtered players to embeddings.")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}

    filtered_pids = [pid for pid, idx in valid_map]
    filtered_embedding_indices = [idx for pid, idx in valid_map]

    # Select the relevant embeddings and filter the reference dataframe
    year_embeddings = all_player_embeddings[filtered_embedding_indices]
    year_filtered_ref_df = year_filtered_ref_df.loc[filtered_pids] # Re-filter ref_df to match embeddings

    # --- Generate Archetype Embeddings ---
    age_min_context = target_age - 1
    age_max_context = target_age + 1
    age_group_context = (age_min_context, age_max_context)
    archetype_embeddings = {}
    if encoder is None: # Add check for encoder
         logger.error("Encoder model is not loaded. Cannot generate archetype embeddings.")
         return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}

    encoder.eval() # Ensure encoder is in eval mode
    with torch.no_grad():
        for pos_group in ['G', 'D', 'F']:
            logger.info(f"Generating archetype embedding for PosGroup={pos_group}, Context Age={age_group_context}, Pct={archetype_percentile}")
            # Pass the global reference_df here
            archetype_input_vec = create_archetype_input_vector(
                pos_group, age_group_context, archetype_percentile,
                feature_cols, scaled_numeric_cols, scaler_pipeline,
                reference_df # Use the main loaded reference_df for context
            )
            if archetype_input_vec is None:
                logger.error(f"Failed to create archetype vector for position group {pos_group}.")
                archetype_embeddings[pos_group] = None # Mark as None
                continue

            archetype_input_tensor = torch.tensor(archetype_input_vec, dtype=torch.float32).unsqueeze(0).to(device)
            # Handle batchnorm with batch size 1
            if archetype_input_tensor.shape[0] <= 1 and hasattr(encoder, 'modules') and any(isinstance(m, nn.BatchNorm1d) for m in encoder.modules()):
                 logger.warning(f"Generating archetype embedding for {pos_group} with batch size 1 and BatchNorm active.")
                 # Potentially add temporary padding or adjust model if this causes issues

            archetype_embeddings[pos_group] = encoder(archetype_input_tensor).cpu().numpy()

    # --- Calculate Scores ---
    final_scores = []
    similarity_scores = []
    perf_scores = []
    recent_perf_scores = []
    fresh_scores = []
    player_ids_for_scores = [] # Keep track of IDs corresponding to scores

    trend_weight = 0.7 # Weight for trend within performance score
    recent_weight = 0.3 # Weight for recent rate within performance score

    # Iterate through the *filtered* player IDs and their corresponding embeddings
    for i, player_id in enumerate(year_filtered_ref_df.index):
        player_ids_for_scores.append(player_id)
        try:
            # Get data for the current player
            player_row_ref = year_filtered_ref_df.loc[player_id]
            # Access ml_ready_df using the player_id index
            player_row_ml = ml_ready_df.loc[player_id]
            player_pos_group = player_row_ref.get('position_group', 'Unknown')
            if pd.isna(player_pos_group): player_pos_group = 'Unknown'
        except KeyError:
            logger.warning(f"Could not find data for player {player_id} in reference or ML dataframes. Skipping score calculation.")
            # Append default scores to maintain alignment
            similarity_scores.append(0.0); perf_scores.append(0.0); recent_perf_scores.append(0.0); fresh_scores.append(0.0); final_scores.append(0.0)
            continue

        # --- Similarity Score ---
        player_embedding = year_embeddings[i:i+1] # Get the embedding for this player
        similarity_score = 0.0
        archetype_embed = archetype_embeddings.get(player_pos_group) # Get archetype for player's position

        if archetype_embed is not None:
            norm_player = np.linalg.norm(player_embedding)
            norm_archetype = np.linalg.norm(archetype_embed)
            if norm_player > 1e-9 and norm_archetype > 1e-9: # Avoid division by zero
                # Calculate cosine similarity
                cos_sim = np.dot(player_embedding, archetype_embed.T) / (norm_player * norm_archetype)
                # Normalize to [0, 1] range
                similarity_score = (cos_sim.item() + 1.0) / 2.0
            # else: similarity remains 0.0
        else:
            logger.debug(f"No valid archetype embedding found for position group '{player_pos_group}' of player {player_id}. Similarity set to 0.")

        # --- Performance & Freshness Scores (from scaled ML data) ---
        game_fresh_score = player_row_ml.get('game_freshness', 0.0)

        # Initialize performance components
        perf_score_combined = 0.0
        recent_perf_rate_scaled = 0.0 # Use scaled recent rate for scoring
        player_trends = []

        # Extract relevant trend and recent stats based on position
        if player_pos_group == 'G':
            # Use adjusted trends (already scaled)
            gaa_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_GAA_trend' in col]
            svp_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_SVP_trend' in col]
            # Lower GAA trend is better (invert score contribution if needed, or use raw difference)
            # For simplicity, let's average: higher SVP trend is good, lower GAA trend is good (1-trend)
            if gaa_trends: player_trends.extend([1.0 - t for t in gaa_trends]) # Assuming trends are differences; lower is better -> higher score
            if svp_trends: player_trends.extend(svp_trends) # Higher is better

            # Use scaled adjusted recent/season stats
            recent_perf_rate_scaled = player_row_ml.get('recent_adj_save_pct', 0.0)
            # Fallback to scaled adjusted season SVP if recent is missing/zero
            if pd.isna(recent_perf_rate_scaled) or recent_perf_rate_scaled == 0.0:
                 recent_perf_rate_scaled = player_row_ml.get('adj_season_svp', 0.0)

        elif player_pos_group in ['D', 'F']:
            # Use adjusted trends (already scaled)
            p_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_P_per_GP_trend' in col]
            g_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_G_per_GP_trend' in col]
            if p_trends: player_trends.extend(p_trends)
            if g_trends: player_trends.extend(g_trends)

            # Use scaled adjusted recent/season stats
            recent_perf_rate_scaled = player_row_ml.get('recent_adj_P_per_GP', 0.0)
            # Fallback to scaled adjusted season PPG if recent is missing/zero
            if pd.isna(recent_perf_rate_scaled) or recent_perf_rate_scaled == 0.0:
                 recent_perf_rate_scaled = player_row_ml.get('adj_season_pointsPerGame', 0.0)

        # Calculate average trend score (already scaled)
        avg_trend_score_scaled = np.mean(player_trends) if player_trends else 0.0

        # Ensure recent perf rate is a float for calculation
        recent_perf_rate_scaled = float(recent_perf_rate_scaled) if pd.notna(recent_perf_rate_scaled) else 0.0

        # Combine trend and recent (already scaled) for the performance component
        perf_score_combined = (trend_weight * avg_trend_score_scaled) + (recent_weight * recent_perf_rate_scaled)

        # Clip individual scores to [0, 1] before final weighting
        similarity_score = np.clip(similarity_score, 0.0, 1.0)
        perf_score_combined = np.clip(perf_score_combined, 0.0, 1.0) # Clip the combined perf score
        # Clip the recent component separately for reporting/weighting
        recent_perf_score_clipped = np.clip(recent_perf_rate_scaled, 0.0, 1.0)
        game_fresh_score = np.clip(game_fresh_score, 0.0, 1.0)

        # Calculate final score using clipped components and weights
        final_score = (w_similarity * similarity_score) + \
                      (w_perf * perf_score_combined) + \
                      (w_recent * recent_perf_score_clipped) + \
                      (w_fresh * game_fresh_score)

        # Append scores for the DataFrame
        similarity_scores.append(similarity_score)
        perf_scores.append(perf_score_combined) # Append the combined performance score
        recent_perf_scores.append(recent_perf_score_clipped) # Append the clipped recent component
        fresh_scores.append(game_fresh_score)
        final_scores.append(final_score)

    # --- Combine Scores and Rank ---
    scores_df = pd.DataFrame({
        player_id_col: player_ids_for_scores, # Use the tracked IDs
        'final_score': final_scores,
        'archetype_similarity': similarity_scores,
        'perf_score_scaled': perf_scores, # Combined performance score
        'recent_perf_score_scaled': recent_perf_scores, # Scaled recent rate component
        'game_freshness_scaled': fresh_scores
    }).set_index(player_id_col) # Set index for joining

    # Merge scores back into the filtered reference dataframe
    # Ensure year_filtered_ref_df index is player_id
    if year_filtered_ref_df.index.name != player_id_col:
        year_filtered_ref_df = year_filtered_ref_df.set_index(player_id_col)

    # Join scores - use left join to keep all players from the filtered ref df
    year_filtered_ref_df = year_filtered_ref_df.join(scores_df, how='left')

    # Fill NaN scores with 0.0 (e.g., if a player was skipped during scoring)
    score_cols_list = ['final_score', 'archetype_similarity', 'perf_score_scaled', 'recent_perf_score_scaled', 'game_freshness_scaled']
    year_filtered_ref_df[score_cols_list] = year_filtered_ref_df[score_cols_list].fillna(0.0)

    # --- Create Shortlists per Position ---
    shortlists = {}
    for pos_group in ['G', 'D', 'F']:
        pos_mask = year_filtered_ref_df['position_group'] == pos_group
        pos_df = year_filtered_ref_df[pos_mask].copy()
        num_found_before_head = len(pos_df)
        logger.info(f"Found {num_found_before_head} players for pos group {pos_group}, birth year {birth_year} (age {target_age}, gender '{target_gender}') before taking top {top_n}.")

        if not pos_df.empty and 'final_score' in pos_df.columns:
            # Sort by final score and take top N
            ranked_pos_df = pos_df.sort_values('final_score', ascending=False).head(top_n)
            # Add original recent stats for display (from reference_df)
            recent_orig_cols = [
                'recent_GP_orig', 'recent_G_orig', 'recent_A_orig', 'recent_TP_orig',
                'recent_PIM_orig', 'recent_plus_minus_orig', 'recent_saves_orig',
                'recent_shots_against_orig', 'recent_adj_P_per_GP_orig', 'recent_adj_save_pct_orig'
            ]
            for col in recent_orig_cols:
                 if col in reference_df.columns: # Check if col exists in the main reference df
                      # Map the original values based on index
                      ranked_pos_df[col.replace('_orig','')] = ranked_pos_df.index.map(reference_df[col])
                 else:
                      ranked_pos_df[col.replace('_orig','')] = np.nan # Add column with NaN if missing

            shortlists[pos_group] = ranked_pos_df
        else:
            logger.warning(f"No players or scores found for pos group {pos_group}, birth year {birth_year} (age {target_age}, gender '{target_gender}').")
            shortlists[pos_group] = pd.DataFrame() # Return empty DF

    return shortlists


# --- FastAPI Request/Response Models ---
class ShortlistRequest(BaseModel):
    birth_year: int
    position: Literal['G','D','F']
    top_n: int = Field(default=10, ge=1, le=100) # Add validation

class PlayerOut(BaseModel):
    # Core Identifiers
    player_id: str
    name: Optional[str] = None
    age_orig: Optional[int] = None
    position_orig: Optional[str] = None
    nationality_orig: Optional[str] = None
    position_group: Optional[str] = None
    gender: Optional[str] = None

    # Scores
    final_score: Optional[float] = None
    archetype_similarity: Optional[float] = None
    perf_score_scaled: Optional[float] = None # Combined trend/recent perf
    recent_perf_score_scaled: Optional[float] = None # Recent perf component only
    game_freshness_scaled: Optional[float] = None

    # Original Season Stats
    season_gamesPlayed_orig: Optional[int] = None
    season_goals_orig: Optional[int] = None
    season_assists_orig: Optional[int] = None
    season_points_orig: Optional[int] = None
    season_pointsPerGame_orig: Optional[float] = None
    season_gaa_orig: Optional[float] = None
    season_svp_orig: Optional[float] = None
    season_shutouts_orig: Optional[int] = None

    # Original Recent Stats (Mapped back for display)
    recent_GP: Optional[int] = None
    recent_G: Optional[int] = None
    recent_A: Optional[int] = None
    recent_TP: Optional[int] = None
    recent_PIM: Optional[int] = None
    recent_plus_minus: Optional[int] = None
    recent_saves: Optional[int] = None
    recent_shots_against: Optional[int] = None
    # Adjusted recent stats (original values)
    recent_adj_P_per_GP: Optional[float] = None # Mapped from recent_adj_P_per_GP_orig
    recent_adj_save_pct: Optional[float] = None # Mapped from recent_adj_save_pct_orig

    # Freshness Original Value
    days_since_last_game: Optional[int] = None


# --- Endpoint ---
@app.post("/shortlist/", response_model=List[PlayerOut])
def get_shortlist(req: ShortlistRequest):
    logger.info(f"Received shortlist request: year={req.birth_year}, position={req.position}, top_n={req.top_n}")
    try:
        # Check if global data is loaded
        if reference_df is None or ml_ready_df is None or all_embeddings is None or encoder is None or scaler_pipeline is None or not feature_cols or not player_id_col:
             logger.error("Service not ready: Data or model components not loaded.")
             raise HTTPException(status_code=503, detail="Service not ready, data or model not loaded.")

        # Generate the shortlist for the requested year
        shortlists_for_year = generate_birth_year_shortlist(
            birth_year=req.birth_year,
            ml_ready_player_df=ml_ready_df, # Pass the indexed DF
            all_player_embeddings=all_embeddings,
            reference_df=reference_df, # Pass the indexed DF
            scaler_pipeline=scaler_pipeline,
            feature_cols=feature_cols,
            scaled_numeric_cols=scaled_numeric_cols,
            player_id_col=player_id_col, # Pass the column name
            top_n=req.top_n,
            archetype_percentile=ARCHETYPE_PERCENTILE,
            w_similarity=WEIGHT_SIMILARITY,
            w_perf=WEIGHT_PERFORMANCE,
            w_recent=WEIGHT_RECENT,
            w_fresh=WEIGHT_FRESH
        )

        # Get the specific position dataframe
        df = shortlists_for_year.get(req.position)

        # Handle empty results
        if df is None or df.empty:
            logger.info(f"No players found for position {req.position}, year {req.birth_year}.")
            return [] # Return empty list

        # Convert DataFrame to list of dictionaries
        # Reset index if player_id is the index, otherwise use the column
        if df.index.name == player_id_col:
            recs = df.reset_index().to_dict(orient='records')
        else:
            # This case shouldn't happen if startup logic is correct, but handle defensively
            if player_id_col not in df.columns:
                 logger.error(f"Player ID column '{player_id_col}' missing from shortlist DataFrame.")
                 return []
            recs = df.to_dict(orient='records')

        # --- Format Output using Pydantic Model ---
        output_players = []
        for r in recs:
            try:
                # Prepare data dictionary, handling potential missing keys and NaN/None values
                player_data = {
                    # Core
                    'player_id': str(r.get(player_id_col, 'N/A')), # Get player ID safely
                    'name': r.get('name'),
                    'age_orig': int(r['age_orig']) if pd.notna(r.get('age_orig')) else None,
                    'position_orig': r.get('position_orig'),
                    'nationality_orig': r.get('nationality_orig'),
                    'position_group': r.get('position_group'),
                    'gender': r.get('gender'),
                    # Scores
                    'final_score': float(r['final_score']) if pd.notna(r.get('final_score')) else None,
                    'archetype_similarity': float(r['archetype_similarity']) if pd.notna(r.get('archetype_similarity')) else None,
                    'perf_score_scaled': float(r['perf_score_scaled']) if pd.notna(r.get('perf_score_scaled')) else None,
                    'recent_perf_score_scaled': float(r['recent_perf_score_scaled']) if pd.notna(r.get('recent_perf_score_scaled')) else None,
                    'game_freshness_scaled': float(r['game_freshness_scaled']) if pd.notna(r.get('game_freshness_scaled')) else None,
                    # Season Orig
                    'season_gamesPlayed_orig': int(r['season_gamesPlayed_orig']) if pd.notna(r.get('season_gamesPlayed_orig')) else None,
                    'season_goals_orig': int(r['season_goals_orig']) if pd.notna(r.get('season_goals_orig')) else None,
                    'season_assists_orig': int(r['season_assists_orig']) if pd.notna(r.get('season_assists_orig')) else None,
                    'season_points_orig': int(r['season_points_orig']) if pd.notna(r.get('season_points_orig')) else None,
                    'season_pointsPerGame_orig': float(r['season_pointsPerGame_orig']) if pd.notna(r.get('season_pointsPerGame_orig')) else None,
                    'season_gaa_orig': float(r['season_gaa_orig']) if pd.notna(r.get('season_gaa_orig')) else None,
                    'season_svp_orig': float(r['season_svp_orig']) if pd.notna(r.get('season_svp_orig')) else None,
                    # Use correct key for shutouts if it exists
                    'season_shutouts_orig': int(r['season_shutouts_orig']) if pd.notna(r.get('season_shutouts_orig')) else (int(r['season_shutouts']) if pd.notna(r.get('season_shutouts')) else None),
                    # Recent Orig (mapped back in shortlist function)
                    'recent_GP': int(r['recent_GP']) if pd.notna(r.get('recent_GP')) else None,
                    'recent_G': int(r['recent_G']) if pd.notna(r.get('recent_G')) else None,
                    'recent_A': int(r['recent_A']) if pd.notna(r.get('recent_A')) else None,
                    'recent_TP': int(r['recent_TP']) if pd.notna(r.get('recent_TP')) else None,
                    'recent_PIM': int(r['recent_PIM']) if pd.notna(r.get('recent_PIM')) else None,
                    'recent_plus_minus': int(r['recent_plus_minus']) if pd.notna(r.get('recent_plus_minus')) else None,
                    'recent_saves': int(r['recent_saves']) if pd.notna(r.get('recent_saves')) else None,
                    'recent_shots_against': int(r['recent_shots_against']) if pd.notna(r.get('recent_shots_against')) else None,
                    # Adjusted recent orig
                    'recent_adj_P_per_GP': float(r['recent_adj_P_per_GP']) if pd.notna(r.get('recent_adj_P_per_GP')) else None,
                    'recent_adj_save_pct': float(r['recent_adj_save_pct']) if pd.notna(r.get('recent_adj_save_pct')) else None,
                    # Freshness Orig
                    'days_since_last_game': int(r['days_since_last_game']) if pd.notna(r.get('days_since_last_game')) else None,
                }
                # Validate and append
                output_players.append(PlayerOut(**player_data))
            except Exception as parse_err:
                 # Log the specific record and error for debugging
                 logger.warning(f"Skipping record for player {r.get(player_id_col, 'UNKNOWN ID')} due to Pydantic parsing/validation error: {parse_err}. Record data: {r}")
                 continue # Skip this record

        return output_players

    except HTTPException as http_exc:
        # Re-raise HTTPException to let FastAPI handle it
        raise http_exc
    except Exception as e:
        # Catch unexpected errors during shortlist generation or processing
        logger.error(f"Error processing /shortlist/ request: {req}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing shortlist request.")

# --- Optional: Add a root endpoint for health check ---
@app.get("/")
def read_root():
    # Check if essential components are loaded
    if encoder is not None and all_embeddings is not None and ml_ready_df is not None and reference_df is not None:
        return {
            "status": "ML Service Ready",
            "device": str(device),
            "embeddings_shape": all_embeddings.shape,
            "ml_data_shape": ml_ready_df.shape,
            "reference_data_shape": reference_df.shape
        }
    else:
        # Indicate which components might be missing
        missing = []
        if encoder is None: missing.append("encoder")
        if all_embeddings is None: missing.append("embeddings")
        if ml_ready_df is None: missing.append("ml_ready_df")
        if reference_df is None: missing.append("reference_df")
        status_detail = f"ML Service Initializing or Failed. Missing: {', '.join(missing)}" if missing else "ML Service Initializing or Failed."
        logger.warning(f"Health check endpoint called, but service not fully ready. Missing: {missing}")
        # Return 503 if not ready
        raise HTTPException(status_code=503, detail=status_detail)
