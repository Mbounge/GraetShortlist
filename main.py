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
# import copy # Removed as it wasn't in the previous local API version
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
LOCAL_DATA_DIR = "/tmp/player_data"
ML_READY_DF_PATH = os.path.join(LOCAL_DATA_DIR, ML_READY_BLOB)
REFERENCE_DF_PATH = os.path.join(LOCAL_DATA_DIR, REFERENCE_BLOB)
SCALER_PIPELINE_PATH = os.path.join(LOCAL_DATA_DIR, SCALER_BLOB)
FEATURE_INFO_PATH = os.path.join(LOCAL_DATA_DIR, FEATURE_INFO_BLOB)
ENCODER_MODEL_PATH = os.path.join(LOCAL_DATA_DIR, ENCODER_MODEL_BLOB)
# --- End GCS Bucket/Blob Names and Local Paths ---

# --- Load BEST Hyperparameters from Feature Info or define here ---
EMBEDDING_DIM = 32
BEST_HIDDEN1_SIZE = 512
BEST_HIDDEN2_SIZE = 64
BEST_DROPOUT1 = 0.4940863172925382
BEST_DROPOUT2 = 0.17214600618183104
# --------------------------------------------------------------------

# Shortlisting Parameters
ARCHETYPE_PERCENTILE = 0.85

# --- NEW: Decoupled Scoring Weights ---
# Ensure these sum to 1.0
W_SIMILARITY_NEW = 0.30
W_SEASON_NEW     = 0.15
W_CAREER_NEW     = 0.15
W_TREND_NEW      = 0.15
W_RECENT_NEW     = 0.15
W_FRESH_NEW      = 0.10
# Verify sum: 0.30 + 0.15 + 0.15 + 0.15 + 0.15 + 0.10 = 1.0
# ------------------------------------

# Thresholds for artifact filtering within archetype context
MIN_GP_SEASON_CHECK = 5
MIN_GP_CAREER_CHECK = 10
SUSPICIOUS_SVP = 0.0
SUSPICIOUS_GAA = 0.0
SUSPICIOUS_PPG = 0.0

# System Settings
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_DATE = datetime(2025, 4, 26, tzinfo=timezone.utc) # Ensure timezone aware
# Determine the reference year for age group calculation (year the season ENDS)
SEASON_REFERENCE_YEAR = CURRENT_DATE.year if CURRENT_DATE.month < 8 else CURRENT_DATE.year + 1
logger.info(f"Using {SEASON_REFERENCE_YEAR} as the reference year for age group calculation (API).")


# --- Set Seeds ---
np.random.seed(SEED); torch.manual_seed(SEED) # Removed random.seed as import was removed
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
logger.info(f"Using device: {DEVICE}")

# --- FastAPI & CORS Setup ---
# !!! IMPORTANT: Replace with your actual frontend URL(s) for production !!!
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://www.graet.ai",
    "https://graet.ai",
    "https://interactive-avatar-next-js-demo-git-master-mbounge-s-team.vercel.app",
    "https://interactive-avatar-next-js-demo-hch93aeaf-mbounge-s-team.vercel.app"
]

app = FastAPI(title="Player Shortlist API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables for Loaded Data ---
ml_ready_df: Optional[pd.DataFrame] = None
reference_df: Optional[pd.DataFrame] = None
scaler_pipeline = None
feature_info: Dict[str, Any] = {}
feature_cols: List[str] = []
scaled_numeric_cols: List[str] = []
unique_nationalities: List[str] = []
unique_positions: List[str] = []
player_id_col: Optional[str] = None
player_embedder_global: Optional[nn.Module] = None # Renamed global encoder variable
all_embeddings_global: Optional[np.ndarray] = None # Renamed global embeddings variable
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
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        if not blob.exists():
             logger.error(f"Blob gs://{bucket_name}/{source_blob_name} does not exist!")
             raise FileNotFoundError(f"Blob gs://{bucket_name}/{source_blob_name} not found in GCS.")
        blob.download_to_filename(destination_file_name)
        logger.info(f"Successfully downloaded {source_blob_name}.")
    except FileNotFoundError as fnf_error: raise fnf_error
    except Exception as e: logger.error(f"Failed to download gs://{bucket_name}/{source_blob_name}: {e}"); raise


# --- Load Data & Model on Startup ---
@app.on_event("startup")
def load_model_and_data():
    global ml_ready_df, reference_df, scaler_pipeline, feature_info, feature_cols
    global scaled_numeric_cols, player_id_col, player_embedder_global, all_embeddings_global, device
    global unique_nationalities, unique_positions

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
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
         raise RuntimeError(f"Failed to download necessary files: {download_err}") from download_err

    try:
        logger.info("--- Loading Preprocessed Data and Metadata from Downloaded Files ---")
        ml_ready_df = pd.read_pickle(ML_READY_DF_PATH)
        reference_df = pd.read_pickle(REFERENCE_DF_PATH)
        with open(SCALER_PIPELINE_PATH, 'rb') as f: scaler_pipeline = pickle.load(f)
        with open(FEATURE_INFO_PATH, 'r') as f: feature_info = json.load(f)
        feature_cols = feature_info['feature_columns']
        scaled_numeric_cols = feature_info['scaled_numeric_columns']
        player_id_col = feature_info['player_id_column']
        unique_nationalities = feature_info.get('unique_nationalities', [])
        unique_positions = feature_info.get('unique_positions', [])
        logger.info("Data and metadata loaded from local files.")

        # --- Data Validation ---
        if ml_ready_df.empty or reference_df.empty or not feature_cols or not player_id_col: raise ValueError("Loaded data incomplete after download.")
        if player_id_col not in ml_ready_df.columns or player_id_col not in reference_df.columns: raise ValueError(f"Player ID column '{player_id_col}' missing after download.")
        missing_features = [col for col in feature_cols if col not in ml_ready_df.columns]
        if missing_features: raise ValueError(f"Missing features in ML-Ready DF after download: {missing_features}")
        required_ref_cols_startup = ['birth_year', 'age_group']
        required_ref_cols_startup.extend([
            'season_gamesPlayed_orig', 'user_career_gamesPlayed_orig',
            'season_svp_orig', 'season_gaa_orig', 'user_career_svp_orig', 'user_career_gaa_orig',
            'season_pointsPerGame_orig', 'user_career_pointsPerGame_orig'
        ])
        missing_ref_startup = [col for col in required_ref_cols_startup if col not in reference_df.columns]
        if missing_ref_startup: logger.warning(f"Reference DF missing columns required for age group filtering or artifact checks: {missing_ref_startup}.") # Warning instead of error

        if ml_ready_df[player_id_col].nunique() == len(ml_ready_df): ml_ready_df = ml_ready_df.set_index(player_id_col)
        else: raise ValueError("Player ID not unique in ML-Ready DF after download.")
        if reference_df[player_id_col].nunique() == len(reference_df): reference_df = reference_df.set_index(player_id_col)
        else: raise ValueError("Player ID not unique in Reference DF after download.")
        logger.info("Data validated and index set.")

        # --- Load Encoder Model ---
        input_size = len(feature_cols)
        if input_size <= 0: raise ValueError("Feature columns list is empty.")
        player_embedder_global = PlayerEmbedder(input_size).to(device) # Use correct global name
        logger.info(f"Loading trained encoder model state from local path: {ENCODER_MODEL_PATH}")
        player_embedder_global.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
        player_embedder_global.eval()
        logger.info("Encoder loaded and set to eval mode.")

        # --- Generate Embeddings ---
        logger.info("Generating player embeddings from loaded data...")
        if ml_ready_df is None or ml_ready_df.empty: raise ValueError("ml_ready_df not loaded.")
        X = ml_ready_df.loc[:, feature_cols].fillna(0).values.astype(np.float32)
        if np.isinf(X).any(): logger.warning("Infinity values found in feature matrix X. Replacing with 0."); X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        ds = TensorDataset(torch.from_numpy(X))
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        embeddings_list = []
        with torch.no_grad():
            for (batch,) in loader:
                if batch.shape[0] == 0: continue
                if batch.shape[0] <= 1 and any(isinstance(m, nn.BatchNorm1d) for m in player_embedder_global.modules()): logger.warning(f"Skipping batch size <= 1 during embedding generation."); continue
                embeddings_list.append(player_embedder_global(batch.to(device)).cpu().numpy()) # Use correct global name
        if not embeddings_list: raise RuntimeError("No embeddings generated.")
        all_embeddings_global = np.vstack(embeddings_list) # Use correct global name
        if all_embeddings_global.shape[0] != len(ml_ready_df): raise ValueError(f"Embeddings shape mismatch.")
        logger.info(f"Embeddings generated with shape: {all_embeddings_global.shape}")
        logger.info("--- Startup data and model loading complete ---")

    except Exception as e:
        logger.error(f"FATAL ERROR during startup after download: {e}", exc_info=True)
        raise RuntimeError(f"Failed to initialize model and data after download: {e}") from e


# --- Helper to determine age group (same as ETL/Inference) ---
def determine_age_group(birth_year, season_ref_year):
    """
    Determines the 'U' age group based on birth year and season reference year,
    applying Czech system rules (U16->U17, U18/U19->U20).
    """
    if not birth_year or not season_ref_year: return "Unknown"
    try:
        age_in_season = season_ref_year - birth_year
        if age_in_season <= 0: return "Unknown"
        u_category = age_in_season + 1
        if u_category == 16: return "U17"
        elif u_category == 18 or u_category == 19: return "U20"
        elif u_category >= 21: return "Senior"
        elif u_category >= 10: return f"U{u_category}"
        else: return "Youth"
    except Exception as e: logger.warning(f"Error determining age group for birth year {birth_year}, season ref {season_ref_year}: {e}"); return "Unknown"

# --- Archetype Creation Function (Modified for Indicators) ---
def create_archetype_input_vector(position_group_target, target_age_group_string, percentile,
                                  feature_cols, scaled_numeric_cols, scaler_pipeline,
                                  unique_nationalities, unique_positions,
                                  reference_data_df, player_id_col):
    """
    MODIFIED: Creates an input vector representing an archetype player,
    filtering the context pool by age_group, removing artifact players,
    and setting missing trend indicators to 0.
    """
    if not feature_cols: logger.error("Feature columns list is empty."); return None
    if reference_data_df is None or reference_data_df.empty: logger.error("Reference DataFrame missing."); return None

    archetype_input = pd.Series(0.0, index=feature_cols)
    original_target_values = {}
    scaled_target_values = {}
    target_gender = 'MEN'; gender_col_name = 'gender'; age_group_col_name = 'age_group'

    # --- 1. Initial Context Filtering ---
    required_context_cols = ['position_group', age_group_col_name, gender_col_name, 'age_orig']
    required_context_cols.extend([
        'season_gamesPlayed_orig', 'user_career_gamesPlayed_orig',
        'season_svp_orig', 'season_gaa_orig', 'user_career_svp_orig', 'user_career_gaa_orig',
        'season_pointsPerGame_orig', 'user_career_pointsPerGame_orig'
    ])
    context_df = pd.DataFrame()
    if all(col in reference_data_df.columns for col in required_context_cols):
        context_df = reference_data_df[
            (reference_data_df['position_group'] == position_group_target) &
            (reference_data_df[age_group_col_name] == target_age_group_string) &
            (reference_data_df[gender_col_name].fillna('Unknown').astype(str) == target_gender)
        ].copy()
        logger.info(f"Found {len(context_df)} initial context players for: PosGroup={position_group_target}, AgeGroup={target_age_group_string}, Gender={target_gender}.")
    else:
        missing_cols_context = [col for col in required_context_cols if col not in reference_data_df.columns]
        logger.error(f"Missing required context columns in Reference DF: {missing_cols_context}."); return None
    if context_df.empty: logger.warning(f"No players found for initial archetype context."); return None

    # --- 2. Artifact Filtering ---
    logger.info("Applying artifact filter to context players...")
    cleaned_context_df = context_df.copy()
    num_before_artifact_filter = len(cleaned_context_df)
    goalie_artifact_cols = ['season_gamesPlayed_orig', 'user_career_gamesPlayed_orig', 'season_svp_orig', 'season_gaa_orig', 'user_career_svp_orig', 'user_career_gaa_orig']
    skater_artifact_cols = ['season_gamesPlayed_orig', 'user_career_gamesPlayed_orig', 'season_pointsPerGame_orig', 'user_career_pointsPerGame_orig']
    artifact_mask = pd.Series(False, index=cleaned_context_df.index)
    if position_group_target == 'G':
        if all(col in cleaned_context_df.columns for col in goalie_artifact_cols):
            df_check_g = cleaned_context_df.copy()
            for col in goalie_artifact_cols:
                 df_check_g[col] = pd.to_numeric(df_check_g[col], errors='coerce')
                 if col in ['season_gaa_orig', 'user_career_gaa_orig']: df_check_g[col] = df_check_g[col].fillna(99.0)
                 else: df_check_g[col] = df_check_g[col].fillna(0.0)
            enough_season_gp_g = df_check_g['season_gamesPlayed_orig'] >= MIN_GP_SEASON_CHECK
            enough_career_gp_g = df_check_g['user_career_gamesPlayed_orig'] >= MIN_GP_CAREER_CHECK
            suspicious_season_stats_g = enough_season_gp_g & ((df_check_g['season_svp_orig'] == SUSPICIOUS_SVP) | (df_check_g['season_gaa_orig'] == SUSPICIOUS_GAA))
            suspicious_career_stats_g = enough_career_gp_g & ((df_check_g['user_career_svp_orig'] == SUSPICIOUS_SVP) | (df_check_g['user_career_gaa_orig'] == SUSPICIOUS_GAA))
            artifact_mask |= (suspicious_season_stats_g | suspicious_career_stats_g)
        else: logger.warning(f"Missing columns for goalie artifact check. Skipping.")
    elif position_group_target in ['F', 'D']:
        if all(col in cleaned_context_df.columns for col in skater_artifact_cols):
            df_check_s = cleaned_context_df.copy()
            for col in skater_artifact_cols:
                 df_check_s[col] = pd.to_numeric(df_check_s[col], errors='coerce')
                 if col in ['season_pointsPerGame_orig', 'user_career_pointsPerGame_orig']: df_check_s[col] = df_check_s[col].fillna(-1.0)
                 else: df_check_s[col] = df_check_s[col].fillna(0.0)
            enough_season_gp_s = df_check_s['season_gamesPlayed_orig'] >= MIN_GP_SEASON_CHECK
            enough_career_gp_s = df_check_s['user_career_gamesPlayed_orig'] >= MIN_GP_CAREER_CHECK
            suspicious_season_stats_s = enough_season_gp_s & (df_check_s['season_pointsPerGame_orig'] == SUSPICIOUS_PPG)
            suspicious_career_stats_s = enough_career_gp_s & (df_check_s['user_career_pointsPerGame_orig'] == SUSPICIOUS_PPG)
            artifact_mask |= (suspicious_season_stats_s | suspicious_career_stats_s)
        else: logger.warning(f"Missing columns for skater artifact check. Skipping.")
    cleaned_context_df = cleaned_context_df[~artifact_mask]
    num_after_artifact_filter = len(cleaned_context_df)
    num_removed_by_artifact = num_before_artifact_filter - num_after_artifact_filter
    logger.info(f"Removed {num_removed_by_artifact} artifact players. Remaining context: {num_after_artifact_filter}")
    if cleaned_context_df.empty:
        logger.error(f"Context empty after artifact filtering."); return None

    # --- 3. Calculate Target Original Values ---
    original_col_map = {}
    for scaled_col in scaled_numeric_cols:
        potential_orig = f"{scaled_col}_orig"
        if potential_orig in reference_data_df.columns: original_col_map[scaled_col] = potential_orig
        elif scaled_col in reference_data_df.columns: original_col_map[scaled_col] = scaled_col
        else: logger.debug(f"Cannot find original source for '{scaled_col}'.")
    percentile_cols_scaled = [col for col in scaled_numeric_cols if '_trend_' in col or 'recent_P' in col or 'recent_save' in col or 'game_freshness' in col or 'season_pointsPerGame' in col or 'season_svp' in col or 'adj_season' in col or 'adj_user_career' in col]
    inverse_percentile_cols = [col for col in percentile_cols_scaled if 'GAA' in col or 'gaa' in col]
    numeric_cols_to_process = [col for col in scaled_numeric_cols if col in feature_cols]
    median_age_context = cleaned_context_df['age_orig'].median() if 'age_orig' in cleaned_context_df.columns and cleaned_context_df['age_orig'].notna().any() else 20.0
    for scaled_col in numeric_cols_to_process:
        if scaled_col.endswith('_missing'):
            target_value_orig = 0.0
        else:
            target_value_orig = 0.0; original_context_col = original_col_map.get(scaled_col)
            source_df_for_stats = cleaned_context_df
            if not original_context_col or original_context_col not in cleaned_context_df.columns or not cleaned_context_df[original_context_col].notna().any():
                 original_context_col = original_col_map.get(scaled_col); source_df_for_stats = reference_data_df
                 logger.debug(f"Falling back to global reference for '{scaled_col}'")
            if original_context_col and original_context_col in source_df_for_stats.columns and source_df_for_stats[original_context_col].notna().any():
                context_series = source_df_for_stats[original_context_col].dropna()
                if not context_series.empty:
                    try:
                        if scaled_col in percentile_cols_scaled:
                            target_percentile_val = percentile;
                            if scaled_col in inverse_percentile_cols: target_percentile_val = 1.0 - percentile
                            target_value_orig = context_series.quantile(target_percentile_val)
                        elif scaled_col == 'age': target_value_orig = median_age_context
                        else: target_value_orig = context_series.median()
                    except Exception as e_stat: logger.warning(f"Error calculating stat for {original_context_col}: {e_stat}. Using 0."); target_value_orig = 0.0
                else: target_value_orig = 0.0; logger.warning(f"Context series empty for {original_context_col}.")
            else: logger.warning(f"No valid data for archetype column '{scaled_col}'. Using 0."); target_value_orig = 0.0
            if pd.isna(target_value_orig): logger.warning(f"Target value NaN for '{scaled_col}'. Using 0."); target_value_orig = 0.0
        original_target_values[scaled_col] = target_value_orig

    # --- 4. Scale Target Values ---
    if scaler_pipeline and numeric_cols_to_process:
        target_df_for_scaling = pd.DataFrame([original_target_values])
        pipeline_features = []
        if hasattr(scaler_pipeline, 'feature_names_in_'): pipeline_features = scaler_pipeline.feature_names_in_
        elif hasattr(scaler_pipeline, 'steps') and hasattr(scaler_pipeline.steps[0][1], 'feature_names_in_'): pipeline_features = scaler_pipeline.steps[0][1].feature_names_in_
        else: logger.warning("Cannot determine feature names from scaler pipeline."); pipeline_features = numeric_cols_to_process
        cols_in_vector = target_df_for_scaling.columns
        missing_cols = set(pipeline_features) - set(cols_in_vector)
        extra_cols = set(cols_in_vector) - set(pipeline_features)
        if missing_cols: logger.warning(f"Archetype vector missing columns: {missing_cols}. Filling with 0."); [target_df_for_scaling.insert(loc=0, column=col, value=0.0) for col in missing_cols]
        if extra_cols: logger.warning(f"Archetype vector has extra columns: {extra_cols}. Dropping."); target_df_for_scaling = target_df_for_scaling.drop(columns=list(extra_cols))
        target_df_for_scaling = target_df_for_scaling[pipeline_features]
        target_df_for_scaling = target_df_for_scaling.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        try:
            scaled_targets_array = scaler_pipeline.transform(target_df_for_scaling)
            scaled_target_values = dict(zip(pipeline_features, scaled_targets_array[0]))
        except Exception as e: logger.error(f"Error scaling archetype vector: {e}. Using 0.5 default."); scaled_target_values = {col: 0.5 for col in pipeline_features}
    else: logger.warning("Scaler pipeline not available/no numeric columns. Using 0.5 default."); scaled_target_values = {col: 0.5 for col in numeric_cols_to_process}

    # --- 5. Assemble Final Archetype Vector ---
    for col, scaled_value in scaled_target_values.items():
        if col in archetype_input.index: archetype_input[col] = scaled_value
    for p in unique_positions:
        p_col = f'pos_{p}'
        if p_col in archetype_input.index: archetype_input[p_col] = 1.0 if p == position_group_target else 0.0
    target_pos_col = f'pos_{position_group_target}'
    if target_pos_col not in archetype_input.index: logger.warning(f"Target position column '{target_pos_col}' not found.");
    for n in unique_nationalities:
        n_col = f'nat_{n}'
        if n_col in archetype_input.index: archetype_input[n_col] = 1.0 if n == 'CZ' else 0.0
    if 'nat_CZ' not in archetype_input.index and unique_nationalities:
         fallback_nat_col = f'nat_{unique_nationalities[0]}'
         if fallback_nat_col in archetype_input.index: archetype_input[fallback_nat_col] = 1.0; logger.warning(f"nat_CZ not found, setting {fallback_nat_col}=1.0.")
    for g in ['MEN', 'WOMEN', 'UNKNOWN']:
        g_col = f'gender_{g}'
        if g_col in archetype_input.index: archetype_input[g_col] = 1.0 if g == target_gender else 0.0
    target_gender_col = f'gender_{target_gender}'
    if target_gender_col not in archetype_input.index: logger.warning(f"Target gender column '{target_gender_col}' not found.");
    indicator_cols = [col for col in feature_cols if col.endswith('_missing')]
    for ind_col in indicator_cols:
        if ind_col in archetype_input.index: archetype_input[ind_col] = 0.0
    if archetype_input.isnull().any():
        logger.warning("NaNs found in final archetype vector. Filling with 0.")
        archetype_input = archetype_input.fillna(0.0)
    return archetype_input.values.astype(np.float32)


# --- Shortlist Generation Function (Modified for Decoupled Scores & Indicators) ---
def generate_birth_year_shortlist(birth_year: int,
                                  ml_ready_player_df: pd.DataFrame, # Local argument
                                  all_player_embeddings: np.ndarray, # Local argument
                                  reference_df: pd.DataFrame, # Local argument
                                  scaler_pipeline, # Local argument
                                  feature_cols, # Local argument
                                  scaled_numeric_cols, # Local argument
                                  player_id_col, # Local argument
                                  top_n=50,
                                  archetype_percentile=ARCHETYPE_PERCENTILE,
                                  # Use NEW weights here
                                  w_similarity=W_SIMILARITY_NEW,
                                  w_season=W_SEASON_NEW,
                                  w_career=W_CAREER_NEW,
                                  w_trend=W_TREND_NEW,
                                  w_recent=W_RECENT_NEW,
                                  w_fresh=W_FRESH_NEW) -> dict:
    """
    Generates a ranked shortlist for a specific birth year, filtering by birth year and gender,
    calculating similarity to a robust archetype based on age group, and combining DECOUPLED scores.
    Relies on embeddings trained with missing trend indicators.
    Handles players with no valid recent performance data correctly.
    """
    # Use globally loaded unique lists and encoder - these ARE global
    global unique_nationalities, unique_positions, player_embedder_global
    # Reference other globals needed ONLY if not passed as args (like reference_df_global for final mapping)
    global reference_df_global

    logger.info(f"\n--- Generating Shortlist: Birth Year {birth_year} (Filtering by Birth Year & Gender='MEN') (Top {top_n}) ---")

    # --- Initial checks ---
    # CORRECTED: Check the function arguments and the correct global names
    if (all_player_embeddings is None
            or reference_df is None         # Check local argument
            or ml_ready_player_df is None   # Check local argument
            or player_embedder_global is None # Check correct global name
            or scaler_pipeline is None      # Check local argument
            or not feature_cols             # Check local argument
            or not player_id_col            # Check local argument
            or not scaled_numeric_cols):    # Check local argument
        logger.error("Required components (arguments or globals) not set for shortlist generation.")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    # Check consistency between the passed ml_ready_df and the global embeddings
    if len(ml_ready_player_df) != all_embeddings_global.shape[0]: # Use global embeddings here for check
        logger.error(f"Mismatch between passed ml_ready_df length ({len(ml_ready_player_df)}) and global embeddings length ({all_embeddings_global.shape[0]}).")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    if ml_ready_player_df.index.name != player_id_col:
         logger.error("ML Ready DF index is not player_id. Cannot map embeddings correctly.")
         return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}

    # --- Filtering by Birth Year and Gender (using the passed reference_df) ---
    birth_year_col_name = 'birth_year'; gender_col_name = 'gender'; target_gender = 'MEN'
    required_cols = [birth_year_col_name, gender_col_name, 'position_group', 'age_group']
    # Use the passed reference_df for filtering
    if not all(col in reference_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in reference_df.columns]
        logger.error(f"Required columns missing from Reference DF: {missing}. Cannot filter.")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    logger.info(f"Filtering for players with {birth_year_col_name} == {birth_year} AND {gender_col_name} == '{target_gender}'")
    ref_local = reference_df.copy() # Use the passed reference_df
    ref_local[birth_year_col_name] = pd.to_numeric(ref_local[birth_year_col_name], errors='coerce')
    valid_birth_year_mask = ref_local[birth_year_col_name].notna()
    ref_local[gender_col_name] = ref_local[gender_col_name].fillna('Unknown').astype(str)
    year_filtered_ref_df = ref_local[valid_birth_year_mask & (ref_local[birth_year_col_name] == birth_year) & (ref_local[gender_col_name] == target_gender)]
    if year_filtered_ref_df.empty:
        logger.warning(f"No players found in Reference DF with {birth_year_col_name} == {birth_year} AND {gender_col_name} == '{target_gender}'.")
        return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    logger.info(f"Found {len(year_filtered_ref_df)} players in Reference DF for birth year {birth_year} and gender '{target_gender}'.")

    # --- Get Embeddings for Filtered Players ---
    target_player_ids = year_filtered_ref_df.index.tolist()
    # Use the passed ml_ready_player_df
    ml_df_index_to_embedding_idx = {idx: i for i, idx in enumerate(ml_ready_player_df.index)}
    year_embedding_indices = [ml_df_index_to_embedding_idx.get(pid) for pid in target_player_ids]
    valid_map = [(pid, idx) for pid, idx in zip(target_player_ids, year_embedding_indices) if idx is not None]
    if len(valid_map) != len(target_player_ids): logger.warning(f"Could not map all {len(target_player_ids)} filtered players to embeddings.")
    if not valid_map: logger.error("Could not map any filtered players to embeddings."); return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    filtered_pids = [pid for pid, idx in valid_map]; filtered_embedding_indices = [idx for pid, idx in valid_map]
    # Use the passed all_player_embeddings
    year_embeddings = all_player_embeddings[filtered_embedding_indices]
    year_filtered_ref_df = year_filtered_ref_df.loc[filtered_pids]

    # --- Generate Archetype Embeddings ---
    target_age_group_string = determine_age_group(birth_year, SEASON_REFERENCE_YEAR)
    logger.info(f"Target Age Group for Archetype (Birth Year {birth_year}): {target_age_group_string}")
    archetype_embeddings = {}
    # Use the correct global encoder name
    if player_embedder_global is None: logger.error("Encoder model not loaded."); return {'G': pd.DataFrame(), 'D': pd.DataFrame(), 'F': pd.DataFrame()}
    player_embedder_global.eval()
    with torch.no_grad():
        for pos_group in ['G', 'D', 'F']:
            logger.info(f"Generating archetype embedding for PosGroup={pos_group}, TargetAgeGroup={target_age_group_string}, Pct={archetype_percentile}")
            # Pass the correct arguments (local or global as appropriate)
            archetype_input_vec = create_archetype_input_vector(
                pos_group, target_age_group_string, archetype_percentile,
                feature_cols, scaled_numeric_cols, scaler_pipeline, # Use passed args
                unique_nationalities, unique_positions, # Use globals
                reference_df, player_id_col # Use passed args
            )
            if archetype_input_vec is None: logger.error(f"Failed to create archetype vector for {pos_group}."); archetype_embeddings[pos_group] = None; continue
            archetype_input_tensor = torch.tensor(archetype_input_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE);
            if archetype_input_tensor.shape[0] <= 1 and hasattr(player_embedder_global, 'modules') and any(isinstance(m, nn.BatchNorm1d) for m in player_embedder_global.modules()): logger.warning(f"Generating archetype embedding for {pos_group} with batch size 1 and BatchNorm active.")
            archetype_embeddings[pos_group] = player_embedder_global(archetype_input_tensor).cpu().numpy() # Use global encoder

    # --- Calculate Scores ---
    # Initialize lists for NEW component scores
    final_scores = []; similarity_scores = []; season_scores = []; career_scores = []
    trend_scores = []; recent_scores = []; fresh_scores = []
    player_ids_for_scores = []

    for i, player_id in enumerate(year_filtered_ref_df.index):
        player_ids_for_scores.append(player_id)
        try:
            player_row_ref = year_filtered_ref_df.loc[player_id]
            player_row_ml = ml_ready_player_df.loc[player_id] # Use passed ml_ready_df
            player_pos_group = player_row_ref.get('position_group', 'Unknown')
            if pd.isna(player_pos_group): player_pos_group = 'Unknown'
        except KeyError: logger.warning(f"Could not find data for player {player_id}. Skipping."); continue

        # --- Component Score Calculation ---
        # 1. Similarity Score
        player_embedding = year_embeddings[i:i+1]
        similarity_score = 0.0
        archetype_embed = archetype_embeddings.get(player_pos_group)
        if archetype_embed is not None:
            norm_player = np.linalg.norm(player_embedding); norm_archetype = np.linalg.norm(archetype_embed)
            if norm_player > 1e-9 and norm_archetype > 1e-9:
                cos_sim = np.dot(player_embedding, archetype_embed.T) / (norm_player * norm_archetype);
                similarity_score = max(0.0, min(1.0, (cos_sim.item() + 1.0) / 2.0))
        similarity_score = np.clip(similarity_score, 0.0, 1.0)

        # 2. Freshness Score
        freshness_score = np.clip(player_row_ml.get('game_freshness', 0.0), 0.0, 1.0)

        # 3. Recent Score (Based on primary recent metric, 0 if no valid recent data)
        recent_score = 0.0
        if player_pos_group == 'G':
            orig_recent_stat_col = 'recent_adj_save_pct_orig'; scaled_recent_stat_col = 'recent_adj_save_pct'
        elif player_pos_group in ['D', 'F']:
            orig_recent_stat_col = 'recent_adj_P_per_GP_orig'; scaled_recent_stat_col = 'recent_adj_P_per_GP'
        else: orig_recent_stat_col = None; scaled_recent_stat_col = None

        if orig_recent_stat_col and orig_recent_stat_col in player_row_ref and pd.notna(player_row_ref[orig_recent_stat_col]):
            if scaled_recent_stat_col and scaled_recent_stat_col in player_row_ml:
                scaled_val = player_row_ml[scaled_recent_stat_col]
                recent_score = float(scaled_val) if pd.notna(scaled_val) else 0.0
        recent_score = np.clip(recent_score, 0.0, 1.0)

        # 4. Trend Score (Average of relevant scaled trend values)
        player_trends_scaled = []
        if player_pos_group == 'G':
            gaa_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_GAA_trend' in col and not col.endswith('_missing')]
            svp_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_SVP_trend' in col and not col.endswith('_missing')]
            if gaa_trends: player_trends_scaled.extend([1.0 - t for t in gaa_trends])
            if svp_trends: player_trends_scaled.extend(svp_trends)
        elif player_pos_group in ['D', 'F']:
            p_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_P_per_GP_trend' in col and not col.endswith('_missing')]
            g_trends = [player_row_ml.get(col, 0.0) for col in feature_cols if 'adj_G_per_GP_trend' in col and not col.endswith('_missing')]
            if p_trends: player_trends_scaled.extend(p_trends)
            if g_trends: player_trends_scaled.extend(g_trends)
        trend_score = np.mean(player_trends_scaled) if player_trends_scaled else 0.0
        trend_score = np.clip(trend_score, 0.0, 1.0)

        # 5. Season Score (Average of relevant scaled adjusted season rates)
        season_rates_scaled = []
        if player_pos_group == 'G':
            svp = player_row_ml.get('adj_season_svp', 0.0); gaa = player_row_ml.get('adj_season_gaa', 0.0); so_rate = player_row_ml.get('adj_season_SO_per_GP', 0.0)
            season_rates_scaled.append(float(svp) if pd.notna(svp) else 0.0)
            season_rates_scaled.append(1.0 - (float(gaa) if pd.notna(gaa) else 1.0))
            season_rates_scaled.append(float(so_rate) if pd.notna(so_rate) else 0.0)
        elif player_pos_group in ['D', 'F']:
            ppg = player_row_ml.get('adj_season_pointsPerGame', 0.0); gpg = player_row_ml.get('adj_season_G_per_GP', 0.0); apg = player_row_ml.get('adj_season_A_per_GP', 0.0)
            season_rates_scaled.append(float(ppg) if pd.notna(ppg) else 0.0)
            season_rates_scaled.append(float(gpg) if pd.notna(gpg) else 0.0)
            season_rates_scaled.append(float(apg) if pd.notna(apg) else 0.0)
        season_score = np.mean(season_rates_scaled) if season_rates_scaled else 0.0
        season_score = np.clip(season_score, 0.0, 1.0)

        # 6. Career Score (Average of relevant scaled adjusted career rates)
        career_rates_scaled = []
        if player_pos_group == 'G':
            svp = player_row_ml.get('adj_user_career_svp', 0.0); gaa = player_row_ml.get('adj_user_career_gaa', 0.0); so_rate = player_row_ml.get('adj_user_career_SO_per_GP', 0.0)
            career_rates_scaled.append(float(svp) if pd.notna(svp) else 0.0)
            career_rates_scaled.append(1.0 - (float(gaa) if pd.notna(gaa) else 1.0))
            career_rates_scaled.append(float(so_rate) if pd.notna(so_rate) else 0.0)
        elif player_pos_group in ['D', 'F']:
            ppg = player_row_ml.get('adj_user_career_pointsPerGame', 0.0); gpg = player_row_ml.get('adj_user_career_G_per_GP', 0.0); apg = player_row_ml.get('adj_user_career_A_per_GP', 0.0)
            career_rates_scaled.append(float(ppg) if pd.notna(ppg) else 0.0)
            career_rates_scaled.append(float(gpg) if pd.notna(gpg) else 0.0)
            career_rates_scaled.append(float(apg) if pd.notna(apg) else 0.0)
        career_score = np.mean(career_rates_scaled) if career_rates_scaled else 0.0
        career_score = np.clip(career_score, 0.0, 1.0)

        # --- Calculate Final Score using NEW weights ---
        final_score = (w_similarity * similarity_score) + \
                      (w_season * season_score) + \
                      (w_career * career_score) + \
                      (w_trend * trend_score) + \
                      (w_recent * recent_score) + \
                      (w_fresh * freshness_score)

        # Append ALL component scores
        similarity_scores.append(similarity_score)
        season_scores.append(season_score)
        career_scores.append(career_score)
        trend_scores.append(trend_score)
        recent_scores.append(recent_score)
        fresh_scores.append(freshness_score)
        final_scores.append(final_score)

    # --- Combine Scores and Rank ---
    scores_df = pd.DataFrame({
        player_id_col: player_ids_for_scores,
        'final_score': final_scores,
        'similarity_score': similarity_scores, # Renamed for clarity
        'season_score': season_scores,
        'career_score': career_scores,
        'trend_score': trend_scores,
        'recent_score': recent_scores,
        'freshness_score': fresh_scores
    }).set_index(player_id_col)

    if year_filtered_ref_df.index.name != player_id_col:
        year_filtered_ref_df = year_filtered_ref_df.set_index(player_id_col)
    year_filtered_ref_df = year_filtered_ref_df.join(scores_df, how='left')
    score_cols_list = ['final_score', 'similarity_score', 'season_score', 'career_score', 'trend_score', 'recent_score', 'freshness_score']
    year_filtered_ref_df[score_cols_list] = year_filtered_ref_df[score_cols_list].fillna(0.0)

    # --- Create Shortlists per Position ---
    shortlists = {}
    for pos_group in ['G', 'D', 'F']:
        pos_mask = year_filtered_ref_df['position_group'] == pos_group
        pos_df = year_filtered_ref_df[pos_mask].copy()
        num_found_before_head = len(pos_df)
        logger.info(f"Found {num_found_before_head} players for pos group {pos_group}, birth year {birth_year} (gender '{target_gender}') before taking top {top_n}.")
        if not pos_df.empty and 'final_score' in pos_df.columns:
            ranked_pos_df = pos_df.sort_values('final_score', ascending=False).head(top_n);
            recent_orig_cols_map = {
                'recent_GP': 'recent_GP_orig','recent_G': 'recent_G_orig','recent_A': 'recent_A_orig',
                'recent_TP': 'recent_TP_orig','recent_PIM': 'recent_PIM_orig','recent_plus_minus': 'recent_plus_minus_orig',
                'recent_saves': 'recent_saves_orig','recent_shots_against': 'recent_shots_against_orig',
                'recent_adj_P_per_GP': 'recent_adj_P_per_GP_orig','recent_adj_save_pct': 'recent_adj_save_pct_orig'
            }
            # Use the correct reference_df_global for mapping back original stats
            for target_col, source_col in recent_orig_cols_map.items():
                 if source_col in reference_df_global.columns:
                      ranked_pos_df[target_col] = ranked_pos_df.index.map(reference_df_global[source_col])
                 else:
                      ranked_pos_df[target_col] = np.nan
            shortlists[pos_group] = ranked_pos_df;
        else: logger.warning(f"No players/scores found for pos group {pos_group}, birth year {birth_year}."); shortlists[pos_group] = pd.DataFrame()
    return shortlists


# --- FastAPI Request/Response Models ---
class ShortlistRequest(BaseModel):
    birth_year: int
    position: Literal['G','D','F']
    top_n: int = Field(default=10, ge=1, le=100)

class PlayerOut(BaseModel):
    # Core Identifiers
    player_id: str
    name: Optional[str] = None
    age_orig: Optional[int] = None
    birth_year: Optional[int] = None
    age_group: Optional[str] = None
    position_orig: Optional[str] = None
    nationality_orig: Optional[str] = None
    position_group: Optional[str] = None
    gender: Optional[str] = None

    # Scores (Updated)
    final_score: Optional[float] = None
    similarity_score: Optional[float] = None
    season_score: Optional[float] = None
    career_score: Optional[float] = None
    trend_score: Optional[float] = None
    recent_score: Optional[float] = None
    freshness_score: Optional[float] = None

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
    recent_adj_P_per_GP: Optional[float] = None
    recent_adj_save_pct: Optional[float] = None

    # Freshness Original Value
    days_since_last_game: Optional[int] = None


# --- Endpoint ---
@app.post("/shortlist/", response_model=List[PlayerOut])
def get_shortlist(req: ShortlistRequest):
    logger.info(f"Received shortlist request: year={req.birth_year}, position={req.position}, top_n={req.top_n}")
    try:
        # --- CORRECTED READINESS CHECK ---
        if (reference_df is None
                or ml_ready_df is None
                or all_embeddings_global is None
                or player_embedder_global is None
                or scaler_pipeline is None
                or not feature_cols
                or not player_id_col
                or not scaled_numeric_cols):
             logger.error("Service not ready: Data or model components not loaded.")
             raise HTTPException(status_code=503, detail="Service not ready, data or model not loaded.")

        # Generate the shortlist for the requested year
        shortlists_for_year = generate_birth_year_shortlist(
            birth_year=req.birth_year,
            ml_ready_player_df=ml_ready_df, # Pass the indexed DF loaded globally
            all_player_embeddings=all_embeddings_global, # Pass global embeddings
            reference_df=reference_df, # Pass the indexed DF loaded globally
            scaler_pipeline=scaler_pipeline, # Pass global scaler
            feature_cols=feature_cols, # Pass global feature list
            scaled_numeric_cols=scaled_numeric_cols, # Pass global scaled numeric list
            player_id_col=player_id_col, # Pass global player ID column name
            top_n=req.top_n, # Use requested top_n
            archetype_percentile=ARCHETYPE_PERCENTILE,
            # Pass NEW weights
            w_similarity=W_SIMILARITY_NEW,
            w_season=W_SEASON_NEW,
            w_career=W_CAREER_NEW,
            w_trend=W_TREND_NEW,
            w_recent=W_RECENT_NEW,
            w_fresh=W_FRESH_NEW
        )

        # Get the specific position dataframe
        df = shortlists_for_year.get(req.position)

        # Handle empty results
        if df is None or df.empty:
            logger.info(f"No players found for position {req.position}, year {req.birth_year}.")
            return [] # Return empty list

        # Convert DataFrame to list of dictionaries
        if df.index.name == player_id_col:
            recs = df.reset_index().to_dict(orient='records')
        else:
             if player_id_col not in df.columns:
                 logger.error(f"Player ID column '{player_id_col}' missing from shortlist DataFrame.")
                 return []
             recs = df.to_dict(orient='records')

        # --- Format Output using Pydantic Model ---
        output_players = []
        for r in recs:
            try:
                # Prepare data dictionary, mapping new score columns
                player_data = {
                    'player_id': str(r.get(player_id_col, 'N/A')),
                    'name': r.get('name'),
                    'age_orig': int(r['age_orig']) if pd.notna(r.get('age_orig')) else None,
                    'birth_year': int(r['birth_year']) if pd.notna(r.get('birth_year')) else None,
                    'age_group': r.get('age_group'),
                    'position_orig': r.get('position_orig'),
                    'nationality_orig': r.get('nationality_orig'),
                    'position_group': r.get('position_group'),
                    'gender': r.get('gender'),
                    # Scores (Updated)
                    'final_score': float(r['final_score']) if pd.notna(r.get('final_score')) else None,
                    'similarity_score': float(r['similarity_score']) if pd.notna(r.get('similarity_score')) else None,
                    'season_score': float(r['season_score']) if pd.notna(r.get('season_score')) else None,
                    'career_score': float(r['career_score']) if pd.notna(r.get('career_score')) else None,
                    'trend_score': float(r['trend_score']) if pd.notna(r.get('trend_score')) else None,
                    'recent_score': float(r['recent_score']) if pd.notna(r.get('recent_score')) else None,
                    'freshness_score': float(r['freshness_score']) if pd.notna(r.get('freshness_score')) else None,
                    # Season Orig
                    'season_gamesPlayed_orig': int(r['season_gamesPlayed_orig']) if pd.notna(r.get('season_gamesPlayed_orig')) else None,
                    'season_goals_orig': int(r['season_goals_orig']) if pd.notna(r.get('season_goals_orig')) else None,
                    'season_assists_orig': int(r['season_assists_orig']) if pd.notna(r.get('season_assists_orig')) else None,
                    'season_points_orig': int(r['season_points_orig']) if pd.notna(r.get('season_points_orig')) else None,
                    'season_pointsPerGame_orig': float(r['season_pointsPerGame_orig']) if pd.notna(r.get('season_pointsPerGame_orig')) else None,
                    'season_gaa_orig': float(r['season_gaa_orig']) if pd.notna(r.get('season_gaa_orig')) else None,
                    'season_svp_orig': float(r['season_svp_orig']) if pd.notna(r.get('season_svp_orig')) else None,
                    'season_shutouts_orig': int(r['season_shutouts_orig']) if pd.notna(r.get('season_shutouts_orig')) else (int(r['season_shutouts']) if pd.notna(r.get('season_shutouts')) else None),
                    # Recent Orig
                    'recent_GP': int(r['recent_GP']) if pd.notna(r.get('recent_GP')) else None,
                    'recent_G': int(r['recent_G']) if pd.notna(r.get('recent_G')) else None,
                    'recent_A': int(r['recent_A']) if pd.notna(r.get('recent_A')) else None,
                    'recent_TP': int(r['recent_TP']) if pd.notna(r.get('recent_TP')) else None,
                    'recent_PIM': int(r['recent_PIM']) if pd.notna(r.get('recent_PIM')) else None,
                    'recent_plus_minus': int(r['recent_plus_minus']) if pd.notna(r.get('recent_plus_minus')) else None,
                    'recent_saves': int(r['recent_saves']) if pd.notna(r.get('recent_saves')) else None,
                    'recent_shots_against': int(r['recent_shots_against']) if pd.notna(r.get('recent_shots_against')) else None,
                    'recent_adj_P_per_GP': float(r['recent_adj_P_per_GP']) if pd.notna(r.get('recent_adj_P_per_GP')) else None,
                    'recent_adj_save_pct': float(r['recent_adj_save_pct']) if pd.notna(r.get('recent_adj_save_pct')) else None,
                    # Freshness Orig
                    'days_since_last_game': int(r['days_since_last_game']) if pd.notna(r.get('days_since_last_game')) else None,
                }
                output_players.append(PlayerOut(**player_data))
            except Exception as parse_err:
                 logger.warning(f"Skipping record for player {r.get(player_id_col, 'UNKNOWN')} due to Pydantic parsing/validation error: {parse_err}. Record keys: {list(r.keys())}")
                 continue

        return output_players

    except HTTPException: raise
    except Exception as e:
        logger.error(f"Error processing /shortlist/ request: {req}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing shortlist request.")

# --- Optional: Add a root endpoint for health check ---
@app.get("/")
def read_root():
    if player_embedder_global is not None and all_embeddings_global is not None and ml_ready_df is not None and reference_df is not None:
        return {
            "status": "ML Service Ready",
            "device": str(device),
            "embeddings_shape": all_embeddings_global.shape if all_embeddings_global is not None else None,
            "ml_data_shape": ml_ready_df.shape if ml_ready_df is not None else None,
            "reference_data_shape": reference_df.shape if reference_df is not None else None
        }
    else:
        missing = []
        if player_embedder_global is None: missing.append("encoder")
        if all_embeddings_global is None: missing.append("embeddings")
        if ml_ready_df is None: missing.append("ml_ready_df")
        if reference_df is None: missing.append("reference_df")
        status_detail = f"ML Service Initializing or Failed. Missing: {', '.join(missing)}" if missing else "ML Service Initializing or Failed."
        logger.warning(f"Health check endpoint called, but service not fully ready. Missing: {missing}")
        raise HTTPException(status_code=503, detail=status_detail)

