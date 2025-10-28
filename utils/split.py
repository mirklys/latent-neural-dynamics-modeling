from utils.config import Config
import polars as pl

def create_splits(recordings: pl.DataFrame, split_params: Config, results_config: Config):
    assert split_params.within_session_split, "Only within session split is supported"
    
