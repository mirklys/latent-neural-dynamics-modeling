import polars as pl
from pathlib import Path

from utils.polars import (
    read_tsv,
    add_modality_path,
    split_file_path,
    remove_rows_with,
    explode_files,
    keep_rows_with,
    band_pass_resample,
)
from .events import construct_events_table
from .motion import construct_motion_table
from utils.file_handling import get_child_subchilds_tuples

from utils.config import Config
from utils.logger import get_logger

iEEG_SCHEMA = pl.Struct(
    [
        pl.Field("LFP_1", pl.List(pl.Float32)),
        pl.Field("LFP_2", pl.List(pl.Float32)),
        pl.Field("LFP_3", pl.List(pl.Float32)),
        pl.Field("LFP_4", pl.List(pl.Float32)),
        pl.Field("LFP_5", pl.List(pl.Float32)),
        pl.Field("LFP_6", pl.List(pl.Float32)),
        pl.Field("LFP_7", pl.List(pl.Float32)),
        pl.Field("LFP_8", pl.List(pl.Float32)),
        pl.Field("LFP_9", pl.List(pl.Float32)),
        pl.Field("LFP_10", pl.List(pl.Float32)),
        pl.Field("LFP_11", pl.List(pl.Float32)),
        pl.Field("LFP_12", pl.List(pl.Float32)),
        pl.Field("LFP_13", pl.List(pl.Float32)),
        pl.Field("LFP_14", pl.List(pl.Float32)),
        pl.Field("LFP_15", pl.List(pl.Float32)),
        pl.Field("LFP_16", pl.List(pl.Float32)),
        pl.Field("ECOG_1", pl.List(pl.Float32)),
        pl.Field("ECOG_2", pl.List(pl.Float32)),
        pl.Field("ECOG_3", pl.List(pl.Float32)),
        pl.Field("ECOG_4", pl.List(pl.Float32)),
        pl.Field("EOG_1", pl.List(pl.Float32)),
        pl.Field("EOG_2", pl.List(pl.Float32)),
        pl.Field("EOG_3", pl.List(pl.Float32)),
        pl.Field("EOG_4", pl.List(pl.Float32)),
        pl.Field("sfreq", pl.List(pl.Float32)),
    ]
)


def _add_full_data(participants: pl.DataFrame, config: Config) -> pl.DataFrame:
    """CODE IF PARTIAL PREPROCESSING IS NOT DONE
    participants = read_tsv(data_path / config.participants_table_name)
    participants = participants.filter(
        pl.col("participant_id").str.starts_with("sub-PD")
    )
    participants = participants.drop("age", "sex", "hand", "weight", "height")

    participants = participants.with_columns(
        pl.concat_str(
            pl.lit(str(data_path)), pl.col("participant_id"), separator="/"
        ).alias("participant_path")
    )
    participants = explode_files(participants, "participant_path", "session_path")
    """
    ieeg_participants = _add_ieeg_data(participants, config)
    motion_participants = construct_motion_table(
        participants.filter(pl.col("labels")), config
    )

    participants = ieeg_participants.join(
        motion_participants, on=["participant_id", "session", "block"], how="left"
    )

    participants = participants.drop(
        "participant_path",
        "session_path",
        "participant_path_right",
        "session_path_right",
        "ieeg_path",
        "ieeg_file",
        "ieeg_headers_file",
        "motion_path",
        "motion_file",
        "type",
        "data_format",
        strict=False,
    )

    participants = (
        participants.with_columns(
            (pl.col("trials") - pl.col("trial"))
            .list.eval(pl.element().eq(0))
            .list.arg_max()
            .alias("trial_index")
        )
        .with_columns(
            pl.col("onsets")
            .list.get(pl.col("trial_index"), null_on_oob=True)
            .alias("onset"),
            pl.col("trials")
            .list.get(pl.col("trial_index"), null_on_oob=True)
            .alias("trial"),
            pl.col("yscores")
            .list.get(pl.col("trial_index"), null_on_oob=True)
            .alias("yscore"),
        )
        .drop("trials", "onsets", "yscores", "trial_index")
    )

    participants = _chunk_recordings(
        participants,
        config.ieeg_process.chunk_margin,
        config.ieeg_process.resampled_freq,
    ).drop("ieeg_parquet")
    return participants


def construct_participants_table(config: Config):
    logger = get_logger()

    data_path = Path(config.data_directory)
    save_path = Path(config.save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    participants_partitions = get_child_subchilds_tuples(
        data_path / config.participants_intermediate_table_name
    )

    for p_part in participants_partitions:
        participant_id, session = p_part
        p_partition_path = (
            data_path / "participants.parquet" / participant_id / session / "*"
        )
        participants = pl.read_parquet(p_partition_path)

        logger.info(f"Loaded participants from: {p_partition_path}")

        participants = _add_full_data(participants, config)

        participants = participants.select(
            "participant_id",
            "session",
            "block",
            "trial",
            "onset",
            "duration",
            "time",
            "start_ts",
            pl.col("chunk_length_ts").alias("trial_length_ts"),
            "chunk_margin",
            "dbs_stim",
            "yscore",
            pl.col("^LFP_.*$"),
            pl.col("^ECOG_.*$"),
            "x",
            "y",
            pl.col("labels").alias("tracing_coordinates_present"),
        )

        participants.write_parquet(
            save_path / "participants", partition_by=["participant_id", "session"]
        )
        logger.info(f"Saved to {save_path / 'participants'}")


def _add_ieeg_data(participants: pl.DataFrame, config: Config) -> pl.DataFrame:
    """CODE IF PARTIAL PREPROCESSING IS NOT DONE

    participants = add_modality_path(participants, "ieeg")

    participants = explode_files(participants, "ieeg_path", "ieeg_file")

    participants = split_file_path(
        participants,
        "ieeg",
        [("session", 1, pl.UInt64), ("task", 2, pl.String), ("run", 3, pl.UInt64)],
    )
    participants = keep_rows_with(participants, task="copydraw").drop("task")
    participants = remove_rows_with(participants, type="channels", data_format="tsv")

    events = construct_events_table(participants)

    participants = participants.join(
        events, on=["participant_id", "session", "run"], how="left"
    )

    participants = remove_rows_with(participants, type="events", data_format="tsv")
    headers = keep_rows_with(participants, type="ieeg", data_format="vhdr").select(
        "participant_id",
        "session",
        "run",
        pl.col("ieeg_file").alias("ieeg_headers_file"),
    )

    participants = participants.join(
        headers, on=["participant_id", "session", "run"], how="left"
    )

    participants = remove_rows_with(participants, type="ieeg", data_format="vhdr")
    participants = remove_rows_with(participants, type="ieeg", data_format="vmkr")

    participants = participants.drop(
        "type", "data_format", "channels_info_right", strict=False
    )
    """
    participants = band_pass_resample(participants, config, iEEG_SCHEMA)

    return participants


def _chunk_recordings(
    participants: pl.DataFrame, chunk_margin: int, sfreq: int
) -> pl.DataFrame:

    participants_ = participants.with_columns(
        (pl.col("onset") - chunk_margin).alias("onset"),
        (pl.col("dt_s") + chunk_margin).alias("duration"),
    )

    participants_ = participants_.with_columns(
        (pl.col("onset") * sfreq).cast(pl.UInt32).alias("start_ts"),
        (pl.col("duration") * sfreq).cast(pl.UInt32).alias("chunk_length_ts"),
    ).with_columns(
        pl.int_ranges(0, pl.col("chunk_length_ts"), dtype=pl.UInt32)
        .truediv(sfreq)
        .add(pl.col("onset"))
        .alias("time")
    )

    for ieeg_field in iEEG_SCHEMA.fields:
        if ieeg_field.name == "sfreq":
            continue
        participants_ = participants_.with_columns(
            pl.col(ieeg_field.name).list.slice(
                pl.col("start_ts"), pl.col("chunk_length_ts") + pl.col("start_ts")
            )
        )

    participants_ = participants_.with_columns(
        pl.lit(chunk_margin).alias("chunk_margin")
    )

    return participants_
