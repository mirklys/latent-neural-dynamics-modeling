def set_polars_config():
    import polars as pl

    print("Loading Polars configuration from utils/package_configs.py")
    pl.Config.set_tbl_width_chars(-1)
    pl.Config.set_tbl_rows(-1)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_fmt_str_lengths(2000)
    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")


def logger_executor(log_dir: str, name: str):
    from .logger import setup_logger

    setup_logger(log_dir, name=name)
