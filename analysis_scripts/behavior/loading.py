from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fire import Fire
from rich.progress import track
from xileh import xPData

from shared.io import load_config
from shared.logging import logger

TO_NUMPY = [
    "cursor_t",
    "scaling_matrix",
    "trace_let",
    "traces_pix",
    "template",
    "template_pix",
    "template_pos",
    "template_size",
]


def load_copydraw_record_yaml(pth: Path) -> dict:
    """Load a single recording"""
    data = yaml.safe_load(open(pth, "r"))

    for k, v in data.items():
        if k in TO_NUMPY:
            try:
                data[k] = np.asarray(v)
            except ValueError as e:
                # COMMENT RAISE HERE TO USE INCOMPLETE
                raise ValueError(
                    "Potentially fragmented trace, "
                    f"Error converting {k} to numpy array: {e}"
                )

    return data


def deriv_and_norm(var, delta_t):
    """
    Given an array (var) and timestep (delta_t), computes the derivative
    for each timepoint and returns it (along with the magnitudes)

    """
    deriv_var = np.diff(var, axis=0) / delta_t
    deriv_var_norm = np.linalg.norm(deriv_var, axis=1)
    return deriv_var, deriv_var_norm


def derive_stim(fpath: Path) -> str:
    """For a given session dir get the stim value for a given block"""
    if fpath.stem.startswith("STIM_OFF_"):
        return "off"
    elif fpath.stem.startswith("STIM_ON_"):
        return "on"
    elif fpath.stem.startswith("STIM_UNKN"):
        if str(fpath.parent).endswith("_cl"):
            return "on"
        else:
            return "unknown"
    else:
        logger.warning(f"Cannot derive stim state from {fpath.stem=}")
        return "unknown"


def load_trial_data(trial_file: Path, use_longest_only: bool = False) -> dict:
    res = load_copydraw_record_yaml(trial_file)
    res["stim"] = derive_stim(trial_file)

    # scale the template to how it would be on the screen in real pixels
    # as the trace_let is recorded in screen pixel coords
    temp = res["template_pix"] * res["template_scaling"]
    scaled_template = temp - (
        res["template_pos"] / res["scaling_matrix"][0, 0] / res["template_scaling"]
    )

    res["scaled_template"] = scaled_template

    res["concat_scaled_traces"] = np.vstack(res["traces_pix"])

    return res


def load_trial_traces(pdata: xPData, save: bool = True) -> xPData:
    trial_yamls = list(
        pdata.config.data["data_root"]
        .joinpath(pdata.config.data["session"], "behavioral")
        .rglob("*block*trial*.yaml")
    )

    data = []
    for trial_yaml in track(trial_yamls, description="Loading trial yamls..."):
        try:
            data.append(load_trial_data(trial_yaml))
        except ValueError as e:
            print(
                "Encountered ValueErro, potentially fragmented trace for "
                f"{trial_yaml.stem}: {e}"
            )

    df = pd.DataFrame(data)

    traces_files = pdata.config.data["data_root"].joinpath(
        pdata.config.data["session"], "processed", "traces.hdf"
    )
    if save:
        traces_files.parent.mkdir(exist_ok=True, parents=True)
        df.to_hdf(traces_files, key="traces")

    pdata.add(df, name="traces", header={"files": trial_yamls})
    return pdata


def main(
    session: str = "",
    data_root: Path = Path("../../../data/"),
    name_filter: str = ".*_copydraw_",
):
    """An example reading ecog data from mat files"""

    pdata = xPData(
        [
            xPData(
                load_config(
                    session=session,
                    data_root=data_root,
                    name_filter=name_filter,
                    tstart=-0.5,
                    tstop=10,
                ),
                name="config",
            )
        ],
        name="report_container",
    )

    pdata = load_trial_traces(pdata)


if __name__ == "__main__":
    session = ""
    data_root = Path("../../../data/")
    name_filter = ".*_copydraw_"
    Fire(main)
