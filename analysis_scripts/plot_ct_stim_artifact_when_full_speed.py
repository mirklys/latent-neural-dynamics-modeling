# The code in this scrip was used to evalute an artifact in the benchtop recordings
# with the CT EvalKit. The artifact was caused by a code segment in the Dareplane
# module for running a while loop at full speed, limiting resources of the same
# process for sending LSL data packages. The plots where used in discussion with
# the second reviewer of the paper.


import pandas as pd
import plotly.express as px
from plot_benchmarking_results import cfgs, xdf_to_data_dict

from paper_plots_aux import apply_default_styles

# ------------- Data for the full speed version ------------------------------
xdf = "./data/CT_pico_loop_full_speed.xdf"

d = xdf_to_data_dict(
    xdf, cfgs=cfgs, tmax_s=500
)  # tmax is only relevant for the sample trace, not for the box plot stats

dff = pd.concat(
    [
        pd.DataFrame(
            {
                "time [s]": d["ct_bic_ireg"]["ts"],
                "x": d["ct_bic_ireg"]["x"][:, 5],
            }
        ).assign(dejitter=False, channel="Ch6"),
        pd.DataFrame(
            {
                "time [s]": d["ct_bic_ireg"]["ts"],
                "x": d["ct_bic_ireg"]["x"][:, 0],
            }
        ).assign(dejitter=False, channel="Ch1"),
    ]
).assign(speed="full")

# ------------- Data for the 1ms sleep version ------------------------------
xdf = "./data/CT_pico_loop_with_1ms_sleep.xdf"

d = xdf_to_data_dict(
    xdf, cfgs=cfgs, tmax_s=500
)  # tmax is only relevant for the sample trace, not for the box plot stats

dfs = pd.concat(
    [
        pd.DataFrame(
            {
                "time [s]": d["ct_bic_ireg"]["ts"],
                "x": d["ct_bic_ireg"]["x"][:, 5],
            }
        ).assign(dejitter=False, channel="Ch6"),
        pd.DataFrame(
            {
                "time [s]": d["ct_bic_ireg"]["ts"],
                "x": d["ct_bic_ireg"]["x"][:, 0],
            }
        ).assign(dejitter=False, channel="Ch1"),
    ]
).assign(speed="1ms_sleep")

# ------------- Plotting -----------------------------------------------------
# select ent of the stimulation period roughly matched
plot_range_full_speed = [292, 306]
plot_range_1ms_sleep = [304, 318]

dffs = dff[
    (dff["time [s]"] > plot_range_full_speed[0])
    & (dff["time [s]"] < plot_range_full_speed[1])
]
dffs.loc[:, "time [s]"] -= dffs["time [s]"].min()

dfss = dfs[
    (dfs["time [s]"] > plot_range_1ms_sleep[0])
    & (dfs["time [s]"] < plot_range_1ms_sleep[1])
]
dfss.loc[:, "time [s]"] -= dfss["time [s]"].min()

dp = pd.concat([dffs, dfss]).rename(columns={"x": "Amplitude [μV]"})

fig = px.line(
    dp,
    x="time [s]",
    y="Amplitude [μV]",
    color="channel",
    facet_col="speed",
    facet_col_wrap=1,
)
fig = apply_default_styles(fig)
fig.show()
