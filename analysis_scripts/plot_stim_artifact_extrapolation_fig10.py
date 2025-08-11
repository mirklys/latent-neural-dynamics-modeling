# Plot to explain the de-jittering to approximately calculate when the stim pulse arrives at the tissue

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from paper_plots_aux import apply_default_styles, xdf_to_data_dict


def make_plot():
    # Note that the meta data for AO was using 22kHz although incoming data was
    # at 5500Hz. Also decoder and control had missing data, which for the
    # stat calculation was ignored as anyways only the non dejittered data was
    # considered (actual LSL time stamps).
    d = xdf_to_data_dict("./data/AO_test.xdf")

    # dp = pd.DataFrame(d["decoder_output_ireg"]["x"], columns=["ch1"]).assign(
    #     ts=d["decoder_output_ireg"]["ts"]
    # )
    # dp['dt'] = dp.ts.diff()

    dp = (
        pd.DataFrame(
            d["AODataStream_ireg"]["x"], columns=[f"ch{i}" for i in range(1, 17)]
        )
        .assign(ts=d["AODataStream_ireg"]["ts"])
        .melt(id_vars=["ts"])
    )

    tmin = 83.26
    tmax = 83.29
    dpp = dp[(dp.variable == "ch1") & (dp.ts < tmax) & (dp.ts > tmin)]
    dpp.loc[:, "ts"] -= dpp.ts.min()
    dpp.loc[:, "ts"] *= 1e3
    dpp.loc[:, "value"] *= 1e-3  # to milli volts

    fig = px.scatter(dpp, x="ts", y="value")

    cmap = {
        "arduino": "#636efa",
        "ct": "#00cc96",
        "ao": "#ffa15a",
    }
    fig = fig.update_traces(
        mode="lines+markers",
        line_color="#333",
        marker_color=cmap["ao"],
        marker_size=10,
        name="raw",
        showlegend=True,
    )

    # Add opaque
    dt = 1 / 5500  # assuming stable 5500kHz sampling
    dmod = dpp.copy().assign(ts_mod=dpp.ts).reset_index(drop=True)
    ixend = np.argmax(dmod.ts.diff()) - 1
    dmod.loc[ixend + 1 :, "ts_mod"] = dmod.loc[ixend, "ts"] + np.arange(
        1, dmod.shape[0] - ixend
    ) * (
        dt * 1000
    )  # in ms
    fig2 = px.scatter(dmod.loc[ixend:], x="ts_mod", y="value")
    fig2 = fig2.update_traces(
        mode="lines+markers",
        line_color="#aaa",
        marker=dict(line_color="#aaa", color="rgba(0,0,0,0)", size=12, line_width=2),
        name="isochron",
        showlegend=True,
    )
    fig2 = fig2.add_traces(fig.data)

    fig2 = apply_default_styles(fig2)
    fig2 = fig2.update_yaxes(zeroline=False, title="Voltage [mV]")
    fig2 = fig2.update_xaxes(zeroline=False, range=[0, 18], title="Time [ms]")
    fig2 = fig2.update_layout(
        height=400,
        width=400,
        legend=dict(
            yanchor="top",
            y=0.29,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(255, 255, 255, 0.9)",
        ),
    )

    fig2.show()
