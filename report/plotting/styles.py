import matplotlib.pyplot as plt
from cycler import cycler

thesis_colors = {
    "light_blue": "#56B4E9",
    "pink": "#CC79A7",
    "orange": "#D55E00",
    "teal": "#009E73",
    "grey": "#565656",
}

thesis_cycler = cycler(
    color=[
        thesis_colors["light_blue"],
        thesis_colors["pink"],
        thesis_colors["orange"],
        thesis_colors["teal"],
        thesis_colors["grey"],
    ]
)


def set_thesis_style():
    plt.rc("axes", prop_cycle=thesis_cycler)

    plt.rc("figure", autolayout=True, dpi=300, figsize=(6, 4))

    plt.rc("image", cmap="coolwarm")

    plt.rc("font", family="sans-serif", sans_serif=["Helvetica", "Arial"], size=10)

    plt.rc(
        "axes",
        axisbelow=True,
        edgecolor=thesis_colors["grey"],
        grid=False,
        labelcolor=thesis_colors["grey"],
        labelsize="medium",
        linewidth=1.0,
        titlepad=10.0,
        titlesize="large",
        titleweight="bold",
    )

    plt.rc("axes.spines", top=False, right=False)

    plt.rc("xtick", color=thesis_colors["grey"], direction="out", labelsize="small")
    plt.rc("ytick", color=thesis_colors["grey"], direction="out", labelsize="small")

    plt.rc("lines", linewidth=1.5, markersize=6)

    plt.rc("legend", frameon=False, fontsize="small")
    plt.rc("savefig", bbox="tight", dpi=300, format="png", transparent=True)
