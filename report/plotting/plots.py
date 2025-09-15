import matplotlib.pyplot as plt


def save_figure(filename, formats=[".png", ".svg"]):
    for fmt in formats:
        plt.savefig(f"{filename}.{fmt}", dpi=300)
        print(f"Figure saved as {filename}.{fmt}")
