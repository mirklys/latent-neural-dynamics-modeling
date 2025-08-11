import matplotlib.pyplot as plt

try:
    import numpy as np
    from dtw import dtw

    def dtw_features(trace, template, keep_internals: bool = True, **kwargs):
        res = {}
        # alignment = dtw(trace[:, 1], template[:, 1], keep_internals=keep_internals, **kwargs)
        alignment = dtw(trace, template, keep_internals=keep_internals, **kwargs)

        # For debugging
        # alignment.plot(type='threeway')
        # alignment.plot(type='twoway')

        ns = trace.shape[0]  # n_samples
        nt = template.shape[0]
        C = alignment.localCostMatrix
        D = alignment.costMatrix
        idx_min = np.argmin(C[-1, 1:]) if ns >= nt else np.argmin(C[1:, -1])

        res["w"] = np.stack([alignment.index2, alignment.index1], axis=1)

        # bug in matlab code here, chooses wrong template axis
        res["pathlen"] = min([idx_min, template.shape[0]])
        res["dt"] = alignment.distance

        d_l = D[-1, idx_min] if ns >= nt else D[idx_min, -1]
        res["dt_l"] = d_l

        return res

    def plot_warp_results(trace: np.ndarray, temp: np.ndarray):
        """Check how the dtw combined the traces"""
        dtw_res = dtw_features(trace, temp)

        fig, ax = plt.subplots()
        ax.plot(*temp.T, label="template")
        ax.plot(*trace.T, label="trace")
        ax.legend()

        # plot the point pairs
        for i1, i2 in dtw_res["w"][: dtw_res["pathlen"]]:
            ax.plot(
                [trace[i2, 0], temp[i1, 0]],
                [trace[i2, 1], temp[i1, 1]],
                linestyle="dotted",
                color="#555555",
                alpha=0.2,
            )

except ImportError:

    def dtw_features(trace, template, keep_internals: bool = True, **kwargs):
        raise ImportError("Please install the dtw package to use this function")
