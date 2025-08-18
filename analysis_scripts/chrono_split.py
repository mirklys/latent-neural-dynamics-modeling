import numpy as np


class ChronoGroupsSplit:
    """Leave out blocks of chronologically sorted pairs"""

    def __init__(self, **kwags):
        for k in kwags:
            print(f"Received kwargs for {k=} which will not be used")

    def split(self, X, y, groups):
        """
        Return the indices of all splits of the data in X and y
        Parameters
        ----------
        X : np.ndarray (nsamples, nfeatures)
            the data array with the sample dimension first
        y : np.ndarray (nsamples, )
            the labels vector
        groups : np.ndarray (nsamples, )
            a grouping vector matching the labels. This will be considered
            to keep groups constant within each fold
        Returns
        -------
        splits : list[tuple[np.ndarray, np.ndarray]]
            a list of splits of (ix_train, ix_test) tuples to loop over
        """

        y = np.asarray(y)
        groups = np.asarray(groups)
        # get groups per label
        gm = {k: list(set(groups[y == k])) for k in set(y)}

        for v in gm.values():
            v.sort()

        # ensure that groups are distinct in labels
        assert all(
            [set(s).intersection(groups[y != k]) == set() for k, s in gm.items()]
        ), " Groups are not unique in label"

        # ensure lengths match
        set_lens = [len(v) for v in gm.values()]
        if not all([e == set_lens[0] for e in set_lens]):
            print(
                "Not all the same number of groups per label - "
                "will zip and thus drop all groups longer than the min"
            )

        Xidcs = np.arange(X.shape[0])

        splits = [
            (
                np.hstack(
                    [
                        Xidcs[groups == list(gv)[j]]
                        for gv in gm.values()
                        for j in range(min(set_lens))
                        if j != i
                    ]
                ),  # the non selected --> train set  # noqa
                np.hstack(
                    [Xidcs[groups == list(gv)[i]] for gv in gm.values()]
                ),  # the selected per grp --> test  # noqa
            )
            for i in range(min(set_lens))
        ]

        return splits
