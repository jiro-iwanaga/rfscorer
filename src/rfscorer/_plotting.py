"""Plotting mixin for RecencyFrequencyScorer.

Provides 2D surface and 1D marginal visualization. Not intended for
standalone use; relies on attributes set by RecencyFrequencyScorer.fit()
and optimize().
"""

import numpy as np


class PlottingMixin:
    """Visualization methods for RecencyFrequencyScorer.

    Required attributes (provided by RecencyFrequencyScorer):
        _normalize_kind, emp_probability_table_, mono_probability_table_,
        mrc_probability_table_, mfc_probability_table_, mcc_probability_table_,
        er_probability_, ef_probability_, mr_probability_, mf_probability_.
    """

    _SURFACE_TITLES = {
        "emp": "Empirical",
        "mono": "Monotonic",
        "mrc": "Monotonic (Recency Convex)",
        "mfc": "Monotonic (Frequency Concave)",
        "mcc": "Monotonic (Convex-Concave)",
    }

    _MARGINAL_TITLES = {
        "er": "Empirical Recency",
        "ef": "Empirical Frequency",
        "mr": "Monotonic Recency (Convex)",
        "mf": "Monotonic Frequency (Concave)",
        "rboth": "Recency: Empirical & Monotonic",
        "fboth": "Frequency: Empirical & Monotonic",
    }

    # ---------------------------------------------------------------------------
    # Surface (2D)
    # ---------------------------------------------------------------------------

    def plot_probability_surface(
        self,
        kind="emp",
        title=None,
        figsize=(6, 5),
        fontsize=12,
        recency_label="recency",
        frequency_label="frequency",
        probability_label="probability",
        path=None,
    ):
        """Plot product-choice probabilities as a 3D surface.

        Visualizes the probability table as a 3D wireframe with recency on
        the x-axis, frequency on the y-axis, and probability on the z-axis.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        To save to a file, call ``fig.savefig("output.png")`` on the returned
        figure or use the ``path`` parameter.
        Japanese axis labels require ``pip install rfscorer[ja]``
        (installs ``japanize-matplotlib``).

        Parameters
        ----------
        kind : {"emp", "mono", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to visualize. "emp" uses fit() results; others use optimize() results.
            Note: "mr", "mf", "er", and "ef" are 1D marginal models and cannot
            be visualized as a surface; use plot_marginal_probability() instead.
        title : str or None, default None
            Figure title. When None, a default title based on kind is shown
            (e.g. "Empirical", "Monotonic (Recency Convex)"). Pass ``""`` to
            suppress the title entirely.
        figsize : tuple[float, float], default (6, 5)
            Figure size in inches as (width, height). For publication, set this
            to the final printed size (e.g., (3.5, 3.0) for a single-column
            figure in a two-column journal).
        fontsize : int, default 12
            Font size for axis labels and tick labels. For publication, match
            this to the body text size of the target journal (typically 8–10 pt)
            and set figsize to the final printed size so the font is not scaled.
        recency_label : str, default "recency"
            Label for the x-axis (recency dimension).
        frequency_label : str, default "frequency"
            Label for the y-axis (frequency dimension).
        probability_label : str, default "probability"
            Label for the z-axis (probability dimension).
        path : str or None, default None
            Save destination. When None, the figure is not saved.
            When a directory path, saves as ``surface_{kind}_probability.png``
            in that directory. When a file path, saves with that name.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If kind is "mr", "mf", "er", or "ef" (1D marginal models cannot
            be plotted as a surface), or if kind is not one of the accepted
            values.
        RuntimeError
            If fit() has not been called for
            kind="emp", or if optimize(kind=...) has not been called for
            the requested optimization kind.
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        kind = self._normalize_kind(kind)
        if kind in ("mr", "mf", "er", "ef"):
            raise ValueError(
                f"kind={kind!r} is a 1D marginal model and cannot be plotted as a surface."
                " Use plot_marginal_probability() instead."
            )
        if kind not in ("emp", "mono", "mrc", "mfc", "mcc"):
            raise ValueError(f"kind must be 'emp', 'mono', 'mrc', 'mfc', or 'mcc', got {kind!r}.")
        if kind == "emp" and self.emp_probability_table_ is None:
            raise RuntimeError("fit() must be called before plot_probability_surface().")
        if kind == "mono" and self.mono_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mono') must be called before plot_probability_surface(kind='mono')."
            )
        if kind == "mrc" and self.mrc_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mrc') must be called before plot_probability_surface(kind='mrc')."
            )
        if kind == "mfc" and self.mfc_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mfc') must be called before plot_probability_surface(kind='mfc')."
            )
        if kind == "mcc" and self.mcc_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mcc') must be called before plot_probability_surface(kind='mcc')."
            )

        if kind == "emp":
            table = self.emp_probability_table_
        elif kind == "mono":
            table = self.mono_probability_table_
        elif kind == "mrc":
            table = self.mrc_probability_table_
        elif kind == "mfc":
            table = self.mfc_probability_table_
        else:
            table = self.mcc_probability_table_

        recency = table.index.tolist()
        frequency = table.columns.tolist()
        X, Y = np.meshgrid(recency, frequency)
        Z = table.values.T

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(X, Y, Z, color="black")
        ax.set_xlabel(recency_label, fontsize=fontsize)
        ax.set_ylabel(frequency_label, fontsize=fontsize)
        ax.set_zlabel(probability_label, fontsize=fontsize, labelpad=10)
        ax.tick_params(labelsize=fontsize)
        # recency と frequency は整数次元なので、自動目盛りが小数刻み (例: 2.5) を
        # 選ばないよう整数ロケータを指定する。
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
        effective_title = self._SURFACE_TITLES[kind] if title is None else title
        if effective_title:
            ax.set_title(effective_title, fontsize=fontsize)
        if path is not None:
            import os

            default_name = f"surface_{kind}_probability.png"
            filepath = os.path.join(path, default_name) if os.path.isdir(path) else path
            fig.savefig(filepath)
        return fig

    # ---------------------------------------------------------------------------
    # Marginal (1D)
    # ---------------------------------------------------------------------------

    def plot_marginal_probability(
        self,
        kind="er",
        title=None,
        figsize=(5, 4),
        fontsize=12,
        axis_label=None,
        probability_label="probability",
        path=None,
    ):
        """Plot product-choice probability along one RF dimension as a line chart.

        The axis (recency or frequency) is inferred from ``kind``:
        "er"/"mr"/"rboth" plot against recency; "ef"/"mf"/"fboth" plot against
        frequency.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        Japanese axis labels require ``pip install rfscorer[ja]``
        (installs ``japanize-matplotlib``).

        Parameters
        ----------
        kind : {"er", "ef", "mr", "mf", "rboth", "fboth"}, default "er"
            Which probability series to draw.
            "er"    draws the empirical recency marginal only.
            "ef"    draws the empirical frequency marginal only.
            "mr"    draws the monotonic recency series only (monotonicity + convexity).
            "mf"    draws the monotonic frequency series only (monotonicity + concavity).
            "rboth" overlays "er" and "mr" on the recency axis (requires optimize(kind='mr')).
            "fboth" overlays "ef" and "mf" on the frequency axis (requires optimize(kind='mf')).
        title : str or None, default None
            Figure title. When None, a default title based on kind is shown
            (e.g. "Empirical Recency", "Monotonic Recency (Convex)"). Pass
            ``""`` to suppress the title entirely.
        figsize : tuple[float, float], default (5, 4)
            Figure size in inches as (width, height).
        fontsize : int, default 12
            Font size for axis labels and tick labels.
        axis_label : str or None, default None
            Label for the x-axis. When None, defaults to "recency" for recency
            plots (kind in "er"/"mr"/"rboth") and "frequency" for frequency plots
            (kind in "ef"/"mf"/"fboth").
        probability_label : str, default "probability"
            Label for the y-axis.
        path : str or None, default None
            Save destination. When None, the figure is not saved.
            When a directory path, saves as ``marginal_{kind}_probability.png``
            in that directory. When a file path, saves with that name.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If kind is not one of the accepted values.
        RuntimeError
            If fit() has not been called, or if optimize(kind='mr') has not
            been called when kind is "mr" or "rboth", or if optimize(kind='mf')
            has not been called when kind is "mf" or "fboth".

        Notes
        -----
        Line styles: for "rboth"/"fboth" the empirical series is solid and the
        optimized series is dashed; single-series plots are solid. All black.
        """
        import matplotlib.pyplot as plt

        kind = self._normalize_kind(kind)
        valid_kinds = ("er", "ef", "mr", "mf", "rboth", "fboth")
        if kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {kind!r}.")
        if self.er_probability_ is None:
            raise RuntimeError("fit() must be called before plot_marginal_probability().")
        if kind in ("mr", "rboth") and self.mr_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mr') must be called before"
                " plot_marginal_probability(kind='mr'/'rboth')."
            )
        if kind in ("mf", "fboth") and self.mf_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mf') must be called before"
                " plot_marginal_probability(kind='mf'/'fboth')."
            )

        if kind in ("er", "mr", "rboth"):
            x_col = "recency"
            x_label = axis_label if axis_label is not None else "recency"
            df_emp = self.er_probability_
        else:
            x_col = "frequency"
            x_label = axis_label if axis_label is not None else "frequency"
            df_emp = self.ef_probability_

        if kind in ("mr", "rboth"):
            df_opt = self.mr_probability_[[x_col, "probability"]].reset_index(drop=True)
        elif kind in ("mf", "fboth"):
            df_opt = self.mf_probability_[[x_col, "probability"]].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=figsize)
        if kind in ("er", "ef", "rboth", "fboth"):
            emp_label = "er" if kind in ("er", "rboth") else "ef"
            ax.plot(
                df_emp[x_col],
                df_emp["probability"],
                color="black",
                linestyle="-",
                marker="o",
                linewidth=1.5,
                markersize=6,
                label=emp_label,
            )
        if kind in ("mr", "mf", "rboth", "fboth"):
            opt_label = "mr" if kind in ("mr", "rboth") else "mf"
            ax.plot(
                df_opt[x_col],
                df_opt["probability"],
                color="black",
                linestyle="--" if kind in ("rboth", "fboth") else "-",
                marker="s",
                linewidth=1.5,
                markersize=6,
                label=opt_label,
            )
        if kind in ("rboth", "fboth"):
            ax.legend(fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(probability_label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        effective_title = self._MARGINAL_TITLES[kind] if title is None else title
        if effective_title:
            ax.set_title(effective_title, fontsize=fontsize)
        fig.tight_layout()
        if path is not None:
            import os

            default_name = f"marginal_{kind}_probability.png"
            filepath = os.path.join(path, default_name) if os.path.isdir(path) else path
            fig.savefig(filepath)
        return fig
