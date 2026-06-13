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
        mrc_probability_table_, mfc_probability_table_,
        mcc_probability_table_, recency_probability_, frequency_probability_,
        mr_probability_, mf_probability_.
    """

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
    ):
        """Plot product-choice probabilities as a 3D surface.

        Visualizes the probability table as a 3D wireframe with recency on
        the x-axis, frequency on the y-axis, and probability on the z-axis.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        To save to a file, call ``fig.savefig("output.png")`` on the returned
        figure.

        Parameters
        ----------
        kind : {"emp", "mono", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to visualize. "emp" uses fit() results; others use optimize() results.
            Note: "mr", "mf", "er", and "ef" are 1D marginal models and cannot
            be visualized as a surface; use plot_marginal_probability() instead.
        title : str or None, default None
            Figure title. If None, no title is shown.
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
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        return fig

    # ---------------------------------------------------------------------------
    # Marginal (1D)
    # ---------------------------------------------------------------------------

    def plot_marginal_probability(
        self,
        axis="recency",
        kind="emp",
        title=None,
        figsize=(5, 4),
        fontsize=12,
        recency_label="recency",
        frequency_label="frequency",
        probability_label="probability",
    ):
        """Plot product-choice probability aggregated along one RF dimension.

        When axis='recency', plots the empirical recency-marginal probability
        (aggregated over all frequency levels) and optionally the mr-optimized
        probability against recency rank.  When axis='frequency', plots the
        empirical frequency-marginal probability and optionally the mf-optimized
        probability against frequency.  Both are shown as line charts with markers.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        To save to a file, call ``fig.savefig("output.png")`` on the returned figure.

        Parameters
        ----------
        axis : {"recency", "frequency"}, default "recency"
            Which dimension to aggregate and plot.
            "recency" plots probability vs recency rank (expected: decreasing).
            "frequency" plots probability vs frequency (expected: increasing).
        kind : {"emp", "er", "ef", "mr", "mf", "all"}, default "emp"
            Which probability series to draw.
            "emp" draws the empirical marginal (R2Prob or F2Prob).
            "er" draws the empirical recency marginal (valid only when
            axis="recency"). Equivalent to "emp" on the recency axis.
            "ef" draws the empirical frequency marginal (valid only when
            axis="frequency"). Equivalent to "emp" on the frequency axis.
            "mr" draws the mr-optimized series (valid only when axis="recency").
            "mf" draws the mf-optimized series (valid only when axis="frequency").
            "all" draws both the empirical and the optimized series together.
        title : str or None, default None
            Figure title. If None, no title is shown.
        figsize : tuple[float, float], default (5, 4)
            Figure size in inches as (width, height). For publication, set this
            to the final printed size (e.g., (3.5, 2.8) for a single-column
            figure in a two-column journal).
        fontsize : int, default 12
            Font size for axis labels and tick labels. For publication, match
            this to the body text size of the target journal (typically 8–10 pt)
            and set figsize to the final printed size so the font is not scaled.
        recency_label : str, default "recency"
            Label for the x-axis when axis="recency".
        frequency_label : str, default "frequency"
            Label for the x-axis when axis="frequency".
        probability_label : str, default "probability"
            Label for the y-axis (probability dimension).

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If axis is not "recency" or "frequency", if kind is not one of the
            accepted values, or if the axis/kind combination is invalid (e.g.,
            kind="mf" or "ef" with axis="recency").
        RuntimeError
            If fit() has not been called, or if
            optimize(kind='mr') / optimize(kind='mf') has not been called when
            kind is "mr", "mf", or "all".
        """
        import matplotlib.pyplot as plt

        kind = self._normalize_kind(kind)
        if axis not in ("recency", "frequency"):
            raise ValueError(f"axis must be 'recency' or 'frequency', got {axis!r}.")
        valid_kinds = ("emp", "er", "ef", "mr", "mf", "all")
        if kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {kind!r}.")
        if axis == "recency" and kind in ("mf", "ef"):
            raise ValueError(f"kind={kind!r} is not valid when axis='recency'. Use 'mr' or 'er'.")
        if axis == "frequency" and kind in ("mr", "er"):
            raise ValueError(f"kind={kind!r} is not valid when axis='frequency'. Use 'mf' or 'ef'.")
        if self.recency_probability_ is None:
            raise RuntimeError("fit() must be called before plot_marginal_probability().")
        opt_kind = "mr" if axis == "recency" else "mf"
        if kind in (opt_kind, "all"):
            opt_attr = f"{opt_kind}_probability_"
            if getattr(self, opt_attr) is None:
                raise RuntimeError(
                    f"optimize(kind='{opt_kind}') must be called before"
                    f" plot_marginal_probability(kind='{kind}')."
                )

        x_col = axis
        x_label = recency_label if axis == "recency" else frequency_label
        if axis == "recency":
            df_emp = self.recency_probability_
        else:
            df_emp = self.frequency_probability_

        if kind in (opt_kind, "all"):
            df_opt = getattr(self, f"{opt_kind}_probability_")[[x_col, "probability"]].reset_index(
                drop=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        if kind in ("emp", "er", "ef", "all"):
            ax.plot(
                df_emp[x_col],
                df_emp["probability"],
                color="black",
                linestyle="-",
                marker="o",
                linewidth=1.5,
                markersize=6,
                label="emp" if kind == "all" else kind,
            )
        if kind in (opt_kind, "all"):
            ax.plot(
                df_opt[x_col],
                df_opt["probability"],
                color="black",
                linestyle="--" if kind == "all" else "-",
                marker="s",
                linewidth=1.5,
                markersize=6,
                label=opt_kind,
            )
        if kind == "all":
            ax.legend(fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(probability_label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        fig.tight_layout()
        return fig
