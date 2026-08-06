"""Plotting and log-analysis helpers."""

__all__ = [
    "plot_metric_vs_time",
]


def __getattr__(name: str):
    if name == "plot_metric_vs_time":
        from asteroid_rl.analysis.plotting import plot_metric_vs_time

        return plot_metric_vs_time
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
