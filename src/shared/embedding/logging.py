"""
Lightweight logger facade that integrates with Comet ML when available, else no-ops.
"""

import os
try:
    from comet_ml import Experiment
except Exception:
    Experiment = None  # soft dependency to avoid hard fail when comet-ml is not installed

class Noop:
    """
    No-operation logger that matches the Comet Experiment API used by this project.
    """

    def log_parameters(self, *args, **kwargs) -> None:
        """
        Accept parameter logs and intentionally do nothing.
        """
        pass

    def log_metrics(self, *args, **kwargs) -> None:
        """
        Accept metric logs and intentionally do nothing.
        """
        pass

    def log_other(self, *args, **kwargs) -> None:
        """
        Accept miscellaneous logs and intentionally do nothing.
        """
        pass

def make_experiment(project_name: str):
    """
    Create a Comet Experiment when COMET_API_KEY is set; otherwise return a no-op logger.

    Args:
        project_name (str): The Comet project name to group runs under.

    Returns:
        Experiment | Noop: A configured Comet Experiment or a no-op stub if Comet is unavailable.
    """
    api = os.getenv("COMET_API_KEY")
    if not api or Experiment is None:
        return Noop()
    return Experiment(project_name=project_name, auto_output_logging="simple")
