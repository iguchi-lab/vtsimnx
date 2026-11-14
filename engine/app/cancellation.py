"""Typed cancellation shared by the solver runner and asynchronous job manager."""


class SimulationCancelled(RuntimeError):
    """Execution stopped in response to a cancellation request."""
