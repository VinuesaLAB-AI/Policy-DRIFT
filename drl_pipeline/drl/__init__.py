from dedalus.tools.config import config as _dedalus_config

# Dedalus picks FFTW transform plans by timing them by default, so two runs can use different
# algorithms and differ at round-off level. Estimated plans are deterministic, which makes every
# simulation bitwise reproducible on the same process mesh. This must be set before dedalus.public
# is imported.
_dedalus_config["transforms-fftw"]["PLANNING_RIGOR"] = "estimate"
