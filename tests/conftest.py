import os

# Persistent compilation cache (09): shared by all test processes.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", ".jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
