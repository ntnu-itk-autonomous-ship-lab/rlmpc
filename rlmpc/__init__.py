"""RLMPC package initialization."""

import os
import pathlib

# Ensure ACADOS_SOURCE_DIR is set to expanded path before importing acados
# This fixes issues where $HOME is not expanded in the environment variable
if "ACADOS_SOURCE_DIR" in os.environ:
    acados_path = os.environ["ACADOS_SOURCE_DIR"]
    # Expand $HOME if present as literal string
    if "$HOME" in acados_path:
        acados_path = acados_path.replace("$HOME", os.path.expanduser("~"))
        os.environ["ACADOS_SOURCE_DIR"] = acados_path
else:
    # Set default if not set
    acados_default = os.path.expanduser("~/acados")
    if pathlib.Path(acados_default).exists():
        os.environ["ACADOS_SOURCE_DIR"] = acados_default

# Fix macOS library loading for acados
# acados libraries use @rpath but don't have rpath entries, so we need to set DYLD_FALLBACK_LIBRARY_PATH
if os.environ.get("ACADOS_SOURCE_DIR"):
    acados_lib_path = os.path.join(os.environ["ACADOS_SOURCE_DIR"], "lib")
    if pathlib.Path(acados_lib_path).exists():
        # Set DYLD_FALLBACK_LIBRARY_PATH for macOS to find acados dependencies
        current_fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if acados_lib_path not in current_fallback:
            if current_fallback:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{acados_lib_path}:{current_fallback}"
            else:
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = acados_lib_path
        # Also ensure DYLD_LIBRARY_PATH is set (for compatibility)
        current_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        if acados_lib_path not in current_dyld:
            if current_dyld:
                os.environ["DYLD_LIBRARY_PATH"] = f"{acados_lib_path}:{current_dyld}"
            else:
                os.environ["DYLD_LIBRARY_PATH"] = acados_lib_path

