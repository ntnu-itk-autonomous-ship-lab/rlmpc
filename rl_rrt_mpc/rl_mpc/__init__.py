import platform

uname_result = platform.uname()

if uname_result.machine == "arm64" and uname_result.system == "Darwin":
    ACADOS_COMPATIBLE = False  # ACADOS does not support arm64 and macOS yet
else:
    ACADOS_COMPATIBLE = True
