from typing import Optional


def ensure_packages_installed():
    import subprocess

    subprocess.call(
        [
            "pip",
            "install",
            "--trusted-host",
            "pypi.python.org",
            "-r",
            "requirements.txt",
        ]
    )


def set_priority(pid: Optional[int] = None, priority: str = "high"):
    import platform
    from os import getpid
    import psutil

    """Set The Priority of a Process.  Priority is a string which can be 'low', 'below_normal',
    'normal', 'above_normal', 'high', 'realtime'. 'normal' is the default priority."""

    if platform.system() == "Windows":
        priorities = {
            "low": psutil.IDLE_PRIORITY_CLASS,
            "below_normal": psutil.BELOW_NORMAL_PRIORITY_CLASS,
            "normal": psutil.NORMAL_PRIORITY_CLASS,
            "above_normal": psutil.ABOVE_NORMAL_PRIORITY_CLASS,
            "high": psutil.HIGH_PRIORITY_CLASS,
            "realtime": psutil.REALTIME_PRIORITY_CLASS,
        }
    else:  # Linux and other Unix systems
        priorities = {
            "low": 19,
            "below_normal": 10,
            "normal": 0,
            "above_normal": -5,
            "high": -11,
            "realtime": -20,
        }

    if pid is None:
        pid = getpid()
    p = psutil.Process(pid)
    p.nice(priorities[priority])


def initialize_before_launch(install_packages: bool = False):
    """Initialize the app"""
    import platform

    if platform.system() == "Windows":
        set_priority(priority="high")
    else:
        set_priority(priority="normal")
    if install_packages:
        ensure_packages_installed()
