import sys
import subprocess
import os.path
try:
    import importlib.resources as importlib_resources
except (ModuleNotFoundError, ImportError):
    import importlib_resources


_fname = "manual.pdf"


def start_docs():
    with importlib_resources.path(__name__, _fname) as res:
        if sys.platform == "win32":
            os.startfile(os.path.normpath(res))
        elif sys.platform == "darwin":
            subprocess.run(['open', res], check=True)
        # fall back to xdg-open for all the unix-likes
        else:
            subprocess.run(['xdg-open', res], check=True)
