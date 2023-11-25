import psutil, toml, os
from importlib.metadata import version

if psutil.Process(os.getpid()).ppid() == 1:
    __startmode__ = "systemd"
    __appname__ = "guck4"
    __appabbr__ = "g4"
    __version__ = version(__appname__)
else:
    __startmode__ = "dev"
    with open("pyproject.toml", mode="r") as config:
        toml_file = toml.load(config)
    __version__ = toml_file["tool"]["poetry"]["version"]
    __appname__ = "guck" + __version__.split(".")[0]
    __appabbr__ = "g" + __version__.split(".")[0]

from .utils import *



