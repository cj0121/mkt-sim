"""Utilities package

Auto-import all submodules by default so functions are available as utils.*.
To opt-out a module from auto-import, set a module-level variable
`__utils_autoload__ = False` in that module.
"""

import importlib
import pkgutil
from typing import List

__all__: List[str] = []

def _autoload_submodules():
    pkg = __name__
    for modinfo in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        name = modinfo.name
        # skip private modules
        if name.startswith("_"):
            continue
        module = importlib.import_module(f"{pkg}.{name}")
        # Check opt-out flag
        if getattr(module, "__utils_autoload__", True):
            # import * from the module namespace
            for attr in getattr(module, "__all__", []) or dir(module):
                if attr.startswith("_"):
                    continue
                globals()[attr] = getattr(module, attr)
                __all__.append(attr)

_autoload_submodules()


