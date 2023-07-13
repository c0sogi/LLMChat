import importlib
from types import ModuleType


class ModuleReloader:
    def __init__(self, module_name: str):
        self._module_name = module_name

    def reload(self) -> ModuleType:
        importlib.reload(self.module)
        return self.module

    @property
    def module(self) -> ModuleType:
        return importlib.import_module(self._module_name)
