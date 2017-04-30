from .exceptions import ValidationFailed
from .spec import Spec
import traceback


class SpecProxy:
    def __init__(self):
        self._specs = []

    def add(self, spec):
        if not isinstance(spec, Spec):
            raise TypeError('`spec` must be Spec instance.')
        self._specs.append(spec)

    def __getattr__(self, name):
        def _func(*args, **kwargs):
            errors = []
            for sp in self._specs:
                try:
                    return getattr(sp, name)(*args, **kwargs)
                except Exception as e:
                    errors.append(traceback.format_exc())
            for error in errors:
                print(error)
            raise Exception('{} failed!'.format(name))
        return _func

    def __getitem__(self, key):
        return self.__getattr__('__getitem__')(key)
