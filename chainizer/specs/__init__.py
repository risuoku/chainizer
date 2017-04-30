from . import spec
from .proxy import SpecProxy
import sys


default = spec.Spec().build()
user = spec.Spec().build(load = False)

proxy = SpecProxy()
proxy.add(default)
proxy.add(user)
