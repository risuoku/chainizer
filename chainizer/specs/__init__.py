import sys

def get_spec():
    import chainer
    import yaml
    import os

    spec = None
    with open(os.path.join(os.path.dirname(__file__), chainer.__version__ + '.yml')) as f:
        spec = yaml.load(f)
    return spec

sys.modules[__name__] = get_spec()
