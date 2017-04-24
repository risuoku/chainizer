import chainer
from chainer.training import extensions as training_extensions
import inspect
import pickle
import collections
import os



def get_module_spec(mod):
    def _filter(o):
        if mod is chainer.links:
            return inspect.isclass(o)
        elif mod is chainer.functions:
            return inspect.isfunction(o)
        elif mod is chainer.optimizers:
            return inspect.isclass(o)
        elif mod is training_extensions:
            return inspect.isclass(o)
        else:
            raise TypeError('invalid type!')

    inspected_objects = [
        (k, v)
        for k, v in inspect.getmembers(mod)
        if _filter(v)
    ]
    result = collections.OrderedDict()

    for k, v in inspected_objects:
        sig = inspect.signature(v)
        result[k] = {
            'args': [
                {
                    'name': p.name,
                    'required': p.default is inspect.Parameter.empty
                }
                for p in sig.parameters.values()
            ]
        }
    return result

def main():
    version = chainer.__version__
    spec = {
        'links': get_module_spec(chainer.links),
        'functions': get_module_spec(chainer.functions),
        'optimizers': get_module_spec(chainer.optimizers),
        'training_extensions': get_module_spec(training_extensions),
    }
    
    # create file
    with open(os.path.join('versions', version + '.pickle'), 'wb') as f:
        pickle.dump(spec, f)


if __name__ == '__main__':
    main()
