import chainer
import inspect
import yaml
import collections


def get_module_spec(mod):
    def _filter(o):
        if mod.__name__ == 'chainer.links':
            return inspect.isclass(o)
        elif mod.__name__ == 'chainer.functions':
            return inspect.isclass(o) or inspect.isfunction(o)
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
    }
    
    # create file
    with open(version + '.yml', 'w') as f:
        f.write(yaml.dump(spec, default_flow_style=False))


if __name__ == '__main__':
    main()
