from jinja2 import Template
import importlib
import collections

mod_self = importlib.import_module(__name__)


def _render(statements, config):
    if not isinstance(statements, list):
        raise TypeError('statements must be list')
    return [Template(stmt).render(config) for stmt in statements]


def _stringify(*s):
    return '\'{}\''.format(''.join(s))


def convert_datasource(config):
    if config['name'] == 'mnist':
        statements = [
            'train, test = chainer.datasets.get_mnist()',
            'train_iter = chainer.iterators.SerialIterator(train, {{ minibatch_size }})',
            'test_iter = chainer.iterators.SerialIterator(test, {{ minibatch_size }})',
        ]
        return _render(statements, config)
    else:
        raise ValueError('unsupported name!')


def convert_optimizer(config):
    if config['name'] == 'adam':
        statements = [
            'optimizer = chainer.optimizers.Adam()'
        ]
        return _render(statements, config)
    else:
        raise ValueError('unsupported name!')


def convert_model(config):
    def render_link(o):
        if o['name'] == 'layer':
            return Template('L.Linear({{ n_in }}, {{ n_out }})').render(o['params'])
        else:
            raise ValueError('unsupported name!')
    
    def render_function(o):
        if o['name'] == 'relu':
            return Template('F.relu').render(o['params'])
        else:
            raise ValueError('unsupported name!')

    def render_component_on_call(idx, o):
        if o['type'] == 'link':
            return 'self.{}{}'.format(o['name'], idx)
        elif o['type'] == 'function':
            return render_function(o)
        else:
            raise ValueError('unsupported type!')
    
    # add link index
    for i, l in enumerate(config['components']):
        if l['type'] == 'link':
            config['components'][i]['_index'] = i

    init_statements = ['self.add_link({}, {})'.format(_stringify(l['name'], str(l['_index'])), render_link(l)) for l in config['components'] if l['type'] == 'link']

    # build call statements
    h = 'x'
    for i, c in enumerate(config['components']):
        h = '{}({})'.format(render_component_on_call(i, c), h)
    h = 'return ' + h
    call_statements = [h]

    return {
        'init': init_statements,
        'call': call_statements,
    }


def convert(config):
    result = collections.OrderedDict()
    for name, local_config in config.items():
        convert_func = getattr(mod_self, 'convert_' + name, None)
        if convert_func is not None:
            result[name] = convert_func(local_config)
    return result
