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


from chainizer import specs
def convert_model(config):
    def validate_component(c):
        def validate_required(c, argspec):
            required_argspec_names = [a['name'] for a in argspec if a['required']]
            if not len(set(required_argspec_names) - set([a['name'] for a in c['params']])) == 0:
                raise Exception('not enough arguments!')
    
        if c['type'] == 'link':
            validate_required(c, specs['links'][c['name']]['args'])
        elif c['type'] == 'function':
            # TODO
            pass
        else:
            raise ValueError('invalid type!')

    def generate_link_instance_name(link_name, idx):
        return '_{}{}'.format(link_name, str(idx))

    def render_link(o):
        return 'L.{}({})'.format(o['name'], ','.join([str(a['value']) for a in o['params']]))
    
    def render_function(o):
        return 'F.{}'.format(o['name'])

    def render_component_on_call(idx, o):
        if o['type'] == 'link':
            return 'self.{}'.format(generate_link_instance_name(o['name'], idx))
        elif o['type'] == 'function':
            return render_function(o)
        else:
            raise ValueError('unsupported type!')

    # validate components
    for c in config['components']:
        validate_component(c)
    
    # add link index
    for i, l in enumerate(config['components']):
        if l['type'] == 'link':
            config['components'][i]['_index'] = i

    init_statements = [
        'self.add_link({}, {})'.format(_stringify(generate_link_instance_name(l['name'], l['_index'])), render_link(l))
        for l in config['components']
        if l['type'] == 'link'
    ]

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
