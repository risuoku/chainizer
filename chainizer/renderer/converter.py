from jinja2 import Template
import importlib
import collections
from chainizer.specs import default as default_spec

mod_self = importlib.import_module(__name__)


def _render(statements, config):
    if not isinstance(statements, list):
        raise TypeError('statements must be list')
    return [Template(stmt).render(config) for stmt in statements]


def _stringify(*s):
    return '\'{}\''.format(''.join(s))

def _args_str(o):
    return ','.join([str(a['value']) for a in o['args']])


def convert_datasource(config):
    statements = [
        'train_iter = chainer.iterators.SerialIterator(train_data, {{ minibatch_size }})',
        'test_iter = chainer.iterators.SerialIterator(test_data, {{ minibatch_size }})',
    ]
    return _render(statements, config)


def convert_optimizer(config):
    default_spec.validate(config)
    statements = [
        'optimizer = chainer.optimizers.{}({})'.format(config['name'], _args_str(config))
    ]
    return _render(statements, config)


def convert_extensions(config):
    for idx, c in enumerate(config):
        default_spec.validate(config[idx])
    statements = [
        'trainer.extend(extensions.{}({}))'.format(c['name'], _args_str(c))
        for c in config
    ]
    return statements


def convert_model(config):
    def generate_link_instance_name(link_name, idx):
        return '_{}{}'.format(link_name, str(idx))

    def render_link(o):
        return 'L.{}({})'.format(o['name'], _args_str(o))
    
    def render_function(o):
        return 'F.{}'.format(o['name'])

    def render_component_on_call(idx, o):
        if o['type'] == 'links':
            return 'self.{}'.format(generate_link_instance_name(o['name'], idx))
        elif o['type'] == 'functions':
            return render_function(o)
        else:
            raise ValueError('unsupported type!')

    # validate components
    for c in config['components']:
        default_spec.validate(c)
    
    # add link index
    for i, l in enumerate(config['components']):
        if l['type'] == 'links':
            config['components'][i]['_index'] = i

    init_statements = [
        'self.add_link({}, {})'.format(_stringify(generate_link_instance_name(l['name'], l['_index'])), render_link(l))
        for l in config['components']
        if l['type'] == 'links'
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
