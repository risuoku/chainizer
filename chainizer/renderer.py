from . import settings
import jinja2
import chainizer.converter as converter
import collections

env = jinja2.Environment(loader=jinja2.FileSystemLoader(settings.TEMPLATES_PATH, encoding='utf-8'))


def render(config):
    result = collections.OrderedDict()
    for tconfig in config['trainers']:
        trainer_tpl = env.get_template('trainer.j2')
        tconfig['statements'] = converter.convert(tconfig)
        result['main'] = trainer_tpl.render(tconfig)
    return result
