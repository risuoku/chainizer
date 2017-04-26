import chainizer.settings as settings
from . import converter
import collections


def render(config):
    #result = collections.OrderedDict()
    #for tconfig in config['trainers']:
    #    trainer_tpl = settings.jinja2_env.get_template('trainer.j2')
    #    tconfig['statements'] = converter.convert(tconfig)
    #    result['main'] = trainer_tpl.render(tconfig)
    #return result
    trainer_tpl = settings.jinja2_env.get_template('trainer.j2')
    config['statements'] = converter.convert(config)
    return trainer_tpl.render(config)
