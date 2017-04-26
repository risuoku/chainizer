import os
import jinja2


BASE_PATH = os.path.dirname(__file__)

TEMPLATES_PATH = os.path.join(BASE_PATH, 'templates')

jinja2_env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES_PATH, encoding='utf-8'))
