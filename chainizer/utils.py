import hashlib
import re


def get_hash(s):
    if isinstance(s, str):
        s = s.encode('utf-8')
    return hashlib.sha256(s).hexdigest()
   

def snake2camel(s):
    return ''.join([a.capitalize() for a in s.split('_')]) # snake case -> camel case


def camel2snake(s):
    return re.sub("([A-Z])",lambda x:"_" + x.group(1).lower(), s)[1:] # camel case -> snake case
