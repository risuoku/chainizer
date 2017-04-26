import sys
import chainer
import pickle
import os


class ValidationFailed(Exception):
    pass


class Spec:
    def __init__(self):
        self._value = None

    def build(self):
        with open(os.path.join(os.path.dirname(__file__), 'versions', chainer.__version__ + '.pickle'), 'rb') as f:
            self._value = pickle.load(f)
        return self

    def __getitem__(self, key):
        return self._value[key]

    def keys(self):
        return self._value.keys()

    def validate(self, obj):
        # objはdict型
        if not isinstance(obj, dict):
            raise ValidationFailed('object type must be dict')

        # keyとして`name`, `type` が存在する。
        if ('name' not in obj) or ('type' not in obj):
            raise ValidationFailed('`name` and `type` must exist')

        # `name`,`type` が仕様として有効なものかどうか確認。
        if obj['type'] not in self._value:
            raise ValidationFailed('invalid module type')

        if obj['name'] not in self[obj['type']]:
            raise ValidationFailed('invalid component name')

        # keyとして`args` が存在するか確認。無ければ 空リストとして付与する。
        if 'args' not in obj:
            obj['args'] = []

        # `args`はlist型
        if not isinstance(obj['args'], list):
            raise ValidationFailed('`args` type must be list')

        # keyとして`name`,`value`が存在する。同時に`args` が仕様として有効なものかどうか確認。
        spec_args = self[obj['type']][obj['name']]['args']
        required_args_names = set([o['name'] for o in spec_args if o['required']])
        all_args_names = set([o['name'] for o in spec_args])

        for arg in obj['args']:
            if ('name' not in arg) or ('value' not in arg):
                raise ValidationFailed('`name` and `value` must exist')
            if arg['name'] not in all_args_names:
                raise ValidationFailed('invalid args name')
            if arg['name'] in required_args_names:
                required_args_names.remove(arg['name'])
        
        if (not len(required_args_names) == 0) and (not obj['type'] == 'functions'): # functionsはrequiredを無視する
            raise ValidationFailed('required arg not set')
