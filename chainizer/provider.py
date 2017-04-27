from chainizer.renderer import render as chainizer_render
from chainizer.logging import getLogger

import sys

logger = getLogger(__name__)


class Provider:
    def __init__(self, config, data):
        self._config = config
        self._data = data
        self._trainers = {}

    def build(self):
        global train_data
        global test_data
        train_data, test_data = self.to_chainerdataset()
        for idx, trc in enumerate(self._config['trainers']):
            if 'customized' in trc['model']:
                setattr(sys.modules[__name__], trc['model']['name'],  trc['model']['customized'])
            exec(chainizer_render(trc), globals())
            self._trainers[self._config.get('name', 'trainer{}'.format(idx))] = get_trainer()
        return self

    def run(self):
        for name, tr in self._trainers.items():
            logger.debug('trainer {} run start.'.format(name))
            tr.run()
            logger.debug('trainer {} run done.'.format(name))
        return self

    @property
    def trainers(self):
        return self._trainers
    
    @property
    def data(self):
        return self._data

    def to_chainerdataset(self):
        raise NotImplementedError()
