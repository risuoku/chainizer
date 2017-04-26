from chainizer.logging import getLogger

logger = getLogger(__name__)


class Scenario:
    provider_cls = {}

    def __init__(self, config):
        self._config = config
        self._name = None
        self._loaded_data = None
        self._providers = {}

    def build(self):
        self._name = self._config.get('name', __class__.__name__)
        self._loaded_data = self.load()

        self._providers = dict([(pc['name'], self.__class__.provider_cls[pc['name']](pc, self._loaded_data)) for pc in self._config['providers']])
        for name, p in self._providers.items():
            p.build()
            logger.debug('provider {} build done.'.format(name))

        return self

    def run(self):
        for name, p in self._providers.items():
            p.run()
            logger.debug('provider {} run done.'.format(name))
        return self

    def load(self):
        return None
