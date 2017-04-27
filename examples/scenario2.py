import chainer
import chainer.links as L
import chainer.functions as F
class CustomizedMLP(chainer.Chain):
    def __init__(self):
        super(CustomizedMLP, self).__init__()
        self.add_link('_Linear0', L.Linear(None,1000))
        self.add_link('_Linear2', L.Linear(None,10))

    def __call__(self, x):
        return F.relu(self._Linear2(F.relu(self._Linear0(x))))


trainer_config = {
    'epoch': 10,
    'datasource': {
        'name': 'mnist',
        'minibatch_size': 100,
    },
    'optimizer': {
        'name': 'Adam',
        'type': 'optimizers',
    },
    'extensions': [
        {
            'name': 'PrintReport',
            'type': 'training_extensions',
            'args': [
                {
                    'name': 'entries',
                    'value': ['epoch', 'main/loss','main/accuracy', 'elapsed_time',]
                },
            ],
        },
        {
            'name': 'LogReport',
            'type': 'training_extensions',
            'args': [
                {
                    'name': 'log_name',
                    'value': None,
                },
            ],
        },
        {
            'name': 'ProgressBar',
            'type': 'training_extensions',
        }
    ],
    'model': {
        'name': 'MLP',
        'customized': CustomizedMLP,
        'use_classifier': True
    },
}

config = {
    'name': 'scenario2',
    'providers': [
        {
            'name': 'mnist',
            'trainers': [
                trainer_config,
            ]
        }
    ]
}

from chainizer.scenario import Scenario
from chainizer.provider import Provider


class MnistProvider(Provider):
    def to_chainerdataset(self):
        return self.data


class MyScenario(Scenario):
    provider_cls = {
        'mnist': MnistProvider
    }

    def load(self):
        import chainer
        return chainer.datasets.get_mnist()


if __name__ == '__main__':
    s = MyScenario(config)
    s.build().run()
