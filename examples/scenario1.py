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
        'components': [
            {
                'name': 'Linear',
                'type': 'links',
                'args': [
                    {
                        'name': 'in_size',
                        'value': None,
                    },
                    {
                        'name': 'out_size',
                        'value': 1000,
                    },
                ],
            },
            {
                'name': 'relu',
                'type': 'functions',
            },
            {
                'name': 'Linear',
                'type': 'links',
                'args': [
                    {
                        'name': 'in_size',
                        'value': None,
                    },
                    {
                        'name': 'out_size',
                        'value': 10,
                    },
                ],
            },
            {
                'name': 'relu',
                'type': 'functions',
            },
        ],
        'use_classifier': True
    },
}

config = {
    'name': 'scenario1',
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
