config = {
    'trainers': [
        {
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
    ]
}


if __name__ == '__main__':
    import chainizer.renderer as renderer
    obj = renderer.render(config)
    print(obj['main'])
