config = {
    'trainers': [
        {
            'epoch': 10,
            'datasource': {
                'name': 'mnist',
                'minibatch_size': 100,
            },
            'optimizer': {
                'name': 'adam',
            },
            'extensions': [
                {
                    'name': 'Evaluator',
                },
                {
                    'name': 'PrintReport',
                    'params': {
                        'entries': [
                            'epoch', 'main/loss', 'validation/main/loss',
                            'main/accuracy', 'validation/main/accuracy', 'elapsed_time',
                        ],
                    },
                },
                {
                    'name': 'ProgressBar',
                }
            ],
            'model': {
                'name': 'MLP',
                'components': [
                    {
                        'name': 'Linear',
                        'type': 'link',
                        'params': [
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
                        'type': 'function',
                        'params': [
                        ],
                    },
                    {
                        'name': 'Linear',
                        'type': 'link',
                        'params': [
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
                        'type': 'function',
                        'params': [
                        ],
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
