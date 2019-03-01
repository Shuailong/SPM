{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path": "./data/snli/snli_1.0_train.jsonl",
    "validation_data_path": "./data/snli/snli_1.0_dev.jsonl",
    "test_data_path": "./data/snli/snli_1.0_test.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "graph_pair",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "pretrained_file": "./data/glove/glove.840B.300d.txt.gz",
                    "trainable": false
                },
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "./data/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "./data/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.5
                },
            }
        },
        "encoder": {
            "hidden_size": (1024 + 300),
            "num_layers": 7,
            "SLSTM_step": 1,
            "dropout": 0.5
        },
        "output_feedforward": {
            "input_dim": (1024 + 300) * 5,
            "num_layers": 1,
            "hidden_dims": 300,
            "activations": "relu",
            "dropout": 0.5
        },
        "output_logit": {
            "input_dim": 300,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear"
        },
        "initializer": [
            [
                ".*linear_layers.*weight",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*linear_layers.*bias",
                {
                    "type": "zero"
                }
            ],
            [
                ".*weight_ih.*",
                {
                    "type": "xavier_uniform"
                }
            ],
            [
                ".*weight_hh.*",
                {
                    "type": "orthogonal"
                }
            ],
            [
                ".*bias_ih.*",
                {
                    "type": "zero"
                }
            ],
            [
                ".*bias_hh.*",
                {
                    "type": "lstm_hidden_bias"
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "premise",
                "num_tokens"
            ],
            [
                "hypothesis",
                "num_tokens"
            ]
        ],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 0.0004
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 2,
        "num_epochs": 75,
        "grad_norm": 10.0,
        "patience": 5,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}