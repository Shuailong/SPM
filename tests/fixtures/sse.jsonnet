{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        }
    },
    "train_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "validation_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "test_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "sse",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": false
                }
            }
        },
        "encoder1": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 512,
            "num_layers": 1,
            "bidirectional": true
        },
        "encoder2": {
            "type": "lstm",
            "input_size": 300 + 512 * 2,
            "hidden_size": 1024,
            "num_layers": 1,
            "bidirectional": true
        },
        "encoder3": {
            "type": "lstm",
            "input_size": 300 + 512 * 2 + 1024 * 2,
            "hidden_size": 2048,
            "num_layers": 1,
            "bidirectional": true
        },
        "output_feedforward1": {
            "input_dim": 2048 * 2 * 4,
            "num_layers": 1,
            "hidden_dims": 1600,
            "activations": "relu",
            "dropout": 0.1
        },
        "output_feedforward2": {
            "input_dim": 1600,
            "num_layers": 1,
            "hidden_dims": 1600,
            "activations": "relu",
            "dropout": 0.1
        },
        "output_logit": {
            "input_dim": 1600,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear"
        },
        "initializer": [
            [".*linear_layers.*weight", {"type": "xavier_uniform"}],
            [".*linear_layers.*bias", {"type": "zero"}],
            [".*weight_ih.*", {"type": "xavier_uniform"}],
            [".*weight_hh.*", {"type": "orthogonal"}],
            [".*bias_ih.*", {"type": "zero"}],
            [".*bias_hh.*", {"type": "lstm_hidden_bias"}]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
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
