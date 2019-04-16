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
        "type": "esim",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 300,
                    "trainable": true
                }
            }
        },
        "encoder": {
            "type": "slstm",
            "hidden_size": 300,
            "num_layers": 7,
        },
        "similarity_function": {
            "type": "dot_product"
        },
        "projection_feedforward": {
            "input_dim": 1200,
            "hidden_dims": 300,
            "num_layers": 1,
            "activations": "relu"
        },
        "inference_encoder": {
            "type": "lstm",
            "input_size": 300,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        },
        "output_feedforward": {
            "input_dim": 2400,
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