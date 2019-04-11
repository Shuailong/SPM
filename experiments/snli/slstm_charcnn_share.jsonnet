{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": false
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 3
            }
        }
    },
    "train_data_path": "./data/snli/snli_1.0_train.jsonl",
    "validation_data_path": "./data/snli/snli_1.0_dev.jsonl",
    "test_data_path": "./data/snli/snli_1.0_test.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "slstm_share",
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "./data/glove/glove.840B.300d.txt.gz",
                    "embedding_dim": 300,
                    "trainable": false
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 100
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 100,
                        "num_filters": 300,
                        "ngram_filter_sizes": [3],
                        "conv_layer_activation": "relu"
                    }
                }
            }
        },
        "encoder": {
            "type": "slstm",
            "hidden_size": 600,
            "num_layers": 7,
        },
        "output_feedforward": {
            "input_dim": 600 * 4,
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
        "regularizer": [
            [
                ".*",
                {
                    "type": "l2",
                    "alpha": 1e-03
                }
            ]
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
