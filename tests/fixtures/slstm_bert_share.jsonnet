{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained-sl",
                "pretrained_model": "data/bert/vocab.txt",
                "start_tokens": [],
                "end_tokens": []
            }
        }
    },
    "train_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "validation_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "test_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "slstm_share",
        "dropout": 0.5,
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "data/bert/bert-base-uncased.tar.gz",
                    "requires_grad": false
                }
            }
        },
        "encoder": {
            "type": "slstm",
            "hidden_size": 768,
            "num_layers": 7,
        },
        "output_feedforward": {
            "input_dim": 768 * 4,
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
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["premise", "num_tokens"],
                         ["hypothesis", "num_tokens"]],
        "batch_size": 16
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
