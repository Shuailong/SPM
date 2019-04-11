{
    "dataset_reader": {
        "type": "snli-bert",
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "bert-basic"
            }
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained-sl",
                "pretrained_model": "data/bert/vocab.txt",
            }
        }
    },
    "train_data_path": "./data/snli/snli_1.0_train.jsonl",
    "validation_data_path": "./data/snli/snli_1.0_dev.jsonl",
    "test_data_path": "./data/snli/snli_1.0_test.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "bert_snli",
        "aggregation": "CLS",
        "dropout": 0.1,
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets", "bert-type-ids"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "data/bert/bert-base-uncased.tar.gz",
                    "requires_grad": true
                }
            }
        },
        "output_logit": {
            "input_dim": 768,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear",
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["sentence_pair", "num_tokens"]],
        "batch_size": 32
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 4e-5
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
