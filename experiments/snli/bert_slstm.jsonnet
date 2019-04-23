local bert_type = 'base';
local run_env = 'docker';
local pretrain_model_name = 'xxx';

local batch_size_base = 32; // 12 GPU mem required
local batch_size_large = 32; // 14 GPU mem required
local feature_size_base = 768;
local feature_size_large = 1024;
local data_root = if run_env == 'local' then 'data' else '/mnt/SPM/data';
local model_root = if run_env == 'local' then 'models' else '/mnt/SPM/models';
local batch_size = if bert_type == 'base' then batch_size_base else batch_size_large;
local feature_size = if bert_type == 'base' then feature_size_base else feature_size_large;

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
                "type": "bert-pretrained",
                "pretrained_model": data_root + "/bert/bert-"+bert_type+"-uncased-vocab.txt"
            }
        }
    },
    "train_data_path": data_root + "/snli/snli_1.0_train.jsonl",
    "validation_data_path": data_root + "/snli/snli_1.0_dev.jsonl",
    "test_data_path": data_root + "/snli/snli_1.0_test.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "bert_with_encoder_classifier",
        "aggregation": "cls+max",
        "bert": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets", "bert-type-ids"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": data_root + "/bert/bert-"+bert_type+"-uncased.tar.gz",
                    "requires_grad": false,
                    "top_layer_only": true
                }
            }
        },
        "encoder":{
            "type": "slstm",
            "hidden_size": feature_size,
            "num_layers": 7
        },
        "classifier": {
            "input_dim": feature_size * 2,
            "num_layers": 1,
            "hidden_dims": 3,
            "activations": "linear",
        },
        "initializer": [
            [".*_bert.*|.*_pooler.*",
                {
                    "type": "pretrained",
                    "weights_file_path": model_root + '/' + pretrain_model_name + '/best.th',
                }
            ]
        ]

    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["tokens", "num_tokens"]],
        "batch_size": batch_size
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": 2e-5
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "grad_norm": 10.0,
        "cuda_device": 0,
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "mode": "max",
            "patience": 0
        }
    }
}
