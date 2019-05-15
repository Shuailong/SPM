# env
local run_env = 'local';
local cuda_device = 0;

# model
local bert_type = 'base';
local num_labels = 2;

# train
local random_seed = 22;
local epochs = 5;
local train_samples =  3668; // 17804; //3668;
local batch_size_base = 32; // 8g GPU mem required
local batch_size_large = 20; // 16g GPU mem required
local learning_rate = 2e-5;

# dependent vars
local feature_size_base = 768;
local feature_size_large = 1024;
local data_root = if run_env == 'local' then 'data' else '/mnt/SPM/data';
local batch_size = if bert_type == 'base' then batch_size_base else batch_size_large;
local feature_size = if bert_type == 'base' then feature_size_base else feature_size_large;


{
    "random_seed": random_seed,
    "numpy_seed": random_seed,
    "pytorch_seed": random_seed,
    "dataset_reader": {
        "type": "mrpc",
        "tokenizer": {
            "type": "word",
            "word_splitter": {
                "type": "just_spaces"
            }
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": data_root + "/bert/bert-"+bert_type+"-uncased-vocab.txt"
            }
        },
        "skip_label_indexing": true,
        "mode": "merge"
    },
    "train_data_path": data_root + "/glue/MRPC/train.tsv",
    "validation_data_path": data_root + "/glue/MRPC/dev.tsv",
    "test_data_path": data_root + "/glue/MRPC/msr_paraphrase_test.txt",
    "evaluate_on_test": true,
    "model": {
        "type": "bert_sequence_classifier",
        "metrics": ["f1"],
        "bert": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets", "bert-type-ids"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": data_root + "/bert/bert-"+bert_type+"-uncased.tar.gz",
                    "requires_grad": true,
                    "top_layer_only": true
                }
            }
        },
        "num_labels": num_labels,
        "classifier": {
            "input_dim": feature_size,
            "num_layers": 1,
            "hidden_dims": num_labels,
            "activations": "linear",
        },
        "initializer": [
            ["_bert.*|_classifier.*|pooler.*",
                {
                    "type": "pretrained",
                    "weights_file_path": "models/mrpc/20190513-bert-base-snli-train/best.th",
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
            "type": "bert_adam",
            "lr": learning_rate,
            "t_total": train_samples/batch_size*epochs,
            "warmup": 0.1,
            "weight_decay": 1e-2,
            "parameter_groups": [
                [["bias", "LayerNorm.bias", "LayerNorm.weight"], {"weight_decay": 0.0}],
            ]
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": epochs,
        "cuda_device": cuda_device,
    }
}
