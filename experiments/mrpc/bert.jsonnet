# input by user
local bert_type = 'base';
local run_env = 'docker';
local epochs = 4;
local train_samples = 3668;
local num_labels = 2;
local batch_size_base = 32; // 8g GPU mem required
local batch_size_large = 20; // 16g GPU mem required
local learning_rate = 2e-5;

local random_seed = 22;

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
        "type": "mrpc-bert",
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
                "bert": ["bert", "bert-type-ids"]
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
            "input_dim": feature_size * 2,
            "num_layers": 1,
            "hidden_dims": num_labels,
            "activations": "linear",
        }
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [["s1", "num_tokens"], ["s2", "num_tokens"],],
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
        // "optimizer":{
        //     "type": "adam",
        //     "lr": learning_rate
        // },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": epochs,
        // "grad_norm": 10.0,
        // "patience": 3,
        "cuda_device": 1,
        // "learning_rate_scheduler": {
        //     "type": "reduce_on_plateau",
        //     "factor": 0.5,
        //     "mode": "max",
        //     "patience": 0
        // }
    }
}
