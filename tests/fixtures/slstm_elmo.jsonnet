local run_env = 'local';
local mode = 'merge'; // 'sep'

local epochs = 3;
local learning_rate = 4e-4;
local batch_size = 5; // xxg GPU mem required
local elmo_projection_dim = 300;

local hidden_size = if elmo_projection_dim == null then 1024  else elmo_projection_dim;
local data_root = if run_env == 'local' then 'data' else '/mnt/SPM/data';

{
    "dataset_reader": {
        "type": "snli",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "validation_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "test_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
    "evaluate_on_test": true,
    "model": {
        "type": "slstm_classifier",
        "mode": mode,
        "dropout": 0.5,
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": data_root + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": data_root + "/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "projection_dim": elmo_projection_dim
                }
            }
        },
        "encoder": {
            "type": "slstm",
            "hidden_size": hidden_size,
            "num_layers": 7
        },
        "output_feedforward": {
            "input_dim": hidden_size * 4,
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
        "batch_size": batch_size
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            "lr": learning_rate,
            "weight_decay": 1e-2,
            "parameter_groups": [
                [["bias", "LayerNorm.bias", "LayerNorm.weight"], {"weight_decay": 0.0}],
            ]
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": epochs,
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
