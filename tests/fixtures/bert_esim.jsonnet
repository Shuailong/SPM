local bert_type = 'base';
local run_env = 'local';

local batch_size_base = 5; //  GPU mem required
local batch_size_large = 20; //  GPU mem required
local feature_size_base = 768;
local feature_size_large = 1024;
local data_root = if run_env == 'local' then 'data' else '/mnt/SPM/data';
local batch_size = if bert_type == 'base' then batch_size_base else batch_size_large;
local feature_size = if bert_type == 'base' then feature_size_base else feature_size_large;


{
  "dataset_reader": {
    "type": "mysnli",
    "mode": "merge",
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
  "train_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
  "validation_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
  "test_data_path": "./tests/fixtures/snli_1.0_sample.jsonl",
  "evaluate_on_test": true,
  "model": {
    "type": "bert-esim",
    "dropout": 0.5,
    "text_field_embedder": {
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
    "encoder": {
      "type": "lstm",
      "input_size": feature_size,
      "hidden_size": 300,
      "num_layers": 1,
      "bidirectional": true
    },
    "similarity_function": {"type": "dot_product"},
    "projection_feedforward": {
      "input_dim": 2400,
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
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size": batch_size
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.0004
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 2,
    "num_epochs": 3,
    "grad_norm": 10.0,
    "patience": 5,
    "cuda_device": -1,
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 0
    }
  }
}
