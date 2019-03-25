{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  "train_data_path": "./data/ner/eng.train",
  "validation_data_path": "./data/ner/eng.testa",
  "test_data_path": "./data/ner/eng.testb",
  "evaluate_on_test": true,
  "model": {
    "type": "transformer_crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": 0.5,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 300,
            "trainable": false
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "embedding_dim": 100
          },
          "encoder": {
            "type": "lstm",
            "input_size": 100,
            "hidden_size": 150,
            "num_layers": 1,
            "dropout": 0,
            "bidirectional": true
          }
        }
      },
    },
    "encoder": {
      "num_layers": 7,
      "model_size": 600,
      "inner_size": 600,
      "key_size": 600,
      "value_size": 600,
      "num_head": 6,
      "dropout": 0.1
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure-overall",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 150,
    "grad_norm": 5.0,
    "patience": 25,
    "cuda_device": 0
  }
}
