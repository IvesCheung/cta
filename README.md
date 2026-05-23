# SemanticCTA Final Directory Guide

This directory contains the current runnable CTA pipeline, including LoRA training, dataset conversion, optional profiling generation, and the supporting `llm_tool` package.

The main files are:

- `train_lora.py`: training entry point.
- `config.py`: task definitions and CLI arguments.
- `data.py`: REVEAL-style data loading and dataset construction.
- `modeling.py`: LoRA backbone and classifier head setup.
- `engine.py`: training, evaluation, and prefix or suffix forward logic.
- `tokenization.py`: prompt templates and tokenization helpers.
- `losses.py`: Focal Loss, SupCon Loss, and Early Stopping.
- `profilling.py`: uses an LLM to generate column descriptions or column-relation descriptions.
- `convert_to_cta.py`: reconstructs single-table CSV files from column-level data and generates CTA training files.
- `count_tokens.py`: counts tokens in a JSON file.
- `llm_tool/`: reusable LLM calls, embedding clients, prompt templates, and helper utilities.

## 1. Environment Setup and Dataset Preparation

It is recommended to create a Python environment from the repository root and install dependencies there. If you already have a working environment for the main repository, you can reuse it.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The current training scripts use a REVEAL-style dataset layout. The processed public datasets for all 6 benchmarks can be downloaded from the following Hugging Face page:

https://huggingface.co/datasets/Tommy-DING/table-column-annotation-benchmark

Download the dataset files from that page and place them under your local dataset directory, for example `./datasets/cta/`. The training commands below assume that your downloaded data is available under `--data_path`.

If you prefer using the Hugging Face CLI, a typical workflow is:

```powershell
huggingface-cli download Tommy-DING/table-column-annotation-benchmark --repo-type dataset --local-dir ./datasets/cta
```

Special thanks to the REVEAL project for organizing and releasing these datasets publicly. That work makes CTA reproduction and follow-up experiments much easier.

After downloading, you will usually get data in a column-level format like this:

```text
table_id, column_index, label, data
```

The training script currently supports the following task names:

- `gt-semtab22-dbpedia-all`
- `gt-semtab22-schema-property-all`
- `sotab`
- `turl`

These names map directly to the task definitions in `config.py`. The value passed to `--task` must be one of them.

## 2. Optional Dataset Conversion and Profiling

If your dataset is still in column-level format and you want to restore it into single-table CSV files plus fold files, run `convert_to_cta.py` first. This is useful for later inspection, visualization, and optional profiling generation.

Example:

```powershell
py -3 final/convert_to_cta.py --input_dir ./datasets/sota-cpa --output_dir ./datasets/sota-cpa-cta --workers 8
```

This script does two things:

- Reconstructs one CSV file per table from the column-level rows.
- Generates `fold_0.csv` to `fold_4.csv` from the training split.

If you want extra table descriptions or column relation descriptions, you can then run `profilling.py`. This step is optional. It is not required for training, but it is useful if you want description augmentation, semantic inspection, or extra inputs for downstream experiments.

Before running `profilling.py`, configure the environment variables required by `llm_tool`. The repository no longer contains hardcoded API keys or private endpoints, so all credentials and endpoints must be provided explicitly.

Example LLM chat configuration:

```powershell
$env:SEMANTICCTA_LLM_API_KEYS="key1,key2"
$env:SEMANTICCTA_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
$env:SEMANTICCTA_LLM_KEY_MODE="round_robin"
```

Example embedding API configuration:

```powershell
$env:SEMANTICCTA_EMBED_URL_QWEN3_06B="https://your-endpoint/v1/embeddings"
$env:SEMANTICCTA_EMBED_AUTH_QWEN3_06B="Bearer token_a,Bearer token_b"
$env:SEMANTICCTA_EMBED_URL_QWEN3_4B="https://your-endpoint/v1/embeddings"
$env:SEMANTICCTA_EMBED_AUTH_QWEN3_4B="Bearer token_c"
```

If you use a local embedding model, you can also set:

```powershell
$env:SEMANTICCTA_EMBED_USE_LOCAL="true"
$env:SEMANTICCTA_EMBED_DEVICE="cuda:0"
$env:SEMANTICCTA_EMBED_LOCAL_QWEN3_06B="./model/qwen3-0.6B-embedding"
```

Then run profiling:

```powershell
py -3 final/profilling.py --root_dir ./datasets/sota-cpa-cta --output_file ./output/prof.json --prompt_version multi_column --sample_size 64 --max_workers 8 --model qwen3-0.6B
```

The most commonly used `profilling.py` arguments are:

- `--root_dir`: directory containing the single-table CSV files.
- `--output_file`: output profiling JSON path.
- `--prompt_version`: prompt template version, default is `multi_column`.
- `--sample_size`: maximum number of sampled rows per table.
- `--max_workers`: number of worker threads.
- `--model`: override the default LLM model name.
- `--save_interval`: how often to save intermediate results.

If you want to estimate the token cost of a profiling JSON file, run:

```powershell
py -3 final/count_tokens.py ./output/prof.json --model ./model/qwen3-0.6B-embedding --verbose
```

## 3. Running the Training Code

The training entry point is `train_lora.py`:

```powershell
py -3 final/train_lora.py --help
```

A typical training command looks like this:

```powershell
py -3 final/train_lora.py ^
  --model_path ./model/qwen3-0.6B ^
  --task sotab ^
  --data_path ./datasets/cta ^
  --result_dir ./results/cta_v2/sotab ^
  --gpu_id 0
```

For GitTables DBpedia, for example:

```powershell
py -3 final/train_lora.py ^
  --model_path ./model/qwen3-0.6B ^
  --task gt-semtab22-dbpedia-all ^
  --data_path ./datasets/cta ^
  --result_dir ./results/cta_v2/gt_dbpedia ^
  --gpu_id 0
```

All model examples in this README use `qwen3-0.6B` as the default backbone. If you use another local model, replace the value passed to `--model_path` or `--model` accordingly.

The most important training arguments are:

- `--model_path`: backbone model path.
- `--task`: task name, which must match a definition in `config.py`.
- `--data_path`: dataset root directory.
- `--result_dir`: output directory for results.
- `--gpu_id`: GPU index to use.
- `--prefix_mode`: `none` means one full prompt per column; `full` means a shared table prefix.
- `--prefix_context_width`: in `full` mode, how many neighboring columns are visible to each target column. `0` means no neighbors, `-1` means the full table.
- `--repr_layers`: hidden-state layers used for representation extraction, supports negative indexing such as `-4,-8`.
- `--repr_pool`: pooling strategy for column representations, either `mean` or `last`.
- `--repr_l2_norm`: whether to apply L2 normalization before classification.
- `--head_type`: classifier head type, one of `linear`, `cosine`, `mlp`, `ln_mlp`, or `res_mlp`.
- `--none_batch_size`: column-level batch size when `prefix_mode=none`.
- `--max_column_chars`: character clipping limit for column text, used to avoid overlong inputs.
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA hyperparameters.
- `--num_unfrozen_layers`: number of backbone layers to unfreeze in addition to LoRA.
- `--learning_rate`, `--num_epochs`, `--grad_accum_steps`: core optimization settings.
- `--contrastive_weight`, `--contrastive_temperature`: SupCon-related settings.
- `--save_model`: whether to save the best checkpoint.

After training, the output directory usually contains:

- `config.json`
- `results.json`
- optional best-model checkpoint files

## 4. Recommended Workflow

If you want to run a full experiment from scratch, the recommended order is:

1. Set up the Python environment and install dependencies.
2. Download the REVEAL-processed datasets from Hugging Face.
3. If the data is still column-level, use `convert_to_cta.py` to reconstruct single-table CSV files and fold files.
4. If you need description augmentation, configure your API keys and endpoints, then run `profilling.py` to generate a description file. This step is optional.
5. Run `train_lora.py` for training and evaluation.
