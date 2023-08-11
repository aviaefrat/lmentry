# Table of contents
1. [LMentry](#lmentry)
2. [Environment adjustment](#environment-adjustment)
3. [Prediction](#prediction)
4. [Evaluation](#evaluation)
5. [README from paper and original github repo authors](#readme-from-paper-and-original-github-repo-authors)

# LMentry

This repository is a fork from the original one which contains the LMentry benchmark from [LMentry: A Language Model Benchmark of Elementary Language Tasks](https://arxiv.org/pdf/2211.02069.pdf), as well as the code to evaluate it.
There was refactoring of the code to support manipulation with external models (like one from HuggingFace or [mlc-llm](https://github.com/mlc-ai/mlc-llm) side). The benchmark consists of two parts: 1. **prediction** or collection of statistics during inference of specified model on the benchmark task datasets; 2. **evaluation** or processing of obtained statistics and calculation final result (accuracy, robustness and LMentry score).

# Environment adjustment
All calculations occurs in conda environment. Unfortunately the original environment becomes old enough and it is needed to update some packeges:
```bash
conda env create -f environment.yml
conda activate lmentry
python3 -m pip install transformers scipy accelerate tokenizers==0.13.3
```

and additionally for support mlc-llm dependencies (do it under lmentry env):
```bash
conda install -c conda-forge libstdcxx-ng=12
python3 -m pip install decorator attrs
```

# Prediction
If we want to benchmark model processed in [mlc-llm](https://github.com/mlc-ai/mlc-llm) the path to root directory with model shared library (so-file) and dependencies is simply needed:
```bash
python3 predict_model.py -m <path_to_mlc-llm>/dist/vicuna-7b-v1.3-q0f16
```

# Evaluation
The evaluation script processes statistics collected during the prediction stage and output values of accuracy, robustness and LMentry score averaged and for each task formed in csv-files. Based on number of processed task (total number is 41) and number of models it can require much time for evaluation. Due to this fact it is recommended to set as many <num-procs> as possible to accelerate the process.
```bash
python3 lmentry/evaluate.py --num-procs <num-procs>
```

# README from paper and original github repo authors
Due to the reason that the original repo do not allows to work with any other LLM from the box excluding ones from the paper it was forked and strongly refactored. But with regards to the authors of the initial work the text from their README is stayed here.

## LMentry

This repository contains the LMentry benchmark from [LMentry: A Language Model Benchmark of Elementary Language Tasks](https://arxiv.org/pdf/2211.02069.pdf), as well as the code to evaluate it.

For any questions, feel free to open a GitHub issue or to contact us at avia.efrat@gmail.com :blush:

### Getting the Data
Simply clone the repo: 
```shell
git clone https://github.com/aviaefrat/lmentry.git
```
The data is in the `data` directory.

### Generating Predictions
We provide functions for generating predictions with Hugging Face and OpenAI models (see below), but you can generate predictions in any method of your choosing.

For Hugging Face and OpenAI models, you can use the 
`generate_all_hf_predictions` and 
`generate_all_openai_predictions` functions from `predict.py`. These are what we used in our experiments.

### Evaluation

The easiest and recommended way is to use `evalutate.py`:
```shell
python -m lmentry.evaluate
```
_Don't forget to activate the lmentry environment (created from `environment.yml`) beforehand._  
Using the `--num-procs=N` optional argument will score the predictions much faster.  
`evalutate.py` will also automatically create files analyzing the results in a separate `results` dir.

To use `evalutate.py`, the predictions must follow the same structure of [lmentry_predictions.zip](https://drive.google.com/file/d/1Ex1fde_PEzhIU5ctGkOJsacaGNnQeqke/view?usp=sharing) (if you used our functions from `predict.py`, your predictions already follow this structure):
1. The top-level directory should be named `predictions`.
2. `predictions` needs to contain exactly 41 directories, named after the 41 files in `data` (the 25 task names + the 16 files for the argument content robustness).
3. Each of the 41 task predictions directories should contain a prediction file for each model you want to evaluate. For example, to evaluate the predictions of a model named `my-model`, each of the 41 directories should contain a file named `my-model.json` with the model's predictions for this task.
4. Each predictions file should contain values in the form `"<id>": {"prediction": <prediction>},` where the `id`s correspond to those in the task's file in `data`.

### Reproducing the Results from the Paper
1. Clone the repository.
2. Unzip `lmentry_predictions.zip` into the top-level lmentry directory.
3. run `evaluate.py` (preferably with a not-very-small value for `--num-procs`, as there are 656 files to score...)
