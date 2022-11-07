# LMentry

This repository contains the LMentry benchmark from [LMentry: A Language Model Benchmark of Elementary Language Tasks](https://arxiv.org/pdf/2211.02069.pdf), as well as the code to evaluate it.

For any questions, feel free to open a GitHub issue or to contact us at avia.efrat@gmail.com :blush:

## Getting the Data
Simply clone the repo: 
```shell
git clone https://github.com/aviaefrat/lmentry.git
```
The data is in the `data` directory.

## Generating Predictions
We provide functions for generating predictions with Hugging Face and OpenAI models (see below), but you can generate predictions in any method of your choosing.

For Hugging Face and OpenAI models, you can use the 
`generate_all_hf_predictions` and 
`generate_all_openai_predictions` functions from `predict.py`. These are what we used in our experiments.

## Evaluation

The easiest and recommended way is to use `evalutate.py`:
```shell
python -m lmentry.evaluate
```
_Don't forget to activate the lmentry environment (created from `environment.yml`) beforehand._  
Using the `--num-procs=N` optional argument will score the predictions much faster.  
`evalutate.py` will also automatically create files analyzing the results in a separate `results` dir.

To use `evalutate.py`, the predictions must follow the same structure of [lmentry_predictions.zip](https://drive.google.com/file/d/1TXToBnXwz22CGtg6wjDE81gFPtVufHi8/view?usp=sharing) (if you used our functions from `predict.py`, your predictions already follow this structure):
1. The top-level directory should be named `predictions`.
2. `predictions` needs to contain exactly 41 directories, named after the 41 files in `data` (the 25 task names + the 16 files for the argument content robustness).
3. Each of the 41 task predictions directories should contain a prediction file for each model you want to evaluate. For example, to evaluate the predictions of a model named `my-model`, each of the 41 directories should contain a file named `my-model.json` with the model's predictions for this task.
4. Each predictions file should contain values in the form `"<id>": {"prediction": <prediction>},` where the `id`s correspond to those in the task's file in `data`.

## Reproducing the Results from the Paper
1. Clone the repository.
2. Unzip `lmentry_predictions.zip` into the top-level lmentry directory.
3. run `evaluate.py` (preferably with a not-very-small value for `--num-procs`, as there are 656 files to score...)