import argparse
import csv
import json
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", type=str, default=None)
parser.add_argument('-t', '--task_name', type=str, default=None)
args = parser.parse_args()
model, task = args.model_name, args.task_name

predictions_path = Path(__file__).parent / 'predictions' / task / f'{model}.json'
data_path = Path(__file__).parent / 'data' / f'{task}.json'

with open(predictions_path) as f:
    predictions = json.load(f)
with open(data_path) as f:
    metadata = json.load(f)['examples']

res = {}
for meta, pred in zip(metadata.values(), predictions.values()):
    if pred['score'] == 1:
        continue
    pred = pred['prediction'].lower()

    for k, v in meta['metadata'].items():
        if not isinstance(v, str):
            continue
        pred = re.sub(fr'\W{v}\W', k, pred)
    res[pred] = res.get(pred, 0) + 1

with open(f'common_patterns_{task}.csv', 'w') as f:
    writer = csv.writer(f)
    for k, v in sorted(res.items(), key=lambda x: -x[1]):
        writer.writerow([k, v])
