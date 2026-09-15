- Question 1:

`pyproject.toml`:
- 4 new lines in `dependencies`: `mlflow>=3.16.0`, `scikit-learn>=1.9.1`, `torch>=2.14.0`, `torchvision>=0.29.0` (uv writes a `>=` on the version it resolved, not a pin).
- the `[[tool.uv.index]]` block declaring the `pytorch-cpu` index (`explicit = true` so nothing else is looked up there by accident).
- `[tool.uv.sources]` mapping only `torch` and `torchvision` to that index.

`uv.lock` went from 2 packages (`mlops-lab-1`, `pillow`) and 64 lines to 102 packages and about 2100 lines. Every transitive dependency of mlflow (flask, sqlalchemy, alembic, pandas, numpy, pyarrow, matplotlib, ...) and torch (sympy, networkx, filelock, fsspec, ...) is now in there with an exact version, the sha256 of every wheel, and the wheel URLs. The torch/torchvision entries are the interesting ones: `source = { registry = "https://download.pytorch.org/whl/cpu" }` and version `2.14.0+cpu` on windows, while everything else has `source = { registry = "https://pypi.org/simple" }`. So the lock records where each package came from, not just which version. Anyone who clones and runs `uv sync` gets the exact same CPU build without having to know about the index trick.

- Question 2:

`--backend-store-uri sqlite:///mlflow.db`: where mlflow keeps the metadata. Experiments, runs, params, metrics (every logged value with its step and timestamp), tags, run status and the artifact URI of each run. Here it's a single sqlite file in the repo root. Without this flag the server would fall back to a file based store under `./mlruns`.

`--default-artifact-root ./mlruns`: where the files go. The logged model (45 MB here), input example, requirements, any image or file you `log_artifact`. One folder per experiment, then per run.

Metadata is small structured data you want to query, sort, filter and plot in the UI, so it belongs in a database. Artifacts are arbitrary blobs that a database is bad at, so they go on a filesystem or object store and the database only keeps the URI pointing at them. In the experiment info the server returned `artifact_location: file:C:/.../mlops-lab-1/mlruns/1`, so since client and server are on the same machine, the training script writes the artifacts straight to that folder itself, only the metadata goes through HTTP.

- Question 3:

git: `mlflow.db` is a binary sqlite file that changes on every single run, so no readable diffs and guaranteed merge conflicts. `mlruns/` gets a ~50 MB model per run, the repo would explode after a few experiments. Both are outputs of running the code, not source, same category as `.venv` or `__pycache__`.

dvc: dvc is for versioned inputs (datasets) that you want tied to a git commit and pullable on another machine. The mlflow store is not one version of one thing, it's an append-only log of many experiments. mlflow is already its own tracking and storage system with its own server, putting its files under dvc would mean two tools versioning the same data with no benefit. In a real setup the tracking server has a proper database (postgres) and an object store (S3) behind it and nothing sits on the laptop anyway.

- Question 4:

Server log when the first run started:

```
GET  /api/2.0/mlflow/experiments/get-by-name?experiment_name=food11  404 Not Found
POST /api/2.0/mlflow/experiments/create                              200 OK
```

So `set_experiment` looks the name up, gets a 404, creates it and then sets it as the active experiment for the process. It got `experiment_id = 1` (Default is `0`) and `artifact_location = file:C:/Users/alial/Desktop/USJ/MLOps/mlops-lab-1/mlruns/1`. In the UI `food11` shows up in the experiments list on the left next to `Default`, empty for a second until the first run appears. Nothing was created on disk at that point, the `mlruns/` folder only appeared later when the first artifact was written. On the next runs `get-by-name` returns 200 and nothing is created.

- Question 5:

`log_param` is a config value fixed before training: lr, batch size, dataset, model name. One value per key per run, stored as a string, and trying to log a different value for the same key in the same run raises an error. `log_metric` is a measured number: loss, accuracy. It can be logged as many times as you want per run, each value keeps a step and a timestamp, and mlflow stores the whole history.

`step` exists because a metric is a time series (val_loss at epoch 1, 2, 3...) and the UI needs an x-axis to draw the chart. A param is a single fact about the run, there is no "lr at epoch 3", so a step would mean nothing. In the runs table mlflow shows the last value of each metric, in the run page you get the full curve.

- Question 6:

Run page: the params table has our 7 keys (dataset, epochs, lr, batch_size, seed, model, optimizer) plus tags mlflow added by itself (git commit, branch, source file, user). The metrics section has one chart per metric with `step` on the x axis for `train_loss`, `val_loss`, `val_accuracy`, and single points for `test_loss` and `test_accuracy`.

The model is not in the run's "Artifacts" tab. `artifacts/list` for the run returns nothing and the folder `mlruns/1/<run_id>/artifacts` does not even exist. mlflow 3 stores the model as a separate "logged model" entity linked to the run, it shows up under the run's models and in the experiment's Models tab. On disk:

```
mlruns/1/models/m-c7dc0c990a224af990fb0189122543b3/artifacts/
    MLmodel                        flavors, signature, run_id, model_id
    data/model.pt2                 45 MB, the actual weights + graph
    conda.yaml, python_env.yaml, requirements.txt
    input_example.json, serving_input_example.json
```

So `--default-artifact-root ./mlruns` gave `mlruns/1/` for experiment 1, and every model goes in `models/m-<id>/artifacts/` under it. After 5 runs `mlruns/` is 241 MB and `mlflow.db` is under 1 MB, which is the metadata vs artifacts split from question 2.

One thing that did not work as written in the lab: `mlflow.pytorch.log_model(model, "model")` raises in mlflow 3.16. The default serialization format is now `pt2` (torch.export) and it needs an `input_example` to trace the graph. Passing 2 images from the test loader fixed it and also gives the model a signature (`float32 [-1, 3, 128, 128]` in, `[-1, 11]` out).

- Question 7:

The 4 comparison runs plus the first dev run, sorted by `val_accuracy` (last epoch):

| run | lr | batch_size | val_accuracy | test_accuracy |
|---|---|---|---|---|
| sedate-bat-474 | 0.0001 | 32 | 0.736 | 0.778 |
| carefree-flea-515 | 0.001 | 64 | 0.612 | 0.669 |
| bustling-boar-347 | 0.001 | 32 | 0.494 | 0.514 |
| adorable-sloth-913 | 0.001 | 32 | 0.494 | 0.514 |
| bald-bass-792 | 0.01 | 32 | 0.151 | 0.143 |

Best learning rate: 0.0001. Higher is not better, it's the opposite here. lr 0.01 destroyed the pretrained weights in the first epoch (val_loss 44.4 after epoch 1) and never recovered, 0.15 accuracy on 11 classes is barely above random (0.09). lr 0.001 learns but val_accuracy bounces around (0.41, 0.54, 0.46, 0.51, 0.49) while train_loss keeps going down, so the steps are too big and it overfits. lr 0.0001 climbs smoothly every epoch and has the lowest val_loss by far. Makes sense for fine tuning: the resnet weights are already good, we want small steps that adjust them, not big ones that erase them. It's not "lower is always better" either, at 0.00001 five epochs would not be enough to move the new fc layer.

The two lr 0.001 / bs 32 runs are identical to the 4th decimal because the seed is fixed (`--seed 0`) and training is on CPU, so a run is fully reproducible.

- Question 8:

On the parallel coordinates plot with axes `lr`, `batch_size`, `val_accuracy`: the lines fan out on the `lr` axis and land on `val_accuracy` in the same order, lr 0.01 at the bottom, 0.001 in the middle, 0.0001 at the top. `lr` is what decides the result. On the `batch_size` axis the only comparison is the two lines at lr 0.001: the one going through 64 ends higher (0.61) than the one going through 32 (0.49). Bigger batch means half as many updates per epoch and each gradient averaged over more images, so it behaves a bit like a smaller lr, which is the direction that helped. Only 4 configs so it is a weak signal for batch_size, but the lr trend is clear.

- Question 9:

Sorted by `val_accuracy` descending the top run is `sedate-bat-474` (lr 0.0001, batch_size 32).

```
run ID:   008f0ee5bb574c9eb4622a04d03a515e
model ID: m-d37a5600782e431bb817d8bfb487c758
val_accuracy 0.7363, test_accuracy 0.7783
```

Model URI for later: `runs:/008f0ee5bb574c9eb4622a04d03a515e/model` or `models:/m-d37a5600782e431bb817d8bfb487c758`.
