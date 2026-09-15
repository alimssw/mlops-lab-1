# Lab 2 - Model training and experiment tracking with MLflow

This lab continues the project started in Lab 1. You now have a git+dvc repo with the raw and processed Food-11 datasets tracked. In this lab you will write the training code, run a local MLflow tracking server, log parameters and metrics for each training run, and compare several runs in the MLflow UI.

> What you need to know:
> - mlflow organises tracking into *experiments* (a named group of runs, e.g. "food11") and *runs* (one training execution with its own params, metrics and artifacts)
> - a tracking server stores this metadata and serves the UI; without one, mlflow just writes to a local ./mlruns folder
> - a *param* is a value set before training and fixed for the run (learning rate, batch size, model architecture...); a *metric* is a value produced during or after training that can evolve over time (loss, accuracy...)
> - autologging can capture most of this automatically for common frameworks, but logging explicitly gives you control over exactly what gets recorded and when

## Environment Setup

### Install mlflow and the training libraries

```bash
uv add mlflow torch torchvision scikit-learn
```

> By default `torch`/`torchvision` install the CUDA-enabled build, which is a multi-GB download you don't need if your machine has no NVIDIA GPU (most laptops, all Macs). The mini dataset in this lab trains fine on CPU. To get the much smaller CPU-only wheels instead, add this to `pyproject.toml` **before** running `uv add`:
>
> ```toml
> [[tool.uv.index]]
> name = "pytorch-cpu"
> url = "https://download.pytorch.org/whl/cpu"
> explicit = true
>
> [tool.uv.sources]
> torch = { index = "pytorch-cpu" }
> torchvision = { index = "pytorch-cpu" }
> ```
>
> Then run the `uv add` command above as usual. If you do have an NVIDIA GPU and want CUDA acceleration, skip this and let uv install the default build.

> Question 1: Look at pyproject.toml and uv.lock. What changed?

### Run a local mlflow tracking server

In its own terminal, from the root of your repo:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Leave this running and open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. You should see an empty "Default" experiment.

> Question 2: What is `--backend-store-uri` used for? What is `--default-artifact-root` used for? What is the difference between the metadata mlflow stores and the artifacts it stores?

Since `mlflow.db` and `mlruns/` are local run outputs, not code or versioned data, keep them out of git and dvc:

```bash
echo "mlflow.db" >> .gitignore
echo "mlruns/" >> .gitignore
git add .gitignore
git commit -m "Ignore local mlflow tracking files"
git push
```

> Question 3: Why shouldn't `mlflow.db` and `mlruns/` be tracked by git, and why shouldn't they be tracked by dvc either?

### Point your code to the tracking server

Your training script will need to tell mlflow where the tracking server is, and which experiment to log into:

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11")
```

> Question 4: What happens the first time you call `set_experiment` with a name that doesn't exist yet? Check the mlflow UI.

## Training the model

### Training script

Create a file `./src/food11/train.py`. It should:

1. Load a Food-11 dataset with `torchvision.datasets.ImageFolder` and a `DataLoader` (use `food11_processed_mini` while you're developing the script, it's much faster to iterate on).
2. Build a model by taking a pretrained `resnet18` from `torchvision.models` and replacing its final layer so it outputs 11 classes instead of 1000.
3. Accept its hyperparameters as command-line arguments, at least: `--dataset` (processed or mini), `--epochs`, `--lr`, `--batch-size`.
4. Wrap the whole training in `with mlflow.start_run():` and:
   - log the hyperparameters with `mlflow.log_param(...)` (or `mlflow.log_params({...})`) once, at the start
   - at the end of every epoch, log `train_loss`, `val_loss` and `val_accuracy` with `mlflow.log_metric(name, value, step=epoch)`
   - at the end of training, log the final test accuracy and the trained model itself with `mlflow.pytorch.log_model(model, "model")`

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
```

> Question 5: What is the difference between `mlflow.log_param` and `mlflow.log_metric`? Why does `log_metric` take a `step` argument and `log_param` doesn't?

> Question 6: Open the run in the mlflow UI. Find the params, the metric charts, and the logged model artifact. Where does the model artifact actually live on disk?

### Run several experiments and compare

Now run the training script several times, changing one hyperparameter at a time, for example:

```bash
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.01 --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.0001 --batch-size 32
uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 64
```

> Question 7: In the mlflow UI, open the `food11` experiment. Select these runs and click "Compare". Which learning rate gave the best `val_accuracy`? Is higher always better?

> Question 8: Use the parallel coordinates plot on the compare page to look at `lr`, `batch_size` and `val_accuracy` together. What pattern do you see?

> Question 9: Sort the runs table by `val_accuracy` descending. Which run is the best one? Note its run ID, you'll need it in the next lab.

## Commit your training code

The code is versioned with git; the run metadata and metrics stay in mlflow, not in git or dvc.

```bash
git add src/food11/train.py pyproject.toml uv.lock
git commit -m "Add training script with mlflow tracking"
git push
```
