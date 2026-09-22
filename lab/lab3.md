# Lab 3 - Containerizing the model with Docker

This lab continues the project from Lab 1 (git+dvc data) and Lab 2 (training + mlflow tracking). You now have a trained model and a run ID for your best experiment. In this lab you will register that model in the mlflow Model Registry, write a small serving API around it, and package everything into a Docker image so it can run anywhere without a local Python setup.

> What you need to know:
> - a Docker **image** is a built, immutable set of layers (filesystem + metadata); a **container** is a running instance of an image
> - a `Dockerfile` describes how to build an image, one instruction (layer) at a time; Docker caches layers, so ordering instructions from least-to-most-frequently-changed speeds up rebuilds
> - a **multi-stage build** uses one stage to install/build dependencies and a second, smaller stage to run the app, so build tools don't bloat the final image
> - the mlflow **Model Registry** lets you name and version a model (e.g. `food11`) independently of the run that produced it; instead of the old (deprecated) stages like `Staging`/`Production`, current mlflow uses **aliases** — mutable pointers like `champion` that you can reassign to any version — plus free-form **tags** for anything else you want to track
> - a container is isolated from the host network by default; reaching a service running on your host machine (like the mlflow tracking server from Lab 2) requires extra configuration

## Before You Start: What You Need Installed

- **Docker** — Docker Desktop (Mac/Windows) or Docker Engine (Linux). Confirm with `docker --version` and `docker run hello-word`.
- Your Lab 2 project, with at least one completed training run and its run ID noted.

## Register your best model

### Promote a run to the Model Registry

Using the run ID you noted at the end of Lab 2 (the run with the best `val_accuracy`), register its model artifact under a shared name. You can do this either from the mlflow UI or from code — pick whichever you prefer:

**Option A — from the UI:**

1. Open the `food11` experiment, sort the runs table by `val_accuracy` descending, and open your best run.
2. In the run page, open the "Artifacts" tab, select the `model` folder, and click "Register Model".
3. Choose "Create New Model", name it `food11`, and confirm.

**Option B — from code:**

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.register_model('runs:/<your-run-id>/model', 'food11')
"
```

> Question 1: Open the "Models" tab in the mlflow UI. What version number was your model given? What's the difference between a run's logged model artifact and a registered model?

### Assign it the `champion` alias

Model Registry stages (`Staging`/`Production`) are deprecated in current mlflow versions. The replacement is **aliases**: named, mutable pointers to a specific model version (e.g. `champion`, `challenger`) that you can reassign at any time without changing the version number itself.

**Option A — from the UI:**

In the mlflow UI, open the `food11` registered model, select the version you just created, and add the alias `champion` to it.

**Option B — from code:**

```bash
uv run python -c "
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = mlflow.MlflowClient()
client.set_registered_model_alias('food11', 'champion', <your-version-number>)
"
```

> Question 2: What aliases replaced the old built-in stages in mlflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?

## Write a serving script

Create `./src/food11/serve.py`. It should:

1. Use **FastAPI** to expose:
   - `GET /health` — returns `{"status": "ok"}`
   - `POST /predict` — accepts an uploaded image file, runs it through the model, and returns the predicted category and confidence score
2. Load the model once at startup with `mlflow.pyfunc.load_model("models:/food11@champion")`, not by loading a `.pth` file directly.
3. Read the mlflow tracking URI from an environment variable (e.g. `MLFLOW_TRACKING_URI`), defaulting to `http://127.0.0.1:5000`, so the value can be overridden from inside a container.

```bash
uv add fastapi uvicorn python-multipart
```

Run it locally first, before containerizing anything:

```bash
uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000
```

Test it with an image from your mini dataset:

```bash
curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/<some-file>.jpg" http://127.0.0.1:8000/predict
```

> Question 3: Why load the model through an mlflow model URI (`models:/food11@champion`) instead of pointing directly at the `.pth` file on disk? What would you have to change to serve a newer model version?

## Write the Dockerfile

Create a `Dockerfile` at the root of the repo using a multi-stage build:

- **Stage 1 (builder):** start from a Python base image, install `uv`, copy only `pyproject.toml` and `uv.lock`, then run `uv sync --frozen --no-dev` to build the virtual environment.
- **Stage 2 (runtime):** start from a slim Python base image, copy the built virtual environment from Stage 1, copy your `src/` folder, expose port `8000`, and set the container's entrypoint to run uvicorn.

> Question 4: Why copy `pyproject.toml`/`uv.lock` and run `uv sync` *before* copying the rest of the source code, instead of copying everything at once? What happens to the build cache when you only change a line in `serve.py`?

> Question 5: What's the size difference between a naive single-stage image and your multi-stage one? Use `docker history <image>` to see which layers are the biggest.

## Add a .dockerignore

Exclude everything the image build doesn't need: `.venv/`, `data/`, `mlruns/`, `mlflow.db`, `.git/`, `__pycache__/`, etc.

> Question 6: What happens to build speed and image size if you forget the `.dockerignore`? Which of the excluded folders would actually break the build if they were sent to the Docker daemon?

## Build and run the image

```bash
docker build -t food11-api:latest .
```

Your mlflow tracking server is running on the host machine, not inside the container, so the container needs a way to reach it:

- **Linux:** `docker run --network host -e MLFLOW_TRACKING_URI=http://127.0.0.1:5000 food11-api:latest`
- **Mac/Windows:** `docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest`

> Question 7: Why can't the container simply use `127.0.0.1:5000` to reach the mlflow server on your host? What does `host.docker.internal` resolve to?

Test the containerized API the same way you tested it locally:

```bash
curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/<some-file>.jpg" http://127.0.0.1:8000/predict
```

> Question 8: Stop the container and start a new one from the same image. Does the model still load correctly without you rebuilding? What does that tell you about what's baked into the image versus fetched at runtime?

## Commit your work

```bash
git add Dockerfile .dockerignore src/food11/serve.py pyproject.toml uv.lock
git commit -m "Containerize model serving with Docker"
git push
```

> Question 9: The Dockerfile and image are versioned differently — one lives in git, the other doesn't (yet). What's still missing before another machine (like a CI runner or a Kubernetes cluster) could reliably pull and run the exact image you just built?
