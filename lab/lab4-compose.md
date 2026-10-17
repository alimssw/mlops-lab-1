# Lab 4 - Orchestrating the app with Docker Compose

This lab continues the project from Lab 1 (git+dvc data), Lab 2 (training + mlflow tracking) and Lab 3 (a single Dockerfile serving the model). So far you've only ever run one container at a time, reaching your host's mlflow server through `host.docker.internal` or `--network host`. In this lab you will containerize mlflow itself and add a small web frontend, then bring all three services up together with Docker Compose: a **mlflow** service (tracking server + registry), an **inference** service (your Lab 3 API, now pointing at the mlflow container instead of your host), and a **frontend** service (a page where a human uploads an image and sees the prediction).

> What you need to know:
> - `docker compose` reads a `docker-compose.yml` and starts a set of *services* together, each built from its own image, on a private network it creates for you
> - inside that network, containers reach each other by **service name** (e.g. `http://mlflow:5000`) instead of `host.docker.internal` or a published port — Compose's embedded DNS resolves the name to the right container
> - a **named volume** is storage managed by Docker and kept outside any single container's filesystem; it's how you keep mlflow's database and artifacts alive across `docker compose down` / `up`, the same way `data/` is kept alive across `dvc checkout`
> - `depends_on` controls **start order**, not **readiness** — mlflow's container can be "started" before it's actually accepting connections, so the inference service still needs to handle a connection failure at startup
> - only the ports a human needs to reach directly should be `published` (mapped to the host); service-to-service traffic never needs a published port, since it stays inside the Compose network

## Before You Start: What You Need Installed

- Everything from Lab 3 already working: a Dockerfile that builds your inference API, and at least one model version registered in mlflow and moved to `Staging`.
- **Docker Compose** — bundled with Docker Desktop; on Linux, confirm the plugin is present with `docker compose version` (note: no hyphen, this is the newer `docker compose`, not the old standalone `docker-compose`).

## Project layout

You'll end up with a repo root that looks roughly like this:

```
./Dockerfile              # from Lab 3, the inference service image
./src/food11/serve.py     # from Lab 3
./mlflow/Dockerfile       # new: the mlflow tracking server image
./frontend/Dockerfile     # new: the frontend image
./frontend/app.py         # new: the Streamlit upload page
./docker-compose.yml      # new: wires the three together
```

## The mlflow service

### Dockerfile

Create `./mlflow/Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--default-artifact-root", "/mlflow-data/mlruns"]
```

Note the path is under `/mlflow-data`, not `/mlruns` at the container root — this is the mount point for the volume you'll declare in `docker-compose.yml`, so the database and artifacts land on the volume instead of the container's writable layer.

> Question 1: What happens to everything written to `/mlflow-data` if you never mount a volume there and just `docker run` this image standalone? Try it: run the container, register nothing, stop it, remove it, start a new one from the same image — what do you see in the UI?

> Question 2: Why a named volume here instead of a bind mount to a folder in your repo (the way you might for local dev)? Would a bind mount work just as well?

## Update the inference service

Your Lab 3 `serve.py` already reads `MLFLOW_TRACKING_URI` from an environment variable, defaulting to `http://127.0.0.1:5000`. Nothing in the code needs to change — only the value you pass at container-run time will differ.

> Question 3: In Lab 3 you had to use `host.docker.internal` or `--network host` to reach mlflow from inside the container. In this lab, `MLFLOW_TRACKING_URI` will simply be `http://mlflow:5000`. Why does that hostname resolve now when it didn't before?

## The frontend service

### A minimal upload page

Create `./frontend/app.py`:

```python
import os
import requests
import streamlit as st

INFERENCE_URL = os.environ.get("INFERENCE_URL", "http://127.0.0.1:8000")

st.title("Food-11 classifier")

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, width=300)
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    response = requests.post(f"{INFERENCE_URL}/predict", files=files)
    if response.ok:
        result = response.json()
        st.write(f"**Prediction:** {result['category']} ({result['confidence']:.1%})")
    else:
        st.error(f"Inference service returned {response.status_code}: {response.text}")
```

> Question 4: Why does the frontend read `INFERENCE_URL` from an environment variable instead of hardcoding `http://inference:8000`? Think about what happens if you ever `docker run` this frontend image on its own, outside Compose.

### Dockerfile

Create `./frontend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir streamlit requests

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

## Wire it together with Docker Compose

Create `./docker-compose.yml` at the repo root:

```yaml
services:
  mlflow:
    build: ./mlflow
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow-data

  inference:
    build: .
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - mlflow

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    environment:
      INFERENCE_URL: http://inference:8000
    depends_on:
      - inference

volumes:
  mlflow-data:
```

> Question 5: Only `mlflow` and `frontend` publish a port to the host. `inference` doesn't. Why not, and how does the frontend still reach it?

> Question 6: `depends_on` here only waits for the mlflow *container process* to start, not for the tracking server inside it to be ready to accept connections. If your `serve.py` tries to load the Staging model at startup and mlflow isn't ready yet, what happens to the `inference` container? Look at `docker compose logs inference` if it fails.

## Bring the stack up

```bash
docker compose up --build
```

Open the mlflow UI at [http://127.0.0.1:5000](http://127.0.0.1:5000) and the frontend at [http://127.0.0.1:8501](http://127.0.0.1:8501). Upload an image and confirm you get a prediction back.

> Question 7: Run `docker compose ps`. Which services have a published port listed, and which don't? Does that match what you'd expect from the `docker-compose.yml`?

## Promote a new model version and reload

In the mlflow UI, register a new version of your model (or re-register the same run under a new version) and move it to `Staging`, replacing the version currently there.

> Question 8: Refresh the frontend and upload an image again. Does the prediction come from the new model version, or the old one? Your `serve.py` loads the model once, at startup — what single command lets you pick up the new Staging version without rebuilding any image?

Try it:

```bash
docker compose restart inference
```

> Question 9: Why does `restart` alone work here — no rebuild needed? What does that tell you about what's baked into the inference image versus fetched at container startup?

## Persistence check

```bash
docker compose down
docker compose up
```

> Question 10: Is your registered model and its Staging assignment still there after this `down`/`up` cycle? Now try `docker compose down -v` followed by `docker compose up` — what's different this time, and why?

## Commit your work

```bash
git add mlflow/Dockerfile frontend/Dockerfile frontend/app.py docker-compose.yml
git commit -m "Orchestrate mlflow, inference and frontend with Docker Compose"
git push
```

> Question 11: This compose file is still meant to run on one machine. What would have to change for the `inference` service to run as three replicas behind a load balancer, or for the mlflow service to survive a machine failure? (You don't need to implement this — just name what Docker Compose can't give you here.)
