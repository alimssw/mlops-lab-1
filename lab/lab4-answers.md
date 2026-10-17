Two changes to the lab's templates, both because of things found in lab 3:

- `mlflow/Dockerfile` uses `--artifacts-destination /mlflow-data/mlruns` instead of `--default-artifact-root /mlflow-data/mlruns`. With `--default-artifact-root` pointing at a plain path, clients write artifacts to that path themselves, on their own disk (lab 3 question 3: model sources ended up as `file:C:/...` and the container could not load them). Here it would be worse: the training script runs on the laptop, so the artifacts would land in `C:\mlflow-data\mlruns` on Windows and the mlflow container would never see them. With `--artifacts-destination` the server keeps its default artifact root `mlflow-artifacts:/`, clients upload through the server over HTTP, and the files land on the volume. mlflow is pinned to 3.16.0 to match the client in `uv.lock`.
- mlflow 3.16 checks the `Host` header of every API call (lab 3 question 7). The default list is localhost + private IPs, and `mlflow:5000` is neither. So the compose file sets `MLFLOW_SERVER_ALLOWED_HOSTS` for the mlflow service, which is the env var behind `--allowed-hosts`. Checked from the host with curl against `/api/2.0/mlflow/experiments/search`: `Host: 127.0.0.1:5000` 200, `Host: mlflow:5000` 200, `Host: other:5000` 403. Without the env var the inference container gets the 403 from lab 3 again.

The containerized mlflow starts empty (nothing from the laptop's `mlflow.db` is in it), so the model had to be trained again against it: `uv run python -m src.food11.train --dataset mini --epochs 2 --lr 0.0001` with the compose stack up (`train.py` already points at `127.0.0.1:5000`, which is now the published port of the mlflow container). 68 s, val_accuracy 0.710, test_accuracy 0.741, run `c869b3a2`. Registered as `food11` version 1 and given the alias `champion`, which is what `serve.py` loads (lab 3 used aliases instead of the deprecated stages, same here, "Staging" in the lab text = `champion` alias).

Image sizes: inference 2.44 GB (same as lab 3), mlflow 1.29 GB, frontend 816 MB.

- Question 1:

Ran the mlflow image alone, no volume: `docker run -d --name mlflow-solo -p 5001:5000 mlops-lab-1-mlflow`, then logged one run into a new experiment from the laptop. `docker diff mlflow-solo` shows where it went:

```
A /mlflow-data
A /mlflow-data/mlflow.db
A /mlflow-data/mlruns/1/3d8b472d.../artifacts/note.txt
```

That is the container's writable layer, the thin read-write layer on top of the image layers. `docker stop`, `docker rm`, then a new container from the same image: the UI shows only `Default`, `search_experiments()` returns `['Default']` and `search_runs` returns 0 rows. The run is gone, the sqlite file is gone, the artifacts are gone. `docker rm` deletes the writable layer together with the container, and the image itself was never modified (images are read-only), so the new container starts from the same empty state as the first one.

A plain `docker stop` + `docker start` would have kept the data, because it is the same container and the same writable layer. It is the `rm` (or `docker compose down`, which removes containers) that loses it. The volume is what moves `/mlflow-data` out of the writable layer so that it outlives the container.

- Question 2:

A bind mount (`./mlflow-data:/mlflow-data`) would work functionally, the server does not care where the directory comes from. Reasons for the named volume anyway:

- it is not a folder in the repo. We already had to gitignore `mlflow.db` and `mlruns/` in lab 2, a bind mount would add another tracking db and 50 MB of model files next to the code, waiting to be `git add`ed by accident.
- no host path in the compose file. `mlflow-data:/mlflow-data` works unchanged on any machine, a bind mount needs a path that exists on the host and the right permissions.
- on Docker Desktop a bind mount goes through the Windows to Linux VM file sharing layer. sqlite over that is slow and file locking is unreliable, which is exactly the kind of workload mlflow's backend store is. A named volume lives inside the VM (`/var/lib/docker/volumes/mlops-lab-1_mlflow-data/_data`) on a native Linux filesystem.
- lifecycle is tied to the project: Compose names it `mlops-lab-1_mlflow-data`, `docker compose down -v` removes it, `docker volume ls` shows it. Nothing to clean up by hand.

When a bind mount is the better choice: local dev where you want to open `mlflow.db` with a sqlite browser or read the artifacts directly, or when the data must be a normal folder for backups. Reusing the lab 2/3 `mlflow.db` through a bind mount would not have helped here anyway, its model sources are `file:C:/Users/...` paths that mean nothing inside a container.

- Question 3:

Compose created a network `mlops-lab-1_default` (driver bridge) and attached all three containers to it. From inside the frontend container:

```
getent hosts inference mlflow frontend
172.18.0.3      inference
172.18.0.2      mlflow
172.18.0.4      frontend
cat /etc/resolv.conf -> nameserver 127.0.0.11
```

`127.0.0.11` is Docker's embedded DNS server, which exists on user-defined networks (not on the default `bridge` that `docker run` uses). It answers with the container IP for every service name and container name on that network, and updates when containers come and go. So `mlflow` resolves because there is now a container called `mlflow` on the same network as the one asking.

In lab 3 neither half was true: the API container was on the default bridge network, which has no name resolution between containers, and mlflow was not a container at all but a process on the host bound to `127.0.0.1`. The only ways to reach it were the special name Docker Desktop injects for the host (`host.docker.internal`) or giving up network isolation (`--network host`). Now both ends are containers, so the traffic never leaves the compose network and never touches a published port.

- Question 4:

`inference` is a name that only exists inside this compose network. Run the frontend image alone (`docker run -p 8501:8501 mlops-lab-1-frontend`) and a hardcoded `http://inference:8000` fails with "Name or service not known" on the first upload, and there is no way to fix it without rebuilding the image. With the env var the same image works everywhere: no env var gives `127.0.0.1:8000` (an API started locally with `uvicorn`, or `--network host`), `-e INFERENCE_URL=http://host.docker.internal:8000` reaches the lab 3 container, in compose it is `http://inference:8000`, in kubernetes it would be the service name there.

It is the same reason `serve.py` reads `MLFLOW_TRACKING_URI`: the image is the code, the wiring between services belongs to whoever deploys it (the compose file here). The image should not know how it is going to be deployed.

- Question 5:

From the host:

```
curl http://127.0.0.1:8000/health -> connection refused
curl http://127.0.0.1:8501/       -> 200
curl http://127.0.0.1:5000/health -> 200
```

From inside the frontend container `requests.get("http://inference:8000/health")` -> `{'status': 'ok'}`, and the inference log shows the request coming from `172.18.0.4`, the frontend's IP.

A `ports:` entry maps a host port to a container port so that something outside Docker (a browser, curl on the laptop) can reach it. Containers on the same network do not need it, they talk to each other's IP directly and every port the process listens on is reachable (`EXPOSE 8000` in the Dockerfile is only documentation). Nobody outside needs the inference API: the browser talks to Streamlit, and the `requests.post` in `app.py` runs inside the frontend container, server side, not in the browser. So the frontend reaches `inference:8000` over the compose network and the API has no door to the host at all.

Not publishing it means no port conflict on the laptop (lab 3's container could still be running on 8000), and the raw API is not exposed to anyone who can reach the machine. mlflow and the frontend are published because a human opens them in a browser.

- Question 6:

Two failure modes seen.

mlflow up but empty (first `docker compose up`, before any model was registered): the inference container dies about 10 s after start:

```
mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found
ERROR:    Application startup failed. Exiting.
```

```
SERVICE     STATUS                      PORTS
frontend    Up 34 seconds               0.0.0.0:8501->8501/tcp
inference   Exited (3) 27 seconds ago
mlflow      Up 35 seconds               0.0.0.0:5000->5000/tcp
```

Exit code 3 is uvicorn's "startup failed" code. The frontend is unaffected (its `depends_on` only ordered the start) and an upload gives a Streamlit traceback with a connection error, since there is nothing behind `inference:8000`. With the lab's compose file nothing restarts the container, it stays `Exited` until someone runs `docker compose up` or `restart` again.

mlflow not reachable (`docker compose stop mlflow`, then `docker compose up -d --no-deps inference`): the container does not die right away. It sits in "Up" for exactly 5 minutes, then:

```
urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='mlflow', port=5000): Max retries exceeded
  (Caused by NameResolutionError("Failed to resolve 'mlflow' ... Name or service not known"))
ERROR:    Application startup failed. Exiting.
```

The mlflow client retries failed requests 7 times with backoff factor 2 (`MLFLOW_HTTP_REQUEST_MAX_RETRIES`, `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR`), which adds up to roughly 4 to 5 minutes, and only then does the lifespan raise. A stopped container is removed from the embedded DNS, hence the resolution error. With mlflow's container running but the server not listening yet (the real `depends_on` race) it would be "Connection refused" instead, same retry loop, same exit 3 if it never comes up. On this laptop the race is hard to hit by accident: the mlflow server is accepting connections about 5 s after its container starts and the inference container needs about 6 s just to import torch and mlflow, but nothing guarantees that order.

Fix in `docker-compose.yml` (serve.py unchanged):

```yaml
  mlflow:
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health')"]
      interval: 5s
      timeout: 3s
      retries: 10
      start_period: 10s

  inference:
    depends_on:
      mlflow:
        condition: service_healthy
    restart: on-failure
```

The healthcheck uses python because `python:3.11-slim` has no curl. `condition: service_healthy` makes Compose wait for a passing healthcheck instead of just a started process (`docker compose up -d --wait mlflow` prints `Waiting` then `Healthy`), and `restart: on-failure` covers the other case: if the model is not registered yet the container crashes, Docker restarts it with backoff, and it comes up on its own once the alias exists instead of staying dead. The alternative would be a retry loop around `load_model` in the lifespan.

- Question 7:

```
docker compose ps
SERVICE     STATUS                    PORTS
frontend    Up 8 minutes              0.0.0.0:8501->8501/tcp, [::]:8501->8501/tcp
inference   Up 7 seconds              8000/tcp
mlflow      Up 20 seconds (healthy)   0.0.0.0:5000->5000/tcp, [::]:5000->5000/tcp
```

mlflow and frontend show `0.0.0.0:<host>-><container>/tcp`, a published mapping. inference shows only `8000/tcp`: the port the image `EXPOSE`s, with no host side. Matches the compose file, only the two services with a `ports:` block are published. The `(healthy)` on mlflow comes from the healthcheck added for question 6.

First end to end test through the frontend at `127.0.0.1:8501`, upload of `validation/Bread/0_1.jpg`: `Prediction: Bread (60.6%)`. Same four images as lab 3, posted from the frontend container to `inference:8000/predict`, model version 1:

```
Bread/0_1.jpg          -> {'category': 'Bread', 'confidence': 0.6061}
Soup/9_0.jpg           -> {'category': 'Soup', 'confidence': 0.9735}
Noodles-Pasta/6_0.jpg  -> {'category': 'Noodles-Pasta', 'confidence': 0.932}
Rice/7_0.jpg           -> {'category': 'Bread', 'confidence': 0.4188}
```

- Question 8:

New version: second training run, same lr but 5 epochs (`--epochs 5 --lr 0.0001`, 127 s, val_accuracy 0.736, test_accuracy 0.778, run `4f75d6b4`), registered as `food11` version 2, alias `champion` moved from 1 to 2 with `set_registered_model_alias`. The registry agrees (`champion -> version 2`).

Frontend, same image uploaded again: `Prediction: Bread (60.6%)`. Still version 1. The container log still has the one `loaded ... model_id=m-0dc9a10d... run_id=c869b3a2` line from its start, nothing happened at the alias move. `serve.py` resolves `models:/food11@champion` once in the lifespan and keeps the object in `app.state`, so the alias can point anywhere, the process does not look again. Posting the four images from the frontend container gives exactly the version 1 numbers from question 7.

The command: `docker compose restart inference`. 7 s later the log has a new line, `loaded models:/food11@champion ... model_id=m-637286e5... run_id=4f75d6b4...`, and the same upload gives `Prediction: Bread (69.1%)`.

| image | v1 (2 epochs) | v2 (5 epochs) |
|---|---|---|
| Bread/0_1.jpg | Bread 0.6061 | Bread 0.6909 |
| Soup/9_0.jpg | Soup 0.9735 | Soup 0.986 |
| Noodles-Pasta/6_0.jpg | Noodles-Pasta 0.932 | Noodles-Pasta 0.9698 |
| Rice/7_0.jpg | Bread 0.4188 (wrong) | Bread 0.4203 (wrong) |

The v2 column is digit for digit the lab 3 table: same seed, same hyperparameters, CPU training is deterministic, so retraining on another tracking server produced the same weights.

- Question 9:

`docker inspect` on the container before and after the restart shows the same image id, `sha256:6a28060e...`, and `docker images` shows the inference image was last built at the beginning of the lab. `restart` just stops the process and starts it again in the same container, nothing is rebuilt and nothing is re-created. `docker compose up -d` would not even have done that: nothing in the compose file changed, so Compose would have said the container is up to date and left the old model running. The restart is needed exactly because the thing that changed (the alias) is outside of everything Compose knows about.

What is in the image: the python env with torch and mlflow (1.7 GB), `src/` with `serve.py`, the uvicorn entrypoint. What is not: the model. Each start does `GET /api/2.0/mlflow/registered-models/alias?name=food11&alias=champion`, gets the version and its source, and downloads the artifacts from `mlflow:5000` via the `mlflow-artifacts` proxy into a temp dir inside the container. Also not in the image: the tracking URI, it comes from `environment:` in the compose file. So the image is the runtime, and which model runs is decided at container start by whatever `champion` points at. Same split as lab 3 question 8, the difference is that now the artifact source is a service on the compose network instead of `host.docker.internal`.

Consequence both ways: swapping models is one alias move plus one `restart`, no build, no push, no new image to roll out. But a container is only as good as the registry at the moment it started: two replicas started at different times could serve different versions, and if the volume is gone (question 10) the image alone cannot serve anything.

- Question 10:

`docker compose down` removes the three containers and the network, and says nothing about the volume. `docker volume ls` still lists `mlops-lab-1_mlflow-data`. `docker compose up -d --wait` afterwards: mlflow healthy, inference `loaded ... model_id=m-637286e5...` after 8 s with no help, registry still `food11` v1 + v2, `champion -> 2`, 2 runs. The new mlflow container is a different container, but it mounted the same volume and found the same `mlflow.db` and `mlruns/` in it. Same idea as `data/` and `dvc checkout`: the containers are disposable and get rebuilt from the compose file, the volume is the state.

`docker compose down -v` prints two extra lines, `Volume mlops-lab-1_mlflow-data Removing` / `Removed`, and `docker volume ls` shows nothing. `docker compose up -d` after that: Compose creates a brand new empty volume, mlflow creates a fresh `mlflow.db` with only `Default` in it, and the inference container is back to question 6:

```
SERVICE     STATUS
frontend    Up 39 seconds
inference   Restarting (3) 10 seconds ago
mlflow      Up 50 seconds (healthy)
```

8 restarts in 50 s, 8 times `RESOURCE_DOES_NOT_EXIST: Registered Model with name=food11 not found` in the log, `search_runs` returns 0. Both runs, both versions, the alias and 49 MB of weights are gone. The images are untouched and the code is untouched, so there is nothing to rebuild, and still the stack cannot serve anything: the model is state, not code, and all of the state was on that one volume. `-v` is the equivalent of deleting `.dvc/cache` with no remote to pull from.

To not have to train twice more, the volume was copied out before the `down -v` and copied back afterwards (`docker create --name vbak -v mlops-lab-1_mlflow-data:/data alpine true`, then `docker cp vbak:/data ./backup`, 98 MB, and `docker cp ./backup/. vbak:/data` into the new volume, mlflow restarted so it reopens the db). `up -d --wait` then loaded version 2 again and the registry shows both versions. Without the copy the only way back is training, registering and setting the alias again: none of that state is in git, dvc or the images.

- Question 11:

Compose is a single-host tool: one daemon, one bridge network, volumes that are directories on that one machine's disk. What is missing for the two scenarios:

Three inference replicas behind a load balancer:
- `docker compose up --scale inference=3` does start three containers, and the embedded DNS even round-robins the name `inference` between them. But that is DNS in the client's resolver, not a load balancer: no health based routing, a dead replica stays in the rotation until Docker notices, no retries, and the three still share one host's CPU and RAM (three copies of the 1.7 GB venv and of torch in memory). Streamlit would also need to stop caching the resolved IP.
- a real setup needs a scheduler that places replicas on several machines (Kubernetes, Swarm, Nomad), a service object that gives them one stable virtual IP and spreads requests over the ready ones, readiness probes on `/health` so a replica that is still downloading its model gets no traffic, and a rollout strategy so that `restart` becomes "replace pods one at a time" instead of a few seconds of downtime. Compose has `deploy:` keys for replicas and resources but ignores most of them outside Swarm.
- the image has to be in a registry (lab 3 question 9), each node pulls it, `build: .` means nothing on a cluster.

mlflow surviving a machine failure:
- the named volume is a directory under `/var/lib/docker/volumes` on this laptop. If the disk or the VM dies, the registry and every model version die with it, `down -v` is just the manual version of that. There is no replication, no snapshot, no second copy.
- the state has to leave the machine: a managed database (Postgres) for the backend store instead of a sqlite file, and object storage (S3, GCS, Azure Blob, or DagsHub which already hosts our dvc remote) as `--artifacts-destination`, so that the mlflow container itself holds nothing and can be restarted anywhere. Then the tracking server is stateless and the scheduler can bring it back on another node, with a stable DNS name or service in front so `MLFLOW_TRACKING_URI` keeps working.
- sqlite also would not survive two mlflow replicas writing to the same file, so the backend store swap is needed for availability as well as durability.

Plus the things Compose never had: secrets management (right now there are none, in production the tracking server needs auth and the inference service needs credentials for it), TLS, centralized logs and metrics, and the Host allowlist would have to contain the real external name. Those are the next labs.
