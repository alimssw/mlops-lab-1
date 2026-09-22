Server command used for this lab (two flags more than lab 2, see questions 3 and 7 for why):

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --artifacts-destination ./mlruns --allowed-hosts "localhost,127.0.0.1,localhost:5000,127.0.0.1:5000,host.docker.internal,host.docker.internal:5000"
```

- Question 1:

Registered from code with `mlflow.register_model('runs:/008f0ee5bb574c9eb4622a04d03a515e/model', 'food11')`. The Models tab shows one registered model `food11` with version 1 (first model ever registered on this server, so the counter starts at 1). mlflow printed a warning while doing it:

```
Run with id 008f0ee5bb574c9eb4622a04d03a515e has no artifacts at artifact path 'model',
registering model based on models:/m-d37a5600782e431bb817d8bfb487c758 instead
```

Same thing as lab 2 question 6: in mlflow 3 the model is not a file under the run's artifacts, it is a separate "logged model" entity with its own id, so the `runs:/<id>/model` URI gets translated to `models:/m-d37a...`. That is what ends up as the `source` of version 1.

Logged model artifact vs registered model:
- the logged model is the output of one training run: the folder with `MLmodel`, `data/model.pt2`, `requirements.txt`, the signature. It is identified by a run id / model id, it is immutable, and it lives wherever the experiment's artifact root is. There is one per run, most of them are throwaway.
- the registered model is an entry in a catalogue with a human name (`food11`) and numbered versions. A version does not contain the weights, it is a pointer (`source`) to a logged model plus metadata: who registered it, when, description, tags, aliases. It is the thing you hand to people who deploy, they never need to know a run id.

- Question 2:

There are no built-in alias names. `Staging`/`Production`/`Archived` were a fixed list, aliases are free strings. `champion` and `challenger` are just the convention the mlflow docs use. The other difference is that an alias is unique inside a registered model: `models:/food11@champion` always resolves to exactly one version, while several versions could sit in the same stage at once.

Set with `client.set_registered_model_alias('food11', 'champion', 1)`, checked with `get_model_version_by_alias`, which returned version 1.

Why version separately from the run: the run is an experiment record (params, metrics, git commit, curves), the registered model is the deployment side. We had 5 runs in lab 2 and only one of them matters. A version can also come from something that is not a run at all: version 2 below was created directly from an artifact URI. And consumers only ever reference `food11@champion`, so the run that produced the current model can change without anyone downstream noticing.

Why an alias beats a stage: it is a mutable pointer. Reassigning it is one API call, no "transition" workflow, no archiving of the previous version, and the version number never changes. You can have as many pointers as you want (`champion`, `challenger`, `shadow`, `rollback`) instead of a fixed vocabulary. In this lab champion went from version 1 to version 2 and the serving code and the docker image did not change at all.

- Question 3:

`src/food11/serve.py` loads `models:/food11@champion` once in the FastAPI lifespan, the tracking URI comes from `MLFLOW_TRACKING_URI` with `http://127.0.0.1:5000` as default. The class names are imported from `data.py`, the folder order of `ImageFolder` is alphabetical and happens to match the Food-11 index order. Preprocessing is the eval transform from `train.py` (128x128, ToTensor, ImageNet mean/std), done with numpy since the pyfunc model takes a float32 array `[n, 3, 128, 128]` and returns logits `[n, 11]`, softmax on top for the confidence.

Local test:

```
curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/0_1.jpg" http://127.0.0.1:8000/predict
{"category":"Bread","confidence":0.6909}
Soup/9_0.jpg          -> {"category":"Soup","confidence":0.986}
Noodles-Pasta/6_0.jpg -> {"category":"Noodles-Pasta","confidence":0.9698}
Rice/7_0.jpg          -> {"category":"Bread","confidence":0.4203}   (wrong, low confidence, 74% val accuracy model)
pyproject.toml        -> 400 {"detail":"file is not an image"}
```

Why the URI and not the `.pth`:
- the serving code does not know where the weights are, which run they came from, or which version is current. The registry resolves alias -> version -> source -> files. The `.pth` is one file on one laptop.
- pyfunc rebuilds the model from the `MLmodel` metadata (flavor pytorch, pt2 format, signature, python/torch versions). With a `.pth` you need the model class code and the exact `torch.load` incantation in the server, and the signature check is gone.
- it works from any machine that can reach the tracking server, which is the whole point of putting it in a container.

To serve a newer version: nothing in `serve.py`, nothing in the Dockerfile. Register the new run (`register_model` gives version 3), move the alias (`set_registered_model_alias('food11', 'champion', 3)`), restart the container so the lifespan runs again. Rollback is the same call with the old number.

One catch found while containerizing: a `models:/` URI is only as portable as the `source` behind it. Version 1 resolves to `file:C:/Users/alial/.../mlruns/1/models/m-d37a.../artifacts`, because the server from lab 2 was started with `--default-artifact-root ./mlruns`, which means clients read and write artifacts directly on disk. Fine on the laptop, but inside the container that path does not exist:

```
mlflow.exceptions.MlflowException: No such artifact: ''
ERROR:    Application startup failed. Exiting.
```

Fix: start the server with `--artifacts-destination ./mlruns` so it also serves that folder over HTTP under `mlflow-artifacts:/`, and create a version whose source goes through it:

```python
client.create_model_version(
    name="food11",
    source="mlflow-artifacts:/1/models/m-d37a5600782e431bb817d8bfb487c758/artifacts",
    run_id="008f0ee5bb574c9eb4622a04d03a515e",
)
client.set_registered_model_alias("food11", "champion", 2)
```

| version | source | download uri | works from |
|---|---|---|---|
| 1 | `models:/m-d37a...` | `file:C:/Users/alial/.../mlruns/1/models/m-d37a.../artifacts` | this laptop only |
| 2 | `mlflow-artifacts:/1/models/m-d37a.../artifacts` | same, fetched via `GET /api/2.0/mlflow-artifacts/artifacts/...` | anywhere that reaches the server |

Same weights, same run, only the access path differs. With a real setup (proxied artifacts or S3 from day one) there would be no need for this.

- Question 4:

Two builds of the same Dockerfile. First one from scratch, second one after changing one line in `serve.py`:

```
build 1   [builder 5/5] RUN uv sync --frozen --no-dev     DONE 822.4s
          [stage-1 3/4] COPY --from=builder /app/.venv    DONE 9.9s
          [stage-1 4/4] COPY src/ ./src/                  DONE 0.4s
          exporting layers                                42.3s
          transferring context: 380.62kB

build 2   [builder 4/5] COPY pyproject.toml uv.lock ./    CACHED
          [builder 5/5] RUN uv sync --frozen --no-dev     CACHED
          [stage-1 3/4] COPY --from=builder /app/.venv    CACHED
          [stage-1 4/4] COPY src/ ./src/                  DONE 0.2s
          total 22s, transferring context: 2.33kB
```

Each instruction is a layer and its cache key is the instruction plus the checksum of the files it copies. Layers are checked top down, and the first miss invalidates everything below it. `COPY pyproject.toml uv.lock` only changes when the dependencies change, so `uv sync` (14 minutes of downloads here) stays cached across code edits. If everything was copied in one `COPY . .` before `uv sync`, any edit to `serve.py` would change that layer's checksum and the sync would run again every time.

Changing a line in `serve.py`: the builder stage is untouched, the runtime stage reuses the base, the workdir and the venv copy, and only `COPY src/` and the final export happen. BuildKit even only transferred the 2.33kB that changed instead of the whole context.

`--mount=type=cache,target=/root/.cache/uv` is a different cache: it keeps uv's downloaded wheels between builds even when the lock changes, without putting them in any image layer.

- Question 5:

Built a naive single-stage version for comparison (not committed): `FROM python:3.14`, `pip install uv`, `COPY . .`, `uv sync --frozen --no-dev`, same entrypoint.

| image | `docker images` | sum of `docker history` layers |
|---|---|---|
| food11-api:naive | 3.61 GB | 2.76 GB |
| food11-api:latest (multi-stage) | 2.44 GB | 1.92 GB |

(`docker images` on Docker Desktop's containerd store also counts the compressed copy of the layers, `docker history` is the unpacked size, so the two columns differ but the gap is the same 0.85 to 1.2 GB.)

`docker history`, biggest layers:

```
naive                                              multi-stage
1.48GB   RUN uv sync --frozen --no-dev             1.78GB   COPY /app/.venv /app/.venv
694MB    apt-get install (gcc, make, git, ...)     87.6MB   debian base (slim)
202MB    apt-get install (dev libs, headers)       42MB     python 3.14
134MB    debian base                               4.95MB   apt-get install
83.2MB   python 3.14                               53.2kB   COPY src/
82.1MB   RUN pip install uv
65MB     apt-get install
19.9MB   apt-get install
```

The whole difference is the base image and the build tooling. The full `python:3.14` carries about 1 GB of apt layers (compiler toolchain, git, headers) that nothing uses at runtime because every package in the lock is a prebuilt wheel, plus 82 MB for a pip-installed uv that only serves the build. The runtime stage of the multi-stage image never had uv or a compiler in it.

The venv layer itself is not smaller, it is bigger, 1.78 GB against 1.48 GB. Checked inside both containers: the builder ran with `UV_COMPILE_BYTECODE=1`, so the copied venv carries 13 393 `.pyc` files (328 MB) that speed up startup but cost space. And in the naive image `/root/.cache/uv` (1.4 GB) is not on top of the venv: `stat` shows 2 links per file, uv hardlinked the venv into its cache, so the same bytes are counted once. Leaving uv's cache inside a layer is the classic single-stage mistake, it only stayed free here because of the hardlinks. With `UV_LINK_MODE=copy` (needed as soon as cache and venv are on different filesystems, like with a cache mount) the naive image would carry every package twice.

Both images are dominated by the same 1.5 GB of torch + mlflow + their dependencies, and no Dockerfile trick shrinks that. The lever there would be a lighter dependency set for serving only (mlflow-skinny, no scikit-learn, matplotlib, pandas), which is a pyproject change, not a Docker one.

- Question 6:

With the `.dockerignore` the context sent to the daemon is 380 kB (`pyproject.toml`, `uv.lock`, `src/`, and a few small files). Sizes of what it leaves out:

```
1.3G   data/
1.3G   .dvc/          (the cache, same content as data/ again, hashed)
1.2G   .venv/
241M   mlruns/
868K   mlflow.db
310K   .git/
```

So without it every `docker build` would start by streaming about 4 GB from the working tree to the daemon and hashing it, before the first instruction even runs. That is minutes, every build, even when nothing changed, because the context checksum has to be recomputed. The lab list does not mention `.dvc/`, which is as big as the data itself.

Image size: with our Dockerfile nothing changes, because it never does `COPY . .`. Only `pyproject.toml`, `uv.lock` and `src/` are copied, so the extra context is slow but does not end up in a layer. With a naive `COPY . .` Dockerfile all of it would land in the image (4 GB of data, dvc cache, a windows venv and sqlite db in a linux image).

Which one would actually break the build: with our Dockerfile none of them, they are only sent, never copied. With a `COPY . .` Dockerfile `.venv/` is the one to worry about. It is a windows venv (`Scripts/` instead of `bin/`, `python.exe`, `pyvenv.cfg` pointing at `C:\Python314`, windows-only torch wheels). Anything that uses it as is (`PATH=/app/.venv/bin`, `.venv/bin/python`) fails because those paths do not exist, and if `uv sync` runs it has to detect the foreign interpreter and rebuild the venv from scratch, so 1.2 GB got shipped for nothing. The others (`data/`, `mlruns/`, `.dvc/`, `mlflow.db`, `.git/`) do not break anything, they make the image huge and leak the dataset, the sqlite tracking db and the whole git history into an artifact that is meant to be pushed to a registry.

- Question 7:

Checked from a throwaway container before running the api:

```
docker run --rm curlimages/curl -m 5 http://127.0.0.1:5000/health           -> exit 7, connection refused
docker run --rm curlimages/curl -m 5 http://host.docker.internal:5000/health -> HTTP 200
docker run --rm alpine getent hosts host.docker.internal                    -> 192.168.65.254
docker run --rm alpine ip route                                             -> default via 172.17.0.1 dev eth0
```

A container has its own network namespace: its own `lo`, its own `eth0` (172.17.0.2 on the default bridge). `127.0.0.1` inside the container is the container's own loopback, and nothing is listening on port 5000 there, hence the refused connection. The mlflow server is bound to `127.0.0.1` of the host, a different loopback interface.

`host.docker.internal` is a name Docker Desktop injects in the container's `/etc/hosts`. It resolves to `192.168.65.254`, the gateway of the small VM network Docker Desktop runs containers in. Docker Desktop's proxy on the host forwards connections to that address to the host's loopback, which is why a server bound only to `127.0.0.1` is still reachable. On a plain linux docker engine there is no VM and no such name by default, that is why the lab says `--network host` for linux (the container shares the host's network stack, so `127.0.0.1:5000` works there) or you add `--add-host=host.docker.internal:host-gateway`.

Second thing that bit me: reaching the server was not enough. The first container run got

```
API request to endpoint /api/2.0/mlflow/registered-models/alias failed with error code 403 != 200.
Response body: 'Invalid Host header - possible DNS rebinding attack detected'
```

mlflow 3.16 validates the `Host` header of every API request. By default it accepts `localhost`, `127.0.0.1`, `0.0.0.0` and private IP ranges, but the container sends `Host: host.docker.internal:5000`. Fixed with `--allowed-hosts`. The list replaces the default instead of extending it (my first attempt with only `host.docker.internal` locked out my own python client on `127.0.0.1`), so both the local names and the docker name have to be in it. Using `http://192.168.65.254:5000` directly would also have passed, since `192.168.*` is in the default list, but the name is the documented, stable one.

- Question 8:

Container test with `champion` on version 2:

```
docker run -d --name food11-v2 -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 food11-api:latest
healthy after 8s
loaded models:/food11@champion from http://host.docker.internal:5000 (model_id=m-d37a5600782e431bb817d8bfb487c758, run_id=008f0ee5bb574c9eb4622a04d03a515e)
Bread/0_1.jpg -> {"category":"Bread","confidence":0.6909}
Soup/9_0.jpg  -> {"category":"Soup","confidence":0.986}
```

Same numbers as the local run. Then `docker stop food11-v2`, `docker start food11-v2`: healthy again after 7s, same "loaded" line in the logs, same prediction. Then `docker rm` and a brand new container from the same image: same thing, healthy after 7s, no rebuild anywhere.

What is baked into the image (checked with `docker exec`):

```
1.7G   /app/.venv     python deps, torch, mlflow, fastapi
52K    /app/src       our code
```

What is not: the model. Each start downloads it from the server into a temp dir inside the container (`/tmp/tmp05um0dut/MLmodel`, 49M) and that dir dies with the container. The server log shows the 33 `GET /api/2.0/mlflow-artifacts/artifacts/...` requests per start. Also not in the image: the tracking URI, it comes from `-e` at run time.

So the image is only the runtime (interpreter + libraries + code). Which model runs is decided at start time by whatever `champion` points at. Upside: swapping models is an alias move plus a restart, one image serves any version. Downside: the container is not self-contained, if the tracking server is down at startup the container exits with code 3 (that is exactly what the version 1 failures looked like), and two containers started at different times can silently run different models if the alias moved in between. Baking the weights in would give a fully reproducible artifact at the cost of a rebuild per model.

- Question 9:

`Dockerfile`, `.dockerignore`, `serve.py`, `pyproject.toml`, `uv.lock` are in git. The image `food11-api:latest` (sha256:d460565b...) only exists in the docker daemon on this laptop. Missing before a CI runner or a cluster could pull and run it:

- a registry. `docker tag food11-api:latest ghcr.io/alimssw/food11-api:<tag>` and `docker push`. Nothing can pull an image that was never pushed.
- an immutable reference. `latest` is a moving tag. Tag with the git sha (or a version) and reference the image by digest (`@sha256:...`) in the deployment, otherwise "the exact image you just built" is not a thing you can name.
- a reproducible build. `python:3.14-slim` is also a moving tag (it resolved to 3.14.7 today), pin it by digest. uv is already pinned (`0.12.10`) and `uv.lock` is frozen, so the python side is fine.
- a build that is tied to a commit. Right now the image was built from a dirty working tree. CI building on every push and stamping the image with the commit sha closes the gap between "what is in git" and "what is in the registry".
- the runtime side. The image needs a tracking server it can reach, with the artifacts served over HTTP or from an object store (not `file:` paths), the `Host` allowlist set, and credentials for it. `host.docker.internal` only exists on Docker Desktop, so the URI has to be injected per environment. Another machine's mlflow also does not have our `food11` model at all, the registry would have to be a shared one (DagsHub hosts mlflow next to the dvc remote from lab 1) and the model version's source has to point at storage that machine can read.
- for kubernetes specifically: health/readiness probes on `/health` (the model download takes a few seconds, the pod is not ready before that), resource limits (the venv with torch is 1.7 GB and the model 200 MB in memory), and an image pull secret if the registry is private.
