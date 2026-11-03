# Lab 5 - CI/CD with GitHub Actions

This lab continues the project from Labs 1–4. So far every step (testing, building the image, choosing the model to serve) was done by hand from your laptop. In this lab you will make GitHub do it for you: on every push, a workflow lints your code, runs unit and data tests, builds the Docker image, checks that the container starts, and (on `main` only) pushes the image to the GitHub Container Registry tagged with the commit SHA. You will also write a small promotion script that applies a quality gate before moving the `champion` alias.

> What you need to know:
> - a **workflow** is a YAML file under `.github/workflows/`; an **event** (push, pull request, schedule, manual click) starts it; it contains **jobs** that run on fresh virtual machines called **runners**; each job is a list of **steps**
> - a step is either a shell command (`run:`) or a reusable **action** (`uses:`), such as `actions/checkout` or `docker/build-push-action`
> - every run starts from an empty machine: nothing from a previous run is there unless you cache it
> - **secrets** are stored in the repo settings and injected at run time; `GITHUB_TOKEN` is a short-lived token GitHub creates for each run, and its rights are set with `permissions:`
> - CI checks that the code, the data contract and the container are healthy. It does **not** train the real model: there is no GPU and the run would take hours
> - the image and the model are released independently: the image is tagged with the **git commit SHA**, the model is selected with the registry **alias** `champion`

## Before You Start

- Labs 1–4 working: `src/food11/data.py`, `train.py`, `serve.py`, the `Dockerfile` and `docker-compose.yml` pushed to your GitHub repo.
- Your serving code reads `MLFLOW_TRACKING_URI` from the environment (Lab 3).

## Make the code testable

Unit tests must run in a second, without a GPU, without data from DVC and without a running MLflow server. That requires two small refactorings.

### Pure functions in data.py

Make sure `src/food11/data.py` exposes at least these two functions (and that the copy loop only runs under `if __name__ == "__main__":`):

```python
CLASSES = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat",
           "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]

def category_from_filename(filename: str) -> str:
    """'3_120.jpg' -> 'Egg'"""
    ...

def resize_image(img: "PIL.Image.Image", size: int = 128) -> "PIL.Image.Image":
    """Returns an RGB image of size x size."""
    ...
```

### A model that can be replaced in tests

In `src/food11/serve.py`, keep the loaded model in a module-level variable named `model`, and only load it from MLflow if the environment variable `SKIP_MODEL` is not set:

```python
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.environ.get("SKIP_MODEL"):
        return
    model = mlflow.pyfunc.load_model("models:/food11@champion")
```

`/health` should still return 200 when `SKIP_MODEL` is set; `/predict` can return 503 if `model is None`.

> Question 1: Why is it a bad idea for a unit test to load the real model from your MLflow server? List at least two ways the test could fail even though your code is correct.

## Write the tests

```bash
uv add --dev pytest ruff httpx
mkdir -p tests/fixtures
```

Copy one small image from `food11_processed_mini` into `tests/fixtures/bread.jpg` and commit it (it is a few kB, so it goes in git, not DVC).

### Unit tests for the data code

Create `tests/test_data.py`:

```python
from PIL import Image
from src.food11.data import category_from_filename, resize_image, CLASSES

def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"

def test_resize_image_shape_and_mode():
    img = Image.new("L", (512, 384))          # a grayscale, non-square image
    out = resize_image(img)
    assert out.size == (128, 128)
    assert out.mode == "RGB"

def test_eleven_classes():
    assert len(CLASSES) == 11
```

### A test of the API with a fake model

Create `tests/test_api.py`. The fake model returns fixed probabilities so the test checks *your* code (decoding the image, building the JSON), not the model:

```python
from fastapi.testclient import TestClient
import src.food11.serve as serve

class FakeModel:
    def predict(self, x):
        return [[0.90] + [0.01] * 10]

def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        assert client.get("/health").status_code == 200

def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as f:
            r = client.post("/predict", files={"file": ("bread.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0
```

Adapt `FakeModel.predict` to whatever your `serve.py` expects back from the model (a list, a numpy array, a DataFrame...).

```bash
uv run ruff check src tests
uv run pytest -q
```

> Question 2: The fake model always answers "Bread" with 0.90 confidence. What exactly does `test_predict_returns_valid_json` prove, and what does it *not* prove?

## Validate the data against a contract

Create `src/food11/validate_data.py`. Given a processed dataset folder (`data/food11_processed` or `data/food11_processed_mini`), it must check and exit with a non-zero status code if anything fails:

1. each split (`training`, `validation`, `evaluation`) contains exactly the 11 class folders
2. every image opens with PIL, is RGB and is 128×128
3. every class has at least `--min-per-class` images (default 50 for the mini dataset)
4. no image appears in two splits: compare a hash of the file content (`hashlib.md5(path.read_bytes()).hexdigest()`)
5. for each class, its share of the training split is within `--tau` (default 0.05) of its share in the validation split

Print a short report (one line per check, PASS/FAIL) and finish with `sys.exit(1)` if any check failed.

```bash
uv run python src/food11/validate_data.py data/food11_processed_mini
```

Now break the contract on purpose: copy one image from `training/Bread` into `evaluation/Bread`, and run the script again. Then remove the copy.

> Question 3: Which check caught it? Why is a train/test duplicate more dangerous than a corrupted image, even though the corrupted image makes the training script crash and the duplicate does not?

> Question 4: Your CI runner cannot see your data (it is in DVC, and maybe in a local remote). Where in your workflow should `validate_data.py` run so that it still protects you? Give two places.

## The CI workflow

Create `.github/workflows/ci.yml`:

```yaml
name: ci
on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
      - run: uv sync --frozen
      - run: uv run ruff check src tests
      - run: uv run pytest -q

  smoke:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t food11-api:ci .
      - name: start the container without a model and probe it
        run: |
          docker run -d --name api -p 8000:8000 -e SKIP_MODEL=1 food11-api:ci
          for i in $(seq 1 20); do
            curl -sf localhost:8000/health && exit 0
            sleep 2
          done
          docker logs api
          exit 1
```

Commit and push. Open the **Actions** tab of your repo and watch the run.

> Question 5: The `smoke` job has `needs: test`. What happens to it if a unit test fails? What would happen without `needs`?

> Question 6: How long did the whole run take? Run it a second time (push an empty commit with `git commit --allow-empty -m "ci"`) and compare. Which step got faster, and why?

> If your `uv sync` downloads the multi-GB CUDA build of torch, the run will be slow. Use the CPU-only index from Lab 2 in `pyproject.toml`.

## Push the image to the registry

So far the image only exists inside the runner, and is thrown away at the end. Add a third job that runs **only on `main`**, logs in to the GitHub Container Registry (ghcr.io) with the run's `GITHUB_TOKEN`, and pushes the image tagged with the commit SHA:

```yaml
  push:
    needs: smoke
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

> The image name must be lowercase. If your GitHub username has capital letters, replace `${{ github.repository }}` with your lowercase `<username>/<repo>`.

Push to `main`, wait for the run, then find the image under **Packages** on your GitHub profile. Pull it on your laptop:

```bash
docker pull ghcr.io/<username>/<repo>:<commit-sha>
```

> Question 7: This answers Question 9 of Lab 3. Why tag with the commit SHA rather than `latest`? Describe a concrete situation where `latest` leads two machines to run different code.

> Question 8: Why does the `push` job declare `permissions:` explicitly? What could a compromised step do with a token that had write access to everything in the repo?

## Protect main with the pipeline

In the repo settings, add a **branch protection rule** (or ruleset) for `main`: require a pull request before merging, and require the `test` and `smoke` checks to pass.

Then, on a new branch, break a test on purpose (for example, make `category_from_filename` return the wrong class), push the branch and open a pull request.

> Question 9: What does the pull request page show? Could you still merge? What does that change for a team compared to "everybody runs the tests on their laptop before pushing"?

Fix the test, push again, and merge once the checks are green.

## A quality gate for the model

The image is now built and shipped by the pipeline. The model is still chosen by hand in the MLflow UI. Write `src/food11/promote.py` that applies the gate from the lecture:

1. take a candidate model version number as argument (`--version 8`)
2. read the `test_accuracy` metric (or the name you used in Lab 2) of the run that produced the candidate, and of the run that produced the current `champion` (use `MlflowClient().get_model_version(...)` / `get_model_version_by_alias(...)`, then `get_run(run_id).data.metrics`)
3. compute the standard error of the champion's accuracy, `se = sqrt(p * (1 - p) / n)`, with `n` the size of your test split
4. if `acc_candidate >= acc_champion + 2 * se`: move the `champion` alias to the candidate; otherwise set the `challenger` alias on it and leave `champion` where it is
5. in both cases, write the decision and both accuracies as **tags** on the candidate version (`client.set_model_version_tag(...)`)

Train two new runs (for example one with a very small learning rate, one with your best settings), register both, and run the script on each.

```bash
uv run python src/food11/promote.py --version <n>
```

> Question 10: For each candidate, what did the gate decide, and with which numbers? With your test-set size, how many accuracy points does `2 * se` represent?

> Question 11: `promote.py` needs to reach your MLflow server, so it cannot run in GitHub's cloud runners while MLflow lives on your laptop. Name two ways to make it run automatically after training. (Hint: DagsHub also offers a hosted MLflow server per repository.)

## Commit your work

```bash
git add .github/workflows/ci.yml tests/ src/food11/validate_data.py src/food11/promote.py src/food11/data.py src/food11/serve.py pyproject.toml uv.lock
git commit -m "Add CI/CD pipeline, tests, data contract and promotion gate"
git push
```

> Question 12: Draw (in `lab/lab5.md`) the pipeline you now have, from `git push` to a running container. Mark which steps are automatic, which still need a human, and where the code version and the model version are decided.
