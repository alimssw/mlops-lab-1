Differences from the lab text:

- `serve.py` keeps the lifespan handler from lab 3 instead of `@app.on_event("startup")`, which FastAPI marks deprecated. Same thing otherwise: module-level `model = None`, filled at startup unless `SKIP_MODEL` is set, `/predict` returns 503 while it is `None`.
- In `data.py` the class list is called `CLASSES` (was `CATEGORIES`), `IMAGE_SIZE` is 128 instead of `(128, 128)`, and `category_from_filename` / `resize_image` are used by both the copy loop and `serve.py`.
- `FakeModel` returns logits `[4.5, 0, ..., 0]` because `serve.py` applies a softmax to the model output (the real model outputs logits). Softmax of that is 0.90 on Bread, the number the lab talks about.
- Scripts are run as modules, `uv run python -m src.food11.validate_data ...`, like `train.py` since lab 2 (`package = false`, nothing is installed). `validate_data.py` also works with the lab's `python src/food11/validate_data.py` form.
- `pyproject.toml` got `[tool.pytest.ini_options]` with `pythonpath = ["."]` (the `pytest` entry point does not put the cwd on `sys.path`, so `src.food11` would not import) and `testpaths = ["tests"]` (otherwise pytest walks the 36k files under `data/`). ruff 0.17's isort classified `mlflow` as first-party because of the `mlflow/` folder from lab 4, hence `known-third-party = ["mlflow"]`. ruff also flagged `File(...)` as a default argument (B008), so `/predict` uses `Annotated[UploadFile, File()]`.

- Question 1:

Loading `models:/food11@champion` in a unit test makes the test depend on things that are not the code:

- the MLflow server has to be running and reachable. On the GitHub runner there is no server at all, on the laptop it depends on whether the compose stack is up. The client retries for about 5 minutes before giving up (lab 4), so the test does not even fail fast.
- the registry state. If the `champion` alias is moved, deleted, or the server is fresh (lab 4: the containerized mlflow started empty), the test breaks with no code change.
- it downloads 45 MB of weights and runs a real ResNet forward pass: seconds instead of milliseconds, and any network problem fails it.
- a correct model can still predict the wrong class for the test image. The `"Bread"` assertion would then be testing the model, not the route. Retraining with another seed could flip it.

- Question 2:

It proves the glue code: the route accepts a multipart upload, PIL decodes the bytes, `preprocess` builds the array the model receives, the argmax of the model output is mapped to the right entry of `CLASSES`, softmax puts the confidence in `[0, 1]`, and the response is JSON with the two expected keys and status 200.

It does not prove that the real model loads (tracking URI, alias, artifact download), that the model is any good, that the normalization constants in `serve.py` are the ones used in training, that the real pyfunc model accepts the numpy batch `preprocess` builds, or that its output has the shape `[0]` expects. The fake ignores the pixel values, so a bug that fed garbage pixels of the right shape would pass. Nothing about the Docker image or the environment variables either; that is what the smoke job checks.

- Question 3:

Check 4, "no image in two splits". After copying `training/Bread/0_0.jpg` to `evaluation/Bread/dup_from_training.jpg`:

```
PASS  class folders
PASS  images open, RGB, 128x128
PASS  at least 50 images per class
FAIL  no image in two splits
      evaluation/Bread/dup_from_training.jpg duplicates training/Bread/0_0.jpg
      ... 34 more
PASS  train/val class share within 0.05
```

The other checks cannot see it: the copy is a valid 128x128 RGB jpg, Bread in evaluation goes from 100 to 101 which is above the minimum, and check 5 only compares training with validation. Exit code 1. Copy removed afterwards, `dvc status` clean.

The "34 more" was there before the copy: the mini dataset already fails check 4 as downloaded, with 43 cross-split duplicates (3 evaluation/training, 2 validation/training, 38 validation/evaluation). On the full `food11_processed` it is 612 (249 evaluation/training, 52 validation/training, 311 validation/evaluation), 7.4 % of the evaluation split is also in training. Not an artefact of the resizing: `cmp` on the raw files, e.g. `food11_raw/training/0_106.jpg` and `food11_raw/evaluation/0_131.jpg`, shows byte-identical 62365-byte files. Food-11 as downloaded has leaked images between its splits, so every test accuracy from labs 2 to 4 is a bit optimistic (on the mini set only 3 of 1096 test images, so at most 0.3 points). The data was left as is for this lab.

Why the duplicate is worse: the corrupted file fails loudly. PIL raises, training crashes in the first minute, you fix it, nothing wrong reaches the registry. The duplicate fails silently. Training runs, the metric looks fine, but it is wrong: the model has seen part of its test set, `test_accuracy` is inflated, and every decision built on that number (which run to register, the gate in `promote.py`) is based on a wrong number. Nobody gets an error, the model just does worse in production than its number said.

- Question 4:

The runner has neither the data nor the DagsHub credentials, so:

1. where the data is produced: on the laptop right after `data.py`, before `dvc add` / `dvc push`. As a DVC stage (`dvc stage add -n validate -d src/food11/validate_data.py -d data/food11_processed python -m src.food11.validate_data data/food11_processed`) so `dvc repro` refuses to produce a new `data.dvc` from bad data, or as the first thing `train.py` does, so no model is ever trained on data that breaks the contract, wherever training runs.
2. in CI with the data: store the DagsHub token as a GitHub secret, `dvc pull` in a job, run the validator. 1.2 GB per run, so in a scheduled or manual job rather than on every push. A cheaper variant is a small committed fixture dataset the validator runs on in the `test` job, which at least checks the validator itself.

- Question 5:

`needs: test` makes `smoke` wait for `test` and run only if it succeeded. If a unit test fails, `smoke` is skipped (grey, never starts a runner) and `push` is skipped too because it needs `smoke`. The run is red with one failing job.

Without `needs`, `smoke` would start at the same time as `test` on its own runner, build the image and probe `/health`, and probably go green although the code is broken, since the smoke test does not exercise the bug. Minutes of runner time for an image nobody can use, and a run with mixed red and green jobs. If `push` lost its `needs` too, the broken commit would be published to ghcr.io.

- Question 6:

Three runs on `main`, timings from the Actions API:

| run | commit | total | `uv sync` | `docker build` (smoke) | `build-push-action` (push) |
|---|---|---|---|---|---|
| 1 | `9043229` | 7 min 09 s | 3 s | 1 min 49 s | 4 min 15 s |
| 2 | `08b7247` (empty commit, pushed 25 s after run 1 started) | 6 min 12 s | 3 s | 1 min 50 s | 3 min 25 s |
| 3 | `f25a1cd` (empty commit, pushed after runs 1 and 2 were green) | 2 min 47 s | 5 s | 1 min 45 s | 4 s |

Run 2 overlapped with run 1, so nothing from run 1 was cached yet when it started, and the numbers are the same. Run 3 is the real second run: 2 min 47 s instead of 7 min. The step that got faster is `docker/build-push-action` in the `push` job, 4 min 15 s down to 4 s, because of `cache-from: type=gha`: run 1 saved every layer of the image in the GitHub Actions cache (`cache-to: type=gha,mode=max`), and the Dockerfile and `uv.lock` had not changed, so run 3 found all layers in the cache and only had to push a manifest that already existed in ghcr.

The other steps did not change. `uv sync` was 3 s to 5 s every time: the runner's network is fast and the uv cache from `setup-uv` only holds the downloaded wheels, which is not where the time goes on a fresh machine. `docker build` in the `smoke` job stayed at about 1 min 45 s because that job uses plain `docker build` with no cache configured, so it rebuilds the `uv sync` layer (torch CPU, 2.4 GB image) from scratch on every run. The push job's build is the same Dockerfile, which shows what the gha cache saves.

- Question 7:

The image appeared under Packages as `mlops-lab-1`, one tag per commit on `main`. Pulled on the laptop:

```
docker pull ghcr.io/alimssw/mlops-lab-1:08b7247e0b137d3f833c36bc8862047307a2023c
```

2.44 GB, same size as the local build from lab 3. A first attempt a few minutes earlier returned `denied`: the `push` job of that run had not finished yet, so the tag did not exist and ghcr answers denied rather than not found for anonymous requests.

`latest` is a name that moves. Every push to `main` overwrites it, so it does not identify a build. Machine A pulls `food11-api:latest` on Monday (commit X). Tuesday CI builds commit Y and pushes it as `latest`. Machine B is set up Tuesday and pulls. A serves X, B serves Y, both show the same image name in `docker ps`. A bug seen on A cannot be reproduced on B, and `docker run food11-api:latest` on A keeps using its cached X until someone runs `docker pull`. Rolling back is impossible, the previous `latest` is gone.

With the commit SHA as tag the image is immutable and says what it contains: `ghcr.io/alimssw/mlops-lab-1:<sha>` is the code of `git checkout <sha>`, every machine running that tag runs the same bytes, rollback is running the previous SHA. That is the answer to lab 3 question 9: Dockerfile, `uv.lock` and source are all pinned by the commit, the tag makes it visible.

- Question 8:

By default the job's `GITHUB_TOKEN` gets whatever the repository setting says, often read and write on everything (contents, pull requests, issues, packages, actions). `permissions:` shrinks it to what the job needs: read the code, write packages. It is also written in the workflow where a reviewer sees it.

The job runs third-party code: four actions, a base image, everything `uv sync` installs. If any of it is compromised, it runs with the token. With write on contents it could push commits or tags (a release with a backdoored image), rewrite the workflow files so the next run does more, approve pull requests, delete branches, change other packages. With `contents: read` and `packages: write` the worst case is a bad image under a tag, which the SHA tagging makes visible and the branch protection keeps off `main`.

- Question 9:

Branch `break-test`, commit "Fix category lookup" with an off-by-one in `category_from_filename` (`CLASSES[... - 1]`, so `0_12.jpg` becomes Vegetable-Fruit). Pushed, pull request #1 opened against `main`.

The PR page shows the checks section with "2 failing and 4 skipped checks": the workflow ran twice for the branch (once for the `push` event, once for `pull_request`), each time `test` failed on `test_category_from_filename` (`assert 'Vegetable-Fruit' == 'Bread'`) and `smoke` and `push` were skipped because of `needs`. At the bottom "Merging is blocked", the merge button is greyed out: the rule requires `test` and `smoke` to pass and includes administrators, so even the repo owner cannot merge.

Second commit on the branch, "Fix off-by-one in category_from_filename", pushed. Both workflow runs went green (`test` and `smoke` success, `push` skipped since the branch is not `main`), the PR state changed to "clean" and the merge button became available. Merged after that, together with the answers commit.

What it changes for a team: "everybody runs the tests before pushing" is a promise, this is a mechanism. Nobody has to remember, nobody can skip it when in a hurry, and it does not depend on what is installed on someone's laptop (which Python, which torch, which data lying around): the check runs on a clean machine from the committed files only, so "works on my machine" does not count. The result is visible on the PR for everyone, the reviewer does not have to re-run anything, and `main` can only contain commits that passed, so anyone pulling `main` or pulling the image tagged with its SHA gets code that passed the suite. The cost is a few minutes per PR and the rule that even the owner goes through a PR.

- Question 10:

Both candidates were trained on the mini dataset against the compose mlflow, where `champion` is version 2 (run `4f75d6b4`, 5 epochs, lr 1e-4, test_accuracy 0.7783). The test split of the mini dataset has 1096 images (Rice only has 96 in evaluation), so `se = sqrt(0.7783 * 0.2217 / 1096) = 0.0125` and `2 * se = 0.025`: a candidate has to beat the champion by 2.5 accuracy points, threshold 0.8034.

Candidate version 3, run `cd95d3fc`, lr 1e-6, 5 epochs: test_accuracy 0.0912, which is chance level (1/11 = 0.0909), the learning rate is too small to move the pretrained weights in 5 epochs. Gate: not promoted, `challenger` alias set on version 3.

```
candidate: version 3, run cd95d3fc, test_accuracy 0.0912
champion: version 2, run 4f75d6b4, test_accuracy 0.7783
n_test 1096, se 0.0125, threshold 0.8034 (champion + 2*se)
decision: not promoted, challenger alias set on version 3, champion stays at 2
```

Candidate version 4, run `6756a26b`, best settings (lr 1e-4, 5 epochs, seed 0): test_accuracy 0.7783, exactly the champion's number since it is the same configuration and the same seed on CPU. 0.7783 < 0.8034, so not promoted either, `challenger` moved to version 4 (an alias points at one version, so version 3 lost it).

```
candidate: version 4, run 6756a26b, test_accuracy 0.7783
champion: version 2, run 4f75d6b4, test_accuracy 0.7783
n_test 1096, se 0.0125, threshold 0.8034 (champion + 2*se)
decision: not promoted, challenger alias set on version 4, champion stays at 2
```

Tags written on both versions: `gate_decision`, `gate_candidate_accuracy`, `gate_champion_accuracy`. `champion` stayed on version 2 and the inference container did not need a restart.

- Question 11:

1. DagsHub hosts an MLflow server per repository (`https://dagshub.com/alimssw/mlops-lab-1.mlflow`). Point `train.py` and `promote.py` at it, store the token as `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` repository secrets, and add a workflow job that runs `promote.py` (a `workflow_dispatch` with the version as input, or a `repository_dispatch` event sent by the training script once it has registered the version).
2. a self-hosted GitHub Actions runner on the laptop or on the machine that hosts MLflow. Same workflow file, the job executes where `127.0.0.1:5000` exists.

Also possible: make `promote.py` the last step of training (called from `train.py` after `register_model`, or a DVC stage `promote` after `train` so `dvc repro` does train, register and gate in one go), no GitHub involved.

- Question 12:

Drawn in `lab/lab5.md`.
