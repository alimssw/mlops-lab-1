# Lab 5 pipeline

Two tracks meet in the running container: the code track (left), driven by GitHub Actions, and the model track (right), driven by MLflow on the laptop. `[auto]` = runs by itself, `[human]` = someone has to do it.

```
            CODE TRACK                                        MODEL TRACK
            ==========                                        ===========

 edit src/, tests/            [human]            uv run python -m src.food11.train ...   [human]
         |                                                  |
 uv run ruff / pytest         [human, optional]    run logged in MLflow (params, metrics,
         |                                         logged model)                          [auto]
 git commit, git push         [human]                       |
         |                                         register run as food11 version n      [human]
         v                                                  |
 +-------------------------------------+           uv run python -m src.food11.promote  [human starts it]
 | GitHub Actions: on push / PR        |             --version n
 |                                     |                    |
 | job test                   [auto]   |           gate: acc_n >= acc_champion + 2*se    [auto]
 |   checkout, uv sync, ruff, pytest   |             yes -> alias champion -> version n
 |          | needs                    |             no  -> alias challenger -> version n
 | job smoke                  [auto]   |           tags written on version n
 |   docker build, run with            |                    |
 |   SKIP_MODEL=1, curl /health        |                    |
 |          | needs, main only         |                    |
 | job push                   [auto]   |                    |
 |   ghcr.io/alimssw/mlops-lab-1:<sha> |                    |
 +-------------------------------------+                    |
         |                                                  |
   * CODE VERSION decided here:                       * MODEL VERSION decided here:
     the image tag is the commit SHA                    whatever carries the champion alias
         |                                                  |
 branch protection on main:   [human merges,                |
 PR needs test + smoke green   checks are auto]             |
         |                                                  |
         v                                                  v
 docker pull ghcr.io/alimssw/mlops-lab-1:<sha>   [human]    |
 docker run -e MLFLOW_TRACKING_URI=... <image>   [human]    |
 (or docker compose up with that image)                     |
         |                                                  |
         +------------------> running container <-----------+
                              loads models:/food11@champion
                              at startup
                              (docker compose restart inference
                               to pick up a new champion)       [human]
```
