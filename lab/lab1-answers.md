- Question 1:

`uv init` created:
- `pyproject.toml`: project metadata, name, version, python version required, dependency list.
- `README.md`: empty.
- `.python-version`: pins the python version uv should use (3.14).
- `src/food11/__init__.py`: package skeleton (uv uses a src/ layout). Renamed the folder from the default `mlops_lab_1` to `food11` since that's where `data.py` needs to go.
- `uv.lock` (shows up after the first `uv run`): exact resolved versions of every dependency, so installs are reproducible.

- Question 2:

`dvc init` created:
- `.dvc/config`: the actual dvc settings (remotes etc), no secrets in it. Goes to git.
- `.dvc/.gitignore`: makes git ignore `config.local`, `tmp`, `cache`. Goes to git.
- `.dvcignore`: same idea as `.gitignore` but for dvc scanning. Goes to git.
- `.dvc/tmp/`: internal scratch/lock files dvc uses. Not committed.
- `.dvc/cache/` (shows up after `dvc add`): the actual file content, stored by md5 hash. Not committed, this is dvc's job not git's.
- `.dvc/config.local` (shows up once you set remote credentials with `--local`): username/token. Never committed.

- Question 3:

Used `dvc remote modify origin --local ...`, so the credentials sit in `.dvc/config.local`, which is git-ignored automatically.

Other than `--global` there's:
- no flag at all: writes straight to `.dvc/config`, which does get committed. Don't put secrets there.
- `--local`: writes to `.dvc/config.local`, stays git-ignored, specific to this repo clone. What we used.
- `--global`: writes to a per-user config outside any repo, applies to every dvc project on the machine.

Credentials should never end up on GitHub, that's the whole reason to use `--local` instead of just editing `.dvc/config`.

- Question 4:

After `dvc add data`, a `.gitignore` showed up in the root with just:

```
/data
```

Once dvc starts tracking `data/` (via `data.dvc` + its cache), git shouldn't touch the actual files anymore, only the pointer. So dvc adds `/data` to `.gitignore` itself so a `git add .` can't accidentally commit the images.

- Question 5:

Yes, `data.dvc`:

```yaml
outs:
- md5: a3a457d03c51ff8b037a833440f6ad13.dir
  size: 1188442712
  nfiles: 16643
  hash: md5
  path: data
```

It's a pointer file: hash of the whole `data/` directory, total size, file count. Git tracks this small file instead of the data itself; dvc uses the hash to find/restore the real files in the cache or remote.

- Question 6:

Checked the GitHub repo contents directly (via the API): root has `.dvc/`, `.dvcignore`, `.gitignore`, `.python-version`, `README.md`, `data.dvc`, `lab/`, `pyproject.toml`, `src/`, `uv.lock`.

- Code: yes, it's there.
- Data: no, there's no `data/` folder on GitHub.
- Pointer to the data: yes, `data.dvc` (question 5). Doesn't hold a literal path, just a hash dvc resolves against whatever remote is set (DagsHub here).
- DagsHub UI: once `dvc push` actually finishes, DagsHub shows the `data/` folder in its web UI like a normal browsable folder, even though git never stored those files, because DagsHub understands dvc pointers natively.

(First `dvc push` attempt died partway through from a network drop, had to retry.)

- Question 7:

A fresh clone gets everything git tracks, code, `data.dvc`, configs, lab notes, but not the `data/` folder itself since that was never in git.

To actually get the files:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <token>
dvc pull
```

`dvc pull` reads `data.dvc`, matches the hash against the remote, downloads into `./data`. Credentials aren't in git either so a fresh clone needs `.dvc/config.local` set up again first, same as question 3.

- Question 8:
