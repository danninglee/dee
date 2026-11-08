# Publishing to GitHub

These steps are run by the repository owner from the `dee-release` directory.

## 1. Pre-flight checks

```bash
git status                      # must be clean
python -m pytest -q             # must pass
grep -rn "qlib\|mlflow" . --include=*.py   # must be empty
```

Confirm `README.md` and `CITATION.cff` contain no remaining `TODO` markers once
the arXiv ID and project page are live.

## 2. Create the empty remote repository

1. Open https://github.com/new
2. Repository name: `dee`
3. Visibility: Public
4. Do **not** initialise with a README, `.gitignore` or licence
5. Click "Create repository"

## 3. Push the local history

```bash
git remote add origin https://github.com/<your-account>/dee.git
git push -u origin main
```

If your account uses SSH:

```bash
git remote add origin git@github.com:<your-account>/dee.git
git push -u origin main
```

## 4. Configure the repository page

- Description: `Reference implementation of Diversified Expert Ensemble (ICAIF '26)`
- Topics: `ensemble-learning`, `quantitative-finance`, `machine-learning`,
  `cross-sectional-prediction`, `pytorch`
- Website: the project page URL
- Confirm GitHub detects the MIT licence

## 5. Optional but recommended

- **Zenodo DOI**: sign in at https://zenodo.org, enable the repository in
  GitHub Settings → Integrations → Zenodo, then create a GitHub release
  (`v0.1.0`). Zenodo archives the release and mints a DOI. Add the DOI badge to
  `README.md`.
- **Branch protection**: require the test workflow to pass before merging.
- **CI**: add `.github/workflows/tests.yml` running `pytest` on Python 3.10–3.12.

Example workflow:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest -q
```

## 6. After the arXiv announcement

Replace the placeholders in `README.md` and `CITATION.cff`, then:

```bash
git add README.md CITATION.cff
git commit -m "docs: add arXiv and project page links"
git push
```

Placeholders to replace:

| File | Placeholder |
|---|---|
| `README.md` | `arXiv:XXXX.XXXXX`, `https://example.org/dee`, `<your-account>` |
| `CITATION.cff` | `arXiv:XXXX.XXXXX`, `https://github.com/<your-account>/dee` |
