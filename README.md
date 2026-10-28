## Intro to ML

Run the slides locally:

```sh
uv run --with marimo==0.24.0 marimo edit --sandbox workshop1/ml_intro.py
```

Open `workshop1/ml_lab.ipynb` in [Google Colab](https://colab.research.google.com/). Save your own copy and use the play buttons to run the cells.

Answers are in `workshop1/ml_lab_solutions.ipynb`.

## Workshop 2

Open `workshop2/ml_codealong.ipynb` in Google Colab.

```sh
uv run --with jupyterlab --with ipywidgets --with scikit-learn --with pandas --with matplotlib --with datasets jupyter lab workshop2/ml_codealong.ipynb
```

## Workshop 3

Open `workshop3/ml_lab.ipynb` in Google Colab. Answers are in `workshop3/ml_lab_solutions.ipynb`.

Both notebooks are generated from the workshop page in the website repo. After editing the page, rebuild them with `python3 workshop3/build_notebooks.py <path to workshop-3/index.mdx>`.

```sh
uv run --with jupyterlab --with ipywidgets --with torch --with scikit-learn --with pandas --with matplotlib jupyter lab workshop3/ml_lab.ipynb
```
