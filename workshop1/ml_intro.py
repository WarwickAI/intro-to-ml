# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo==0.24.0",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Workshop 1",
    layout_file="layouts/ml_intro.slides.json",
)

with app.setup(hide_code=True):
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

    BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#52514e"
    plt.rcParams.update(
        {
            "figure.figsize": (7, 3.0),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "grid.color": "#e4e3df",
            "xtick.color": INK,
            "ytick.color": INK,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


@app.cell(hide_code=True)
def welcome():
    mo.md("""
    # Workshop 1

    This workshop series builds up your intuition for **neural networks**,
    the technology behind ChatGPT, Claude and image generators.

    Today we use small models you can see on one graph.
    Every idea you meet today is also used to train large neural networks.
    """)
    return


@app.cell(hide_code=True)
def today():
    mo.md("""
    ## Today’s session takes 50 minutes

    | Start | Duration | Activity |
    |---|---|---|
    | 17:05 | 15 minutes | Models, loss and learning |
    | 17:20 | 20 minutes | Lab 1: predict house prices |
    | 17:40 | 15 minutes | Testing a model: data splits, underfitting and overfitting |

    The take-home task builds on the second half. It takes about 15 minutes.
    """)
    return


@app.cell(hide_code=True)
def black_box():
    mo.vstack(
        [
            mo.md("## AI turns inputs into outputs"),
            mo.mermaid("""
            graph LR
                I["Input<br/>a 75 m² flat"] --> M["Model<br/>(black box)"] --> O["Output<br/>£225,000"]
            """),
            mo.md("""
            | System | Input | Output |
            |---|---|---|
            | Price predictor | A flat’s floor area | Its price |
            | Large language model (LLM) | The text so far | The next word |
            | Image classifier | A photo | “cat” or “dog” |
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def learnt():
    mo.vstack(
        [
            mo.md("## Models learn rules from examples"),
            mo.mermaid("""
            graph LR
                R["Programming:<br/>a person writes the rules"] --> P["Program"] --> O1["Output"]
                E["Machine learning:<br/>examples of inputs and correct outputs"] --> T["Training"] --> M["Model"] --> O2["Output"]
            """),
            mo.md("""
            Nobody could write rules for “the next word”, so we show a model examples instead.
            Training adjusts the model’s numbers, called **parameters**, until its outputs match the examples.
            Our price model has 2 parameters. LLMs have billions.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def flat_data():
    _rng = np.random.default_rng(1)
    _area = _rng.uniform(30, 120, 40).round()
    flats = pd.DataFrame({"area": _area, "price": (3 * _area + _rng.normal(0, 25, 40)).round()})


    def show_prediction(model, new_flat):
        _price = model.predict(new_flat)[0]
        _fig, _ax = plt.subplots(figsize=(7, 2.4))
        _line = pd.DataFrame({"area": [30, 120]})
        _ax.scatter(flats["area"], flats["price"], color=BLUE, label="Flats it learnt from")
        _ax.plot(_line["area"], model.predict(_line), color=INK, label="What it learnt")
        _ax.scatter(new_flat["area"], [_price], color=ORANGE, s=150, zorder=3, label="New flat")
        _ax.set(xlabel="Floor area (m²)", ylabel="Price (£k)")
        _ax.legend()
        _fig.tight_layout()
        return mo.vstack([mo.md(f"## Demo: a {new_flat['area'][0]} m² flat costs £{_price:.0f}k"), _fig])

    return flats, show_prediction


@app.cell
def demo_fit(flats, show_prediction):
    model = LinearRegression()
    model.fit(flats[["area"]], flats["price"])  # Learn from 40 example flats.

    new_flat = pd.DataFrame({"area": [100]})  # Try 300: beyond the 30–120 m² it learnt from.
    show_prediction(model, new_flat)
    return


@app.cell(hide_code=True)
def training_loop():
    mo.vstack(
        [
            mo.md("## Training repeats three steps"),
            mo.mermaid("""
            graph LR
                X["Input<br/>75 m²"] -->|1. forward pass| M["Model"] --> P["Prediction (untrained)<br/>£190k"]
                P --> L["2. Loss: how far from<br/>the correct £225k?"]
                L -.->|3. backward pass| M
            """),
            mo.md("""
            1. **Forward pass:** the model turns inputs into predictions.
            2. **Loss:** one number that measures how wrong the predictions are.
            3. **Backward pass:** backpropagation finds which way each parameter should move. **Gradient descent** nudges it.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def weight_slider():
    weight = mo.ui.slider(1.0, 5.0, step=0.1, value=1.5, label="*w*", show_value=True)
    return (weight,)


@app.cell(hide_code=True)
def loss_demo(flats, weight):
    _w = weight.value
    _x, _y = flats["area"], flats["price"]
    _ws = np.linspace(1, 5, 200)
    _losses = [np.mean((_y - w * _x) ** 2) for w in _ws]
    _loss = np.mean((_y - _w * _x) ** 2)

    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10, 2.8))
    _left.vlines(_x, _y, _w * _x, color=ORANGE, linewidth=1, label="Errors")
    _left.scatter(_x, _y, color=BLUE, label="Flats")
    _left.plot([30, 120], [30 * _w, 120 * _w], color=INK, label=f"price = {_w:.1f} × area")
    _left.set(xlabel="Floor area (m²)", ylabel="Price (£k)", ylim=(0, 620))
    _left.legend(loc="upper left")
    _right.plot(_ws, _losses, color=INK)
    _right.scatter([_w], [_loss], color=ORANGE, s=100, zorder=3)
    _right.set(xlabel="w", ylabel="Loss (MSE)")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(r"""
            ## Loss measures how wrong a model is

            Model: $\text{price} = w \times \text{area}$, with no intercept to keep one parameter.
            Loss: $\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$, the average squared error.
            """),
            mo.hstack([weight, mo.md(f"Loss: **{_loss:,.0f}**")], justify="start", gap=2),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def descent_controls():
    steps = mo.ui.slider(0, 15, value=0, label="Steps", show_value=True)
    learning_rate = mo.ui.dropdown(
        {"Small (0.00002)": 0.00002, "Large (0.00014)": 0.00014, "Too large (0.00017)": 0.00017},
        value="Small (0.00002)",
        label="Learning rate",
    )
    return learning_rate, steps


@app.cell(hide_code=True)
def descent_demo(flats, learning_rate, steps):
    _x, _y = flats["area"].to_numpy(), flats["price"].to_numpy()
    _path = [0.5]
    for _ in range(steps.value):
        _w = _path[-1]
        _gradient = 2 * np.mean(_x * (_w * _x - _y))
        _path.append(_w - learning_rate.value * _gradient)
    _path = np.array(_path)
    _path_losses = [np.mean((_y - w * _x) ** 2) for w in _path]

    _ws = np.linspace(-1, 7, 200)
    _fig, _ax = plt.subplots(figsize=(5, 3.4))
    _ax.plot(_ws, [np.mean((_y - w * _x) ** 2) for w in _ws], color=INK, label="Loss for each w")
    _ax.plot(_path, _path_losses, color=ORANGE, linewidth=1, marker="o", label="Training steps")
    _ax.scatter(_path[-1:], _path_losses[-1:], color=ORANGE, s=120, zorder=3)
    _ax.set(xlabel="w", ylabel="Loss (MSE)", xlim=(-1, 7), ylim=(0, 140_000))
    _ax.legend(loc="upper center")
    _fig.tight_layout()

    _status = (
        "The steps overshoot, so the loss grows."
        if _path_losses[-1] > _path_losses[0]
        else f"w = {_path[-1]:.2f}, loss = {_path_losses[-1]:,.0f}"
    )

    mo.vstack(
        [
            mo.md("## Training walks downhill on the loss"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(r"""
                            The **gradient** is the slope of the loss at $w$.
                            Backpropagation calculates it for every parameter.

                            Each step: $w \leftarrow w - \text{rate} \times \text{gradient}$
                            """),
                            steps,
                            learning_rate,
                            mo.md(_status),
                        ]
                    ),
                    _fig,
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def loss_functions():
    _errors = np.linspace(-3, 3, 200)
    _p = np.linspace(0.01, 1, 200)
    _fig, (_mse, _mae, _ce) = plt.subplots(1, 3, figsize=(10, 2.0))
    _mse.plot(_errors, _errors**2, color=BLUE)
    _mse.set(title="Squared error", xlabel="Error")
    _mae.plot(_errors, np.abs(_errors), color=BLUE)
    _mae.set(title="Absolute error", xlabel="Error")
    _ce.plot(_p, -np.log(_p), color=BLUE)
    _ce.set(title="Cross-entropy", xlabel="Probability of the right answer")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(r"""
            ## Choose a loss to suit the task

            | Loss | Formula | Used for | Effect |
            |---|---|---|---|
            | MSE | $\frac{1}{n}\sum (y - \hat{y})^2$ | Numbers | Big errors cost most |
            | MAE | $\frac{1}{n}\sum \lvert y - \hat{y} \rvert$ | Outliers | All errors count evenly |
            | Cross-entropy | $-\frac{1}{n}\sum \log p_\text{correct}$ | Categories | Confident mistakes cost most |
            """),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def lab_one():
    mo.md("""
    ## Lab 1 takes 20 minutes

    Open **ml_lab.ipynb** in Colab and go to **Lab 1**. You will use real house prices from California.

    1. Plot neighbourhood income against house price.
    2. Fit a model and predict prices for two neighbourhoods.
    3. Measure the loss and compare it with a simple guess.
    4. Give the model more inputs and see whether the loss falls.

    If you have time, try the extensions, including gradient descent.
    """)
    return


@app.cell(hide_code=True)
def car_data():
    _rng = np.random.default_rng(2)
    _age = _rng.uniform(0, 15, 50).round(1)
    cars = pd.DataFrame({"age": _age, "price": (25 * np.exp(-_age / 5) + 2 + _rng.normal(0, 1.5, 50)).round(1)})
    cars_train, cars_val = train_test_split(cars, train_size=10, random_state=119)


    def fit_curve(degree):
        _curve = make_pipeline(MinMaxScaler(), PolynomialFeatures(degree), LinearRegression())
        return _curve.fit(cars_train[["age"]], cars_train["price"])


    def plot_curve(curve, show_validation, ax=None):
        if ax is None:
            _fig, ax = plt.subplots(figsize=(5, 3.4))
        _ages = pd.DataFrame({"age": np.linspace(0, 15, 300)})
        ax.scatter(cars_train["age"], cars_train["price"], color=BLUE, zorder=3, label="Training cars")
        if show_validation:
            ax.scatter(cars_val["age"], cars_val["price"], color=ORANGE, label="Validation cars")
        ax.plot(_ages["age"], curve.predict(_ages), color=INK, label="Model")
        ax.set(xlabel="Age (years)", ylabel="Price (£k)", ylim=(0, 32))
        ax.legend(loc="upper right")
        ax.figure.tight_layout()
        return ax.figure


    def losses(curve):
        _train = mean_squared_error(cars_train["price"], curve.predict(cars_train[["age"]]))
        _val = mean_squared_error(cars_val["price"], curve.predict(cars_val[["age"]]))
        return _train, _val

    return fit_curve, losses, plot_curve


@app.cell(hide_code=True)
def degree_slider():
    degree = mo.ui.slider(1, 9, value=1, label="Degree", show_value=True)
    return (degree,)


@app.cell(hide_code=True)
def flexibility(degree, fit_curve, losses, plot_curve):
    _curve = fit_curve(degree.value)
    mo.vstack(
        [
            mo.md("## Low training loss can mislead"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                            Used cars lose value as they age.

                            A higher **degree** gives the curve more parameters, so it can bend more.
                            """),
                            degree,
                            mo.md(f"Training loss: **{losses(_curve)[0]:.1f}**"),
                        ]
                    ),
                    plot_curve(_curve, show_validation=False),
                ],
                widths=[1, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def splits():
    mo.vstack(
        [
            mo.md("## Split the data before you train"),
            mo.mermaid("""
            graph TB
                D["All examples"] --> T["Training (about 70%)<br/>learn the parameters"]
                D --> V["Validation (about 15%)<br/>compare models, choose one"]
                D --> X["Test (about 15%)<br/>one final check"]
            """),
            mo.md("""
            The model never trains on validation or test data, so they act like new examples.
            Choosing by validation loss can fit the validation set by chance, so the test set gives the honest final score.
            For categories, such as spam or not spam, we often measure **accuracy**: the fraction the model gets right.
            """),
        ]
    )
    return


@app.cell(hide_code=True)
def validation(degree, fit_curve, losses, plot_curve):
    _degrees = range(1, 10)
    _train, _val = zip(*[losses(fit_curve(d)) for d in _degrees])
    _fig, _ax = plt.subplots(figsize=(5, 3.4))
    _ax.plot(_degrees, _train, color=BLUE, marker="o", label="Training")
    _ax.plot(_degrees, _val, color=ORANGE, marker="o", label="Validation")
    _ax.axvline(degree.value, color=INK, linewidth=1, linestyle="--")
    _ax.text(0.03, 0.95, "Underfitting", color=INK, transform=_ax.transAxes)
    _ax.text(0.7, 0.95, "Overfitting", color=INK, transform=_ax.transAxes)
    _ax.set(xlabel="Degree", ylabel="Loss (MSE, log scale)", yscale="log")
    _ax.legend(loc="center left")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Validation loss reveals overfitting"),
            mo.hstack([degree, mo.md("Choose the degree with the lowest validation loss.")], justify="start", gap=2),
            mo.hstack([plot_curve(fit_curve(degree.value), show_validation=True), _fig], widths=[1, 1]),
        ]
    )
    return


@app.cell(hide_code=True)
def diagnosis(fit_curve, losses, plot_curve):
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 2.6), sharey=True)
    for _ax, _degree, _name in zip(_axes, [1, 3, 9], ["Underfitting", "A good fit", "Overfitting"]):
        _curve = fit_curve(_degree)
        _train, _val = losses(_curve)
        plot_curve(_curve, show_validation=True, ax=_ax)
        _ax.set_title(f"{_name}: degree {_degree}\ntraining {_train:.1f}, validation {_val:,.1f}", fontsize=10)
        _ax.get_legend().remove()
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("""
            ## Models fail in two ways

            | | Training loss | Validation loss | Cause |
            |---|---|---|---|
            | **Underfitting** | High | High | Too simple to capture the pattern |
            | **Overfitting** | Low | High | Memorised noise in the training data |
            """),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def neural_networks():
    mo.md(r"""
    ## These ideas power neural networks

    | Today | In a neural network |
    |---|---|
    | 1 to 10 parameters | Millions to billions |
    | Forward pass: $w \times \text{area}$ | Inputs flow through many layers |
    | Loss: MSE | MSE, or cross-entropy for LLMs |
    | Gradient descent on one $w$ | Backpropagation and gradient descent on every parameter |

    Training, validation and test sets work the same way.
    Next workshop: regularisation and early stopping, which prevent overfitting.
    """)
    return


@app.cell(hide_code=True)
def take_home():
    mo.md("""
    ## Take-home: find the sweet spot

    Go to the **Take-home** section of the lab notebook. It takes about 15 minutes.

    1. Split the California data into training, validation and test sets.
    2. Train decision trees of increasing depth. Plot both losses.
    3. Choose the best depth and explain why. Check it once on the test set.

    Open-ended: can you get the validation MSE below **0.40**? Try other tree settings, inputs or models.
    """)
    return


if __name__ == "__main__":
    app.run()
