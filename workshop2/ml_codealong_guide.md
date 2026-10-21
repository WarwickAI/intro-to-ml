# Workshop 2: code-along, final version

The completed code for the five code-along cells in `ml_codealong.ipynb`. It assumes the recap's first cell has run, which imports everything. Add `from sklearn.model_selection import train_test_split` to that cell's imports.

## 1. Load

```python
tips = load_dataset("scikit-learn/tips")["train"].to_pandas()
train, test = train_test_split(tips, test_size=0.3, random_state=0)  # Hold out 30% for the check.
tips.head()
```

244 bills: 170 to train on, 74 held out.

## 2. Look

```python
print(len(train), "bills to learn from")
train.plot.scatter(x="total_bill", y="tip", s=8)
plt.show()
```

A cloud rising to the right.

## 3. Fit

```python
model = LinearRegression()
model.fit(train[["total_bill"]], train["tip"])
print("slope:", model.coef_[0], "intercept:", model.intercept_)
```

Slope 0.102, intercept 0.92: about 10% of the bill plus 92 cents.

## 4. Predict

```python
new_bills = pd.DataFrame({"total_bill": [20, 50]})
model.predict(new_bills)
```

2.96 for a $20 bill, 6.03 for a $50 bill.

## 5. Check

Score on the held-out bills, which the model has never seen. The baseline always guesses the average tip from the training bills.

```python
predicted = model.predict(test[["total_bill"]])
print("model MSE:", mean_squared_error(test["tip"], predicted))
print("baseline MSE:", mean_squared_error(test["tip"], np.full(len(test), train["tip"].mean())))

test.plot.scatter(x="total_bill", y="tip", s=8)
ends = pd.DataFrame({"total_bill": [0, 55]})
plt.plot(ends["total_bill"], model.predict(ends), color="black")
plt.show()
```

Model 0.87, baseline 1.74.
