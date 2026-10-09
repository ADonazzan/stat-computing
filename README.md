# SWAG implementation notes

In the monoid case, the aggregation can produce the requested outcomes in the following way:
- Average: the operation keeps track of `(sum, count)` and finalizes with `sum / count`.
- Standard deviation: keep track of `(sum, sum_of_squares, count)` and finalizes with the standard deviation formula.
- Regression: track of `(sum_x, sum_y, sum_x_x, sum_x_y, count)` and finalizes computing the intercept and slope.

The implementation allows for other operations and identities too. String concatenation is an important example, as we need to keep track of the order.

## Task 1
Data on what we need to aggregate is contained in `aggregators.py`. Here we define a `Monoid` through an associative operation `op` and an identity `munit`. The function `basic` then takes 

```python 
concat_agg = basic(Monoid(""))
```