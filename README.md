# SWAG implementation notes

In the monoid case, the aggregation can produce the requested outcomes in the following way:
- Average: the accumulator keeps track of `(sum, count)` and finalizes with `sum / count`.
- Standard deviation: keep track of `(sum, sum_of_squares, count)` and finalizes with the standard deviation formula.
- Regression: track of `(sum_x, sum_y, sum_x_x, sum_x_y, count)` and finalizes computing the intercept and slope.

For these aggregations, the operation is addition and the identity is the zero vector. The implementation allows for other operations and identities too. String concatenation is an example, as we need to keep track of the order.

## Tasks 1-3
`aggregators.py` defines a `Monoid` through an associative operation `op` and an identity `munit`. Information about the monoid, how to annotate it and finalize it is then stored in `Agg`.

Since `Monoid` delegates to Python's addition as the default operation, you can set up screen concatenation by:
```python 
concat_agg = Agg(
        monoid=Monoid(""),
        annotate=identity,
        finalize=identity,
    )
```
`aggregators.py` also contains the code to set up the aggregations above.

`stack.py` defines an aggregate stack. Each entry stores the value and the aggregate to that entry. The `generate()` function in `window.py` adds items to the stack by pushing them on the back. It then computes the total by applying the binary operation to the last elements of front and back stacks, and then transfers them in reversed order to the front stack for removal. 

This setup means that the binary operation is called at most three times for every element in the sequence:
- Once per input added to the back stack.
- At most once per input transferred to the front stack.
- Once per complete output window.

### Streaming

`stream.py` wraps a Python iterator and buffers at most one value. It provides `make_stream`, `next`, and `is_active`.

`swag_stream` consumes inputs incrementally and returns a generator that yields one result per complete window.

The `swag` function collects the generator’s results and returns a list.

## Task #4

To implement the group version, I add an inverse operation to the monoid. We can then remove the oldest value from a running aggregate. `swag_group_stream` stores values in a deque and maintains one total. Here we can use a deque, since the inverse operation can remove the oldest value’s contribution.