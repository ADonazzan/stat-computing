# Streams 
## Takeaways

I implemented streams through generators. This requires storing a function that can be called to generate the stream. This gives immutability by default, but doesn't natively cache values, so reading the stream twice requires computing everything again.

Laziness and caching are two separate choices. Generators give laziness for free. Caching I had to add by hand: `memo` in the differential equation solver, `chunked` instead of reading the stream three times in `euler_improve`, and a stored tuple in `from_seq`.

The major tradeoffs I faced were the following:
- I used loops in primitives, instead of defining them recursively through `cons`. This was to avoid hitting Python's recursion depth.
- The `collect` accumulator appends to one single list, and creates a copy at the end, instead of building the list at each step, which would be the pure alternative.
- I used `Reduced` as a wrapper to indicate that we should stop, but I need to check that every time.
- For fold definitions, I need a thunk. This makes folds reusable, but it clutters the code.
- `convolve` keeps all the terms it has read so far in two lists. This makes each new term cheap to compute, but memory grows.
- `group_by` collects a whole group in a list before returning it. This keeps the code simple and each group re-readable, but a group that never ends is never returned.
- I defined `pairs` recursively using `merge`. The definition is short, but each new row adds a layer of merging, so it slows down and can hit the recursion depth.
- `line_from` opens the file inside the generator. This way the file is opened only when the stream is read and is always closed, but each stream manages only its own file, so on its own it can't limit how many files are open at once (for example with `interleave`).
- To fix that I added `Region`, which owns every file opened during a run, closes them all at the end, and keeps at most `max_open` of them open at a time. This makes `interleave` work over thousands of files, but I have to pass the region by hand, a stream built in a region can't be read after it, and when the pool is full each line costs a reopen and a seek.
- In `lowess` the window is a fixed number of points instead of a distance in x. This lets it work in one pass with little memory, but the data must be sorted by x and the points at the two ends get no smoothed value.

`zip_with(f, s, s)` and `diff(s)` run effects twice, because they read their input twice. I did not solve this in the package; what I have is a list of the operations that are affected (in `docs/answers.md`). For resources I got further, with the `Region` described above. The point I keep is that the stream only runs in a fold or scan, so the block around the fold is the natural place to own the resources.

Operations that are not safe on effectful streams, because they repeat or hide effects:

- reading the input twice: `split_at`, `diff`, `zip_with(f, s, s)`, `head` then `tail`;
- effects for items that are not returned: `drop`, `at`, `slice`;
- look-ahead that loses an item: `take_while`, `group_by`;
- `is_empty`, which must pull an item;
- buffering, which runs effects early: `group_by`, `chunked`, `memo`.

## Answers to the exercises

### Folds

**Puzzle 1.** `take n = fold (collect n)`: `step` appends and signals the end with `Reduced`
when the list has n items.

**dimap** replaces `wrap` by `wrap . f` and `done` by `g . done`. Types:
```
dimap : (v' -> v) -> (r -> r') -> Fold v w r -> Fold v' w r'
lmap  : (v' -> v) -> Fold v w r -> Fold v' w r
rmap  : (r -> r') -> Fold v w r -> Fold v w r'
```

`defer` makes `done` return a fold that starts from the paused accumulator; `resolve` finishes
it. `tee` keeps a tuple of accumulators and stops when all its folds have stopped.

### Estimating pi

`pi_4_series` is `cumsum` of the terms; `euler_improve` maps Euler's formula over
`chunked(s, 3, 1)`.

### Differential equations

`integrate` is a `scan` with a running sum, so it emits `initial_value` first. That is what lets
`y` and `dy` be defined in terms of each other:

```python
def solve_diffeq1(f, y0, dt):
    y = memo(integrate(lambda: dy, dt, y0))
    dy = map(f, y)
    return y
```

`solve_diffeq2(a, b, y0, dy0, dt)` does the same for `y'' = a y' + b y` with two integrations.

### Statistics

All samplers take a stream of uniforms (`uniforms(seed)`), so runs are reproducible.

- **Rejection sampling**: propose uniformly on `[lo, hi]`, accept if `u * bound <= density(x)`;
  it is `map_maybe` over pairs of uniforms.
- **Metropolis-Hastings**, Normal-Normal with known variance: random-walk proposal, written as a
  `scan` over triples of uniforms.
- **Gibbs**, unknown mean and variance: alternates the two full conditionals. True
  values 5 and 4, posterior means 4.990 and 4.198.
- **Lowess**: weighted local line (tricube weights) on each window of k consecutive points;
  `derivative=True` returns the slope. On noisy `sin(x)`, k = 61: maximum error 0.050 for the
  fit and 0.26 for the slope against `cos`.

