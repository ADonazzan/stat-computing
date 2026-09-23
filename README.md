
## Pipeline
Stores a source DataFrame and a sequence of steps. The >> operator builds or combines pipelines without executing them. Each operation records its name and arguments in a Step.

### Execute:
Runs the steps in order using EXECUTORS. Requires a source, either attached to the pipeline or passed to execute(). An attached source takes precedence.

### Select:
Keeps the named columns in the requested order. Returns a new DataFrame and clears grouping, even if grouping columns remain.

### Exclude:
Drops the named columns, preserving the order of the others. Uses select internally, so grouping is cleared.

### Derive:
Adds or replaces columns using expressions. Evaluates them in order, so later expressions can use earlier results. Preserves grouping, but replacing a grouping column does not rebuild its index.

### GroupBy:
Creates `groups` as a field to `DataFrame`: a dictionary of columns the dataframe should be grouped by, and an index for the rows that satisfy this condition. All operations that change rows or drop a grouping column clear `groups`. This is all operations except for `derive` and `rename` (this operation also renames the grouping keys).

### Aggregate:
Acts on a grouped dataframe, creates one row per group, and adding one column per named aggregation. Missing values are excluded before applying the function. If a group has no valid values, its result is None.


## Optimization
For each expression, the method `cols_used()` returns the columns it reads, and `is_rowwise()` says whether row i of the result depends only on row i of the input. A function call counts as row-wise only if it is a NumPy ufunc or the user has marked it with the `@rowwise` decorator.

`optimize()` is a separate interpreter of the pipeline. It switches neighboring steps as long as the results don't change, in two passes:

- Predicate pushdown: a keep/remove moves past a select/exclude, and past a derive if the predicate doesn't read the derived columns and the derive is row-wise.
- Projection pushdown: a select/exclude moves past a filter that uses only columns it keeps, and past an unrelated derive. When a select meets a derive that creates one of its columns, a narrower copy goes ahead of the derive while the original stays in place. The copy keeps the selected columns the derive doesn't create, plus the columns the derive reads.

Running projection pushdown second places the narrowed select before the filter, so the filter copies only the columns that are needed. 

For ex: `derive(km = distance * 1.609) >> select("carrier", "km")` becomes `select("carrier", "distance") >> derive(km = ...) >> select("carrier", "km")`.