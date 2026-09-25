# DataFrames design doc

## Columns

Columns store data separately from the information about missing values. For numeric data, I use NumPy arrays with zeros filling missing values. These can then be recovered from a validity map, recording which entries contain data. To minimize size, this map is stored as a bitmap. Strings are stored together in a single UTF-8 byte buffer. A separate array stores the offsets, so that we can recover where each string starts and ends. Here, the validity map allows us to distinguish whether an empty string is missing or empty.

```haskell
record Column
    values     : Array a                     -- numeric values or UTF-8 bytes
    data_type  : Maybe DataType
    is_valid   : Maybe (Vector Byte)         -- packed validity bits
    offsets    : Maybe (Vector Int)          -- n + 1 boundaries for strings
    n          : Int                         -- logical number of rows
    bit_offset : Int                         -- first validity bit within bitmap
```

This allows us to represent a column such as `[Nothing, Some 128, Some 420321]` without assigning each value to a separate object.

To read one entry, I first check its validity bit and, if it is a string, decode the bytes between the corresponding offsets. If we need to perform calculations over the whole column, we unpack the validity bits in a Boolean array, and then pack them again for storage. 

Slicing doesn't copy the data: it shares the buffer of the original column and adjusts the reference to start reading it. However, if one selects rows with `take`, we return a copy and rebuild the entire byte buffer for strings.

## Frame
DataFrames store columns and a dictionary so that we can reference to them by name. The `schema` attribute stores the type of each column, while the `groups` attribute is used for GroupBy operations. For now, `metadata` is not used, but one could implement a cache there.

```haskell
record DataFrame
    columns  : Vector Column
    colnames : Map String Int              -- label -> column index
    schema   : Vector (Maybe DataType)
    dims     : (Int, Int)                  -- (ncols, nrows)
    metadata : Vector Attributes
    groups   : Maybe Grouping
```

Upon construction, we check that columns have equal length and the schema and dimensions are correct. One can make a DataFrame using both `from_columns()`, where they need to pass objects of type `column` along with names, or let the code build columns and infer the schema by using `from_list()`. (The `dtypes` helper `guess()` is used for this)

When one makes changes to the DataFrame, a new instance is returned, sharing the columns that were not changed. This means that replacing a column leaves the original frame untouched. Retyping builds the selected column again using the parser, and rejects values that it cannot convert.

### Reading CSV files
`read_csv(path, header=True, skip=0)` uses Polars to read the CSV fields
as text, then builds columns using this library's own type inference.

```haskell
data DataType = Boolean | Int64 | Float64 | String
```
Inference ignores missing entries, and falls back to String when the types of values are incompatible. Headers are used for column names. In the case they are missing, names are generated as `x_0`, `x_1` etc.


## Expressions

Expressions are built to describe a calculation without processing any data. We build them as a tree containing column references, constants and operations. The supported arithmetic, comparisons and Boolean combinations are `+ - * / **, comparisons, and Boolean & |`.

When evaluating an expression, we keep track of both values and the validity mask. The calculation runs on the stored values, and if one of the operand is missing, the result is also marked as missing. 

`call(fn, ...)` is used to include a function in the tree. The result also inherits the validity of the arguments.

## Pipeline

Stores a sequence of steps to execute on a DataFrame, which can be passed later. The `>>` operator builds the pipeline, by storing a multiple objects of type `Step` of different `kind` and arguments. 

Among the kinds I implemented, I report the most notable.

### Keep / Remove

`keep(predicate)` keeps the rows for which the predicate is true. `remove(predicate)` retains the complement, including the rows where the predicate is missing. Both create new columns and clear groups.

### Derive:
Creates or replaces columns by using expressions. Preserves grouping, but if you replace a grouping column this does not rebuild the index.

### GroupBy:
Creates `groups` as a field to `DataFrame`: a dictionary of columns the dataframe should be grouped by, and an index for the rows that satisfy this condition. All operations that change rows or drop a grouping column clear `groups`. This is all operations except for `derive` and `rename` (this operation also renames the grouping keys).

### Execute:
Runs the steps in order using EXECUTORS. Requires a source, either attached to the pipeline or passed to execute(). An attached source takes precedence.

### Aggregate:
Acts on a grouped dataframe, creates one row per group, adding one column per named aggregation. Missing values are excluded before applying the function. If a group has no valid values, its result is None.


## Optimization
For each expression, the method `cols_used()` returns the columns it reads, and `is_rowwise()` says whether row i of the result depends only on row i of the input. A function call counts as row-wise only if it is a NumPy ufunc or the user has marked it with the `@rowwise` decorator.

`optimize()` is a separate interpreter of the pipeline. It switches neighboring steps as long as the results don't change, in two passes:

- Predicate pushdown: a keep/remove moves past a select/exclude, and past a derive if the predicate doesn't read the derived columns and the derive is row-wise.
- Projection pushdown: a select/exclude moves past a filter that uses only columns it keeps, and past an unrelated derive. When a select meets a derive that creates one of its columns, a narrower copy goes ahead of the derive while the original stays in place. The copy keeps the selected columns the derive doesn't create, plus the columns the derive reads.
For ex: `derive(km = distance * 1.609) >> select("carrier", "km")` becomes `select("carrier", "distance") >> derive(km = ...) >> select("carrier", "km")`

Select only copies column references, while keep copies every column. So, pushing filters alone can make a pipeline slower: the filter ends up copying columns that are dropped later. Running projection pushdown as the second step places the narrowed select before the filter, so the filter copies only the columns that are needed. 


# Summary of testing

### Columns
I first check if the type inference functions return the expected data, for strings and integers. Then, I check that validity maps and offsets (for strings) are correctly generated. When slicing, I check that validity maps update, and that filling also correctly updates them.

### DataFrames
The basic checks I run first are for dimensions, names and inferred schema. There is also a test ensuring column names are generated automatically when they are not passed as arguments, and that errors are raised when only one name is passed for two columns, or columns have different length. Then I wrote tests for lookups both by name and index. 

I test assignment by adding and replacing columns, broadcasting a scalar, changing individual cells and the schema. Among others, it's important that the original frame doesn't get changed, while untouched columns remain shared. Other tests cover slicing, missing values, and conversions between integer and string columns.

### Expressions
I test the validity mask and results, to check missing entries are excluded from the value comparison. For optimization, I check that the columns used are reported correctly.

### Read CSV
The CSV tests check the airlines table's dimensions and string type, then verify numeric and string inference on the flights table. They also check that the departure-delay validity mask has one entry per row and contains both present and missing values.

### Pipeline
I check that the pipeline doesn't execute unless the method is called, and that rendering it actually works, as per the handout. I then execute it and assert that the expected column names and size are returned. Once I implemented grouping, I made sure that the rename function rewrites the grouping correctly and other functions drop it, and the aggregate function works.

### Optimization
These test that the `type` of each step in the optimized pipeline is in the expected order (select should go before keep to drop irrelevant columns, etc). I also check that an optimized pipeline outputs the same dataframe as a non optimized one. Both on toy data and on the loaded flights dataframe.

