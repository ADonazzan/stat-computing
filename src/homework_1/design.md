## Dataframe representation

A dataframe over `T` is a tuple ⟨C, L, S⟩ (column, label, schema), where `T` is a set of types, represented as:

```haskell
record DataFrame
    columns  : Vector Column            -- arrays
    colnames : Map String Int           -- label -> column index
    schema   : Vector (Maybe DataType)  -- information on col types
    dims     : (Int, Int)               -- (ncols, nrows)
    metadata : Vector Attributes        -- df length, other info...
```

Each type has a parsing function: U -> Maybe t

### Data Types

Implemented for now:
- bool
- int64
- float64
- string

The infer function determines the type of a string in the order above, returning the first match.
The guess function is used to determine the type of a column. Loops through the values using infer and returns the type that is compatible with all values, or none.

### Columns
Example:
```haskell
col1 : Vector (Maybe Int)
col1 = [Nothing, Some 128, Some 420321, Nothing, Some -1, Some 482]
```

Column structure:
```haskell
record Column
    values   : Array (Maybe T)          -- array of values
    data_type: DataType                 -- type of the column
    is_valid : Vector Bool              -- validity mask
    offsets  : Vector Int               -- offsets for variable-length data
```

bit offset added for column slicing: once I slice I copy is_valid and tell where to start looking for new first valid value.

### Questions
- Tests? 