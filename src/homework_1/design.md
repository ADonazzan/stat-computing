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

Column-level example:
```haskell
col1 : Vector (Maybe Int)
col1 = [Nothing, Some 128, Some 420321, Nothing, Some -1, Some 482]
```

Column structure:
```haskell
record Column
    values   : Array (Maybe T)          -- array of values
    is_valid : Vector Bool              -- validity mask
    offsets  : Vector Int               -- offsets for variable-length data
```
