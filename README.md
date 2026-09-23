

### GroupBy:
Creates `groups` as a field to `DataFrame`: a dictionary of columns the dataframe should be grouped by, and an index for the rows that satisfy this condition. All operations that change rows or drop a grouping column clear `groups`. This is all operations except for `derive` and `rename` (this operation also renames the grouping keys).



