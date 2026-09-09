record DataFrame
    columns : Vector Column -- more on the Column type below
    colnames : Map String Int -- associates label with column index
    schema : Vector (Maybe DataType) -- DataType is the type representing T
    dims : (Int, Int) -- (ncols, nrows)
    metadata : Vector Attributes -- more on Attributes type below