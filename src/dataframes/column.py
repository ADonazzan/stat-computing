import numpy as np

from dataframes.dtypes import DataType, guess, parse_int, parse_float, parse_bool, is_missing

PARSERS = {
    DataType.INT64:   parse_int,
    DataType.FLOAT64: parse_float,
    DataType.BOOLEAN: parse_bool,
    DataType.STRING:  lambda s: s,
}

class Column:
    def __init__(self, values, data_type: DataType | None = None, n=None, is_valid=None, offsets=None, bit_offset=0):
        self.values = np.asarray(values)
        self.data_type = data_type
        self.is_valid = is_valid
        self.offsets = offsets
        self.bit_offset = bit_offset
        if n is None:
            n = len(values) if offsets is None else len(offsets)-1
        self.n = n

    @staticmethod
    def split_validity(values):
        valid_mask = np.array([v is not None for v in values], dtype=bool)
        packed_mask = np.packbits(valid_mask, bitorder="little")
        clean = [0 if v is None else v for v in values]
        return (np.asarray(clean), packed_mask)

    @staticmethod
    def build_strings(values: list[str|None]):
        """(byte buffer, packed mask, offsets of length n+1)"""
        mask = np.array([v is not None for v in values], dtype=bool)
        buf = bytearray()
        offsets = [0]
        for v in values:
            if v is not None:
                buf.extend(str(v).encode("utf-8"))
            offsets.append(len(buf))
        return (np.frombuffer(bytes(buf), dtype=np.uint8),
                np.packbits(mask, bitorder="little"),
                np.asarray(offsets, dtype=np.int64))

    @classmethod
    def from_list(cls, values, data_type=None):
        if data_type is None:
            data_type = guess(values)
        values = list(values)
        if data_type is DataType.STRING:
            vals, valid, offsets = cls.build_strings(values)
            return cls(vals, data_type=data_type, is_valid=valid,
                    offsets=offsets, n=len(values))
        vals, valid = cls.split_validity(values)
        return cls(vals, data_type=data_type, is_valid=valid, n=len(values))

    @classmethod
    def from_strings(cls, strings, data_type=None, on_fail="degrade"):
        """Build a column from raw text. Infers the type if not given."""
        if data_type is None:
            data_type = guess(strings)
        if data_type is None:                       # column entirely missing
            return cls.from_list([None] * len(strings), data_type=DataType.STRING)

        parser = PARSERS[data_type]
        typed = []
        for s in strings:
            if s is None or is_missing(s):
                typed.append(None)
                continue
            v = parser(s)
            if v is None:
                if on_fail == "degrade":
                    return cls.from_list(list(strings), data_type=DataType.STRING)
                raise ValueError(f"{s} does not parse as {data_type}")
            typed.append(v)
        return cls.from_list(typed, data_type=data_type)

    @classmethod
    def from_arrays(cls, values, mask, n: int) -> "Column":
        """Build from a values array plus a boolean mask (for pipelines)."""
        values = np.asarray(values)
        if values.ndim == 0:                      # constant expression
            values = np.full(n, values)
        mask = np.asarray(mask) if not isinstance(mask, bool) else np.full(n, mask)
        return cls(values, data_type=guess(values), n=n,
                is_valid=np.packbits(mask, bitorder="little"))

    def _valid_at(self, key:int) -> bool:
        if self.is_valid is None:
            return True
        i = key + self.bit_offset
        byte  = self.is_valid[i // 8]       # determine byte
        shift = i % 8                       # determine bit
        valid = bool((byte >> shift) & 1)
        return valid

    def _valid_mask(self) -> np.ndarray:
        if self.is_valid is None:
            return np.ones(self.n, dtype=bool)
        bits = np.unpackbits(self.is_valid, bitorder="little")
        return bits[self.bit_offset : self.bit_offset + self.n].astype(bool)

        
    def __getitem__(self, key):
        if not self._valid_at(key):
            return None
        if self.offsets is None:
            return self.values[key]
        return self.values[self.offsets[key]:self.offsets[key+1]].tobytes().decode("utf-8")

    def __repr__(self):
        length = self.__len__()
        header = f"Column (len = {length}, dtype = {self.data_type}): "
        shown = [self[i] for i in range(min(self.n, 10))]
        body = ", ".join("None" if v is None else repr(v) for v in shown)
        if self.n > 10:
            body += ", …"
        return header + body

    def __len__(self):
        return self.n

    def slice(self, start: int|None=None, stop: int|None=None):
        '''
            Return sliced column, adjusting bit offset and validity mask for sliced data
        '''       
        start = max(0, min(start, self.n)) if start is not None else 0
        stop = max(start, min(stop, self.n)) if stop is not None else self.n
        
        if self.is_valid is None:
            new_valid, new_bit_offset = None, 0
        else:
            abs_bit = self.bit_offset + start
            new_valid = self.is_valid[abs_bit // 8 :] # drop whole bytes we don't need, and adjust bit offset
            new_bit_offset = abs_bit % 8

        if self.offsets is not None:
            new_offsets = self.offsets[start:stop+1] - self.offsets[start]
            lo, hi = self.offsets[start], self.offsets[stop]
            return Column(self.values[lo:hi], data_type=self.data_type,
                        n=stop-start, is_valid=new_valid, offsets=new_offsets,
                        bit_offset=new_bit_offset)

        return Column(self.values[start:stop], data_type=self.data_type, n=stop-start, is_valid=new_valid, bit_offset=new_bit_offset)

    def with_value(self, i: int, value) -> "Column":
        """New Column with element i set. Copies this column's buffers."""
        if not 0 <= i < self.n:
            raise IndexError(f"Row {i} out of bounds")
        if self.offsets is not None:
            raise NotImplementedError("string columns need the rebuild path")

        vals = self.values.copy()
        if value is not None:
            vals[i] = value

        valid = None if self.is_valid is None else self.is_valid.copy()
        # adjust validity mask: if value is None, set bit to 0; else set bit to 1
        if valid is not None:
            j = i + self.bit_offset
            bit = np.uint8(1 << (j % 8))
            if value is None:
                valid[j // 8] &= ~bit
            else:
                valid[j // 8] |= bit
        return Column(vals, data_type=self.data_type, n=self.n,
                    is_valid=valid, bit_offset=self.bit_offset)

    @property
    def array(self) -> np.ndarray:
        if self.offsets is None:
            return self.values
        return np.array([self[i] for i in range(self.n)], dtype=object)

    def take(self, idx: np.ndarray) -> "Column":
        """Gather rows by integer index"""
        if self.offsets is not None:
            vals = [self[i] for i in idx]
            return Column.from_list(vals, data_type=self.data_type)
        mask = self._valid_mask()[idx]
        return Column(self.values[idx], data_type=self.data_type, n=len(idx),
                    is_valid=np.packbits(mask, bitorder="little"))

    


if __name__ == "__main__":
    col = Column.from_list((1,2,3,4,5), data_type=DataType.INT64)
    print(col)

    col = Column.from_list([None, 128, 420321, None, -1, 482], data_type=DataType.INT64)
    print(col.is_valid)
    print([col[i] for i in range(6)])
    print(col)

    vals = [None if i % 3 == 0 else i for i in range(20)]
    col = Column.from_list(vals, data_type=DataType.INT64)
    sub = col.slice(5, 14)

    print(len(sub), sub)
