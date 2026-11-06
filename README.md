# Streams Implementation Notes

The type Stream m v  -> produces values in type v, in an effectful context m. Start with pure streams -> m is the Identity.

Take allows you to pass infinity, so that we drain an infinite stream.


A Fold with a mutable accumulator must copy in done, otherwise scan exposes aliasing