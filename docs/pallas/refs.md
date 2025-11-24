---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(pallas_refs)=
# Pallas Refs

<!--* freshness: { reviewed: '2025-11-24' } *-->

## Pallas function inputs and outputs are `Ref`s

Let's look at a trivial Pallas kernel that adds two vectors

```{code-cell} ipython3
def add_vectors_kernel(x_ref, y_ref, o_ref):
  x, y = x_ref[...], y_ref[...]
  o_ref[...] = x + y
```

Unlike regular JAX functions, it does not take in `jax.Array`s as inputs and
doesn't return any values. Instead, it takes in *`Ref`* objects as inputs and
outputs. These represent mutable buffers in memory. The kernel can read from
the input buffers and write to the output buffer.

**Reading from `Ref`s**

In the body, we are first reading from `x_ref` and `y_ref`, indicated by the
`[...]` (the ellipsis means we are reading the whole `Ref`; alternatively we
also could have used `x_ref[:]`). Reading from a `Ref` like this returns a
`jax.Array`.

**Writing to `Ref`s**

We then write `x + y` to `o_ref`. Mutation has not historically been supported
in JAX -- `jax.Array`s are immutable! `Ref`s are new (experimental) types that
allow mutation under certain circumstances. We can interpret writing to a `Ref`
as mutating its underlying buffer.

**Distunguishing between inputs and outputs**

In the kernel above you may have noticed that the code does not explicitly
specify which `Ref`s are inputs and which are output. This is specified when
calling the kernel via `pallas_call`:

```
@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(
      add_vectors_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
  )(x, y)
add_vectors(jnp.arange(8), jnp.arange(8))
```

`output_shape` is a `PyTree` of class `jax.ShapeDtypeStruct` describing the
shape and dtypes of the outputs. This matches the trailing arguments to the
kernel. All arguments that come prior to the outputs are inputs.

## Indexing and Slicing Refs with `.at`

It is possible to access or modify specific sub-regions (slices) of `Ref`s,
using the `.at` property. Using `my_ref.at[slice]` does not immediately read or
write data; it creates a new `Ref` object that points to a subset of the
original buffer. For example:
 - Slicing: `ref.at[0:128]` creates a view of the first 128 elements.
 - Striding: `ref.at[::2]` creates a strided view.

Once you have a new `Ref` that represents a slice, it can be read or written to
like any other `Ref`. Here is simple example:

```
def add_sliced_kernel(x_ref, y_ref, o_ref):
  mid = x_ref.shape[0] // 2

  x_left = x_ref[:mid][...]
  x_right = x_ref[mid:][...]
  y_left = y_ref[:mid][...]
  y_right = y_ref[mid:][...]

  # The output shape is (4, mid).
  o_ref.at[0][...] = x_left + y_left
  o_ref.at[1][...] = x_left + y_right
  o_ref.at[2][...] = x_right + y_left
  o_ref.at[3][...] = x_right + y_right
```