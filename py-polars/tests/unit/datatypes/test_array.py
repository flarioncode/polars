import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_series_equal


def test_cast_list_array() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]
    s = pl.Series(payload)

    dtype = pl.Array(pl.Int64, 3)
    out = s.cast(dtype)
    assert out.dtype == dtype
    assert out.to_list() == payload
    assert_series_equal(out.cast(pl.List(pl.Int64)), s)

    # width is incorrect
    with pytest.raises(
        pl.ComputeError,
        match=r"not all elements have the specified width",
    ):
        s.cast(pl.Array(pl.Int64, 2))


def test_array_construction() -> None:
    payload = [[1, 2, 3], [4, 2, 3]]

    dtype = pl.Array(pl.Int64, 3)
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # inner type
    dtype = pl.Array(pl.UInt8, 2)
    payload = [[1, 2], [3, 4]]
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # create using schema
    df = pl.DataFrame(
        schema={
            "a": pl.Array(pl.Float32, 3),
            "b": pl.Array(pl.Datetime("ms"), 5),
        }
    )
    assert df.dtypes == [
        pl.Array(pl.Float32, 3),
        pl.Array(pl.Datetime("ms"), 5),
    ]
    assert df.rows() == []


def test_array_in_group_by() -> None:
    df = pl.DataFrame(
        [
            pl.Series("id", [1, 2]),
            pl.Series("list", [[1, 2], [5, 5]], dtype=pl.Array(pl.UInt8, 2)),
        ]
    )

    assert next(iter(df.group_by("id", maintain_order=True)))[1]["list"].to_list() == [
        [1, 2]
    ]

    df = pl.DataFrame(
        {"a": [[1, 2], [2, 2], [1, 4]], "g": [1, 1, 2]},
        schema={"a": pl.Array(pl.Int64, 2), "g": pl.Int64},
    )

    out0 = df.group_by("g").agg(pl.col("a")).sort("g")
    out1 = df.set_sorted("g").group_by("g").agg(pl.col("a"))

    for out in [out0, out1]:
        assert out.schema == {
            "g": pl.Int64,
            "a": pl.List(pl.Array(pl.Int64, 2)),
        }
        assert out.to_dict(as_series=False) == {
            "g": [1, 2],
            "a": [[[1, 2], [2, 2]], [[1, 4]]],
        }


def test_array_invalid_operation() -> None:
    s = pl.Series(
        [[1, 2], [8, 9]],
        dtype=pl.Array(pl.Int32, 2),
    )
    with pytest.raises(
        InvalidOperationError,
        match=r"`sign` operation not supported for dtype `array\[",
    ):
        s.sign()


def test_array_concat() -> None:
    a_df = pl.DataFrame({"a": [[0, 1], [1, 0]]}).select(
        pl.col("a").cast(pl.Array(pl.Int32, 2))
    )
    b_df = pl.DataFrame({"a": [[1, 1], [0, 0]]}).select(
        pl.col("a").cast(pl.Array(pl.Int32, 2))
    )
    assert pl.concat([a_df, b_df]).to_dict(as_series=False) == {
        "a": [[0, 1], [1, 0], [1, 1], [0, 0]]
    }


def test_array_equal_and_not_equal() -> None:
    left = pl.Series([[1, 2], [3, 5]], dtype=pl.Array(pl.Int64, 2))
    right = pl.Series([[1, 2], [3, 1]], dtype=pl.Array(pl.Int64, 2))
    assert_series_equal(left == right, pl.Series([True, False]))
    assert_series_equal(left.eq_missing(right), pl.Series([True, False]))
    assert_series_equal(left != right, pl.Series([False, True]))
    assert_series_equal(left.ne_missing(right), pl.Series([False, True]))

    left = pl.Series([[1, None], [3, None]], dtype=pl.Array(pl.Int64, 2))
    right = pl.Series([[1, None], [3, 4]], dtype=pl.Array(pl.Int64, 2))
    assert_series_equal(left == right, pl.Series([True, False]))
    assert_series_equal(left.eq_missing(right), pl.Series([True, False]))
    assert_series_equal(left != right, pl.Series([False, True]))
    assert_series_equal(left.ne_missing(right), pl.Series([False, True]))

    # TODO: test eq_missing with nulled arrays, rather than null elements.


def test_array_list_supertype() -> None:
    s1 = pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2))
    s2 = pl.Series([[1.0, 2.0], [3.0, 4.5]], dtype=pl.List(inner=pl.Float64))

    result = s1 == s2

    expected = pl.Series([True, False])
    assert_series_equal(result, expected)


def test_array_in_list() -> None:
    s = pl.Series(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        dtype=pl.List(pl.Array(pl.Int8, 2)),
    )
    assert s.dtype == pl.List(pl.Array(pl.Int8, 2))
