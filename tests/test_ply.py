"""P0.3: PLY reading and writing.

Most inputs here are built byte by byte, independently of psplat's writer, in
the layout the Inria reference code writes through plyfile. That way a shared
misunderstanding between reader and writer cannot hide.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from psplat.io.ply import PlyFormatError, read_ply, read_ply_header, write_ply
from psplat.io.scene import Scene

INRIA_BASE = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
INRIA_TAIL = ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def inria_names(degree: int = 3) -> list[str]:
    rest = 3 * ((degree + 1) ** 2 - 1)
    return [*INRIA_BASE, *(f"f_rest_{i}" for i in range(rest)), *INRIA_TAIL]


def inria_bytes(values: np.ndarray, names: list[str], comments: list[str] = ()) -> bytes:  # type: ignore[assignment]
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines += [f"element vertex {values.shape[0]}"]
    lines += [f"property float {n}" for n in names]
    lines += ["end_header"]
    return ("\n".join(lines) + "\n").encode() + values.astype("<f4").tobytes()


def random_values(n: int, p: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, p)).astype(np.float32)


# Reading the standard layout -------------------------------------------------------


def test_reads_all_62_inria_attributes(tmp_path: Path) -> None:
    names = inria_names(3)
    assert len(names) == 62
    values = random_values(10, 62)
    (tmp_path / "s.ply").write_bytes(inria_bytes(values, names))
    scene = read_ply(tmp_path / "s.ply")
    col = {name: values[:, i] for i, name in enumerate(names)}
    assert scene.count == 10 and scene.sh_degree == 3
    np.testing.assert_array_equal(scene.means, np.stack([col["x"], col["y"], col["z"]], 1))
    np.testing.assert_array_equal(scene.opacities, col["opacity"])
    np.testing.assert_array_equal(scene.quats[:, 0], col["rot_0"])
    np.testing.assert_array_equal(scene.scales[:, 2], col["scale_2"])
    np.testing.assert_array_equal(scene.sh0[:, 1], col["f_dc_1"])
    assert list(scene.extra) == ["nx", "ny", "nz"]


def test_f_rest_is_channel_major(tmp_path: Path) -> None:
    names = inria_names(3)
    values = np.zeros((1, 62), np.float32)
    for i in range(45):
        values[0, names.index(f"f_rest_{i}")] = i
    (tmp_path / "s.ply").write_bytes(inria_bytes(values, names))
    shn = read_ply(tmp_path / "s.ply").shN
    for k in range(15):
        for c in range(3):
            assert shn[0, k, c] == c * 15 + k


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_round_trip_is_byte_identical(tmp_path: Path, degree: int) -> None:
    names = inria_names(degree)
    source = inria_bytes(random_values(37, len(names), degree), names, ["made by a trainer"])
    (tmp_path / "in.ply").write_bytes(source)
    write_ply(read_ply(tmp_path / "in.ply"), tmp_path / "out.ply")
    assert (tmp_path / "out.ply").read_bytes() == source


def test_round_trip_after_reordering(tmp_path: Path) -> None:
    names = inria_names(3)
    values = random_values(20, 62)
    (tmp_path / "in.ply").write_bytes(inria_bytes(values, names))
    order = np.arange(20)[::-1]
    write_ply(read_ply(tmp_path / "in.ply").take(order), tmp_path / "rev.ply")
    assert (tmp_path / "rev.ply").read_bytes() == inria_bytes(values[order], names)


def test_mixed_types_and_unknown_attributes_are_preserved(tmp_path: Path) -> None:
    props = [("float", n) for n in inria_names(1)[:3]]
    props += [("uchar", "red"), ("double", "confidence")]
    props += [("float", n) for n in inria_names(1)[3:]]
    props += [("int", "cluster")]
    dtype = np.dtype([(n, {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4"}[k])
                      for k, n in props])  # fmt: skip
    rec = np.zeros(4, dtype)
    for name in dtype.names or ():
        rec[name] = np.arange(4) + 1
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", "element vertex 4"]
        + [f"property {k} {n}" for k, n in props]
        + ["end_header"]
    )
    source = (header + "\n").encode() + rec.tobytes()
    (tmp_path / "in.ply").write_bytes(source)
    scene = read_ply(tmp_path / "in.ply")
    assert list(scene.extra) == ["red", "confidence", "nx", "ny", "nz", "cluster"]
    assert scene.extra["red"].dtype == np.uint8
    assert scene.extra["confidence"].dtype == np.float64
    write_ply(scene, tmp_path / "out.ply")
    assert (tmp_path / "out.ply").read_bytes() == source


def test_big_endian_and_crlf_headers_read(tmp_path: Path) -> None:
    names = inria_names(0)
    values = random_values(3, len(names))
    header = "\r\n".join(
        ["ply", "format binary_big_endian 1.0", "element vertex 3"]
        + [f"property float {n}" for n in names]
        + ["end_header"]
    )
    (tmp_path / "be.ply").write_bytes((header + "\r\n").encode() + values.astype(">f4").tobytes())
    (tmp_path / "le.ply").write_bytes(inria_bytes(values, names))
    assert read_ply(tmp_path / "be.ply").equals(read_ply(tmp_path / "le.ply"))


def test_ascii_ply_reads(tmp_path: Path) -> None:
    names = inria_names(0)
    values = np.arange(2 * len(names), dtype=np.float32).reshape(2, -1)
    header = "\n".join(
        ["ply", "format ascii 1.0", "element vertex 2"]
        + [f"property float {n}" for n in names]
        + ["end_header"]
    )
    body = "\n".join(" ".join(f"{v:g}" for v in row) for row in values)
    (tmp_path / "a.ply").write_text(header + "\n" + body + "\n")
    (tmp_path / "b.ply").write_bytes(inria_bytes(values, names))
    assert read_ply(tmp_path / "a.ply").equals(read_ply(tmp_path / "b.ply"))


def test_elements_before_vertex_are_skipped_with_a_warning(tmp_path: Path) -> None:
    names = inria_names(0)
    values = random_values(2, len(names))
    header = "\n".join(
        ["ply", "format binary_little_endian 1.0", "element meta 3", "property int tag",
         "element vertex 2"]
        + [f"property float {n}" for n in names]
        + ["end_header"]
    )  # fmt: skip
    body = np.array([7, 8, 9], "<i4").tobytes() + values.astype("<f4").tobytes()
    (tmp_path / "m.ply").write_bytes((header + "\n").encode() + body)
    with pytest.warns(UserWarning, match="meta"):
        scene = read_ply(tmp_path / "m.ply")
    np.testing.assert_array_equal(scene.means, values[:, :3])


# Writing fresh scenes ------------------------------------------------------------------


def _fresh(n: int = 5, k: int = 15) -> Scene:
    rng = np.random.default_rng(3)
    f = np.float32
    return Scene(
        means=rng.normal(size=(n, 3)).astype(f),
        scales=rng.normal(size=(n, 3)).astype(f),
        quats=rng.normal(size=(n, 4)).astype(f),
        opacities=rng.normal(size=n).astype(f),
        sh0=rng.normal(size=(n, 3)).astype(f),
        shN=rng.normal(size=(n, k, 3)).astype(f),
    )


def test_fresh_scene_writes_the_standard_62_property_layout(tmp_path: Path) -> None:
    write_ply(_fresh(), tmp_path / "s.ply")
    vertex = read_ply_header(tmp_path / "s.ply").element("vertex")
    assert [name for name, _ in vertex.properties] == inria_names(3)
    back = read_ply(tmp_path / "s.ply")
    np.testing.assert_array_equal(back.shN, _fresh().shN)
    np.testing.assert_array_equal(back.extra["nx"], 0)


def test_writing_is_deterministic_and_normals_optional(tmp_path: Path) -> None:
    write_ply(_fresh(), tmp_path / "a.ply")
    write_ply(_fresh(), tmp_path / "b.ply")
    assert (tmp_path / "a.ply").read_bytes() == (tmp_path / "b.ply").read_bytes()
    write_ply(_fresh(), tmp_path / "c.ply", normals=False)
    assert len(read_ply_header(tmp_path / "c.ply").element("vertex").properties) == 59


def test_empty_scene_round_trips(tmp_path: Path) -> None:
    write_ply(_fresh(0), tmp_path / "e.ply")
    assert read_ply(tmp_path / "e.ply").count == 0


def test_file_can_be_overwritten_and_deleted_after_reading(tmp_path: Path) -> None:
    # On Windows a live memory map blocks both; the reader must not leave one.
    path = tmp_path / "s.ply"
    write_ply(_fresh(), path)
    scene = read_ply(path)
    write_ply(scene.take(np.arange(2)), path)
    assert read_ply(path).count == 2
    path.unlink()
    assert not list(tmp_path.iterdir())  # no leftover temporary file either


# Errors ------------------------------------------------------------------------------


def test_rejects_non_splat_and_compressed_files(tmp_path: Path) -> None:
    (tmp_path / "points.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n" + bytes(12)
    )
    with pytest.raises(PlyFormatError, match="missing f_dc_0"):
        read_ply(tmp_path / "points.ply")
    (tmp_path / "c.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property uint packed_position\nend_header\n" + bytes(4)
    )
    with pytest.raises(PlyFormatError, match="compressed"):
        read_ply(tmp_path / "c.ply")
    (tmp_path / "n.ply").write_bytes(b"not a ply")
    with pytest.raises(PlyFormatError, match="magic"):
        read_ply(tmp_path / "n.ply")


def test_rejects_truncated_and_list_properties(tmp_path: Path) -> None:
    names = inria_names(0)
    data = inria_bytes(random_values(4, len(names)), names)
    (tmp_path / "t.ply").write_bytes(data[:-5])
    with pytest.raises(PlyFormatError, match="truncated"):
        read_ply(tmp_path / "t.ply")
    (tmp_path / "l.ply").write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement face 1\n"
        b"property list uchar int vertex_indices\nend_header\n"
    )
    with pytest.raises(PlyFormatError, match="list property"):
        read_ply(tmp_path / "l.ply")


# Performance -----------------------------------------------------------------------------


@pytest.mark.slow
def test_reading_three_million_splats_is_fast(tmp_path: Path) -> None:
    n, names = 3_000_000, inria_names(3)
    path = tmp_path / "big.ply"
    header = inria_bytes(np.zeros((0, 62), np.float32), names).replace(
        b"element vertex 0", f"element vertex {n}".encode()
    )
    with path.open("wb") as f:
        f.write(header)
        chunk = random_values(250_000, 62)
        for _ in range(n // len(chunk)):
            f.write(chunk.tobytes())
    # Time the parser, not the disk: warm the page cache first. A cold read
    # adds whatever the disk costs (about 3 s for 744 MB at 250 MB/s).
    with path.open("rb") as f:
        while f.read(1 << 24):
            pass
    start = time.perf_counter()
    scene = read_ply(path)
    elapsed = time.perf_counter() - start
    assert scene.count == n
    assert elapsed < 5.0, f"parsing {n} splats took {elapsed:.2f} s"
