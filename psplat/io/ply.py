"""Read and write 3D Gaussian Splat PLY files.

The standard layout, written by the Inria reference code, gsplat, msplat and
most trainers, is one ``vertex`` element with 62 float properties::

    x y z  nx ny nz  f_dc_0..2  f_rest_0..44  opacity  scale_0..2  rot_0..3

``f_rest`` is stored channel-major: all red coefficients, then green, then
blue. Degree-0 to degree-2 scenes simply have fewer ``f_rest`` properties.

Reading memory-maps the file, copies it out in one sequential pass of
row chunks, and releases the mapping, so no file handle or mapping outlives
the call. That matters on Windows, where a live mapping blocks
overwriting or deleting the file. Any property that is not one of the splat
attributes is kept in ``Scene.extra`` in file order, and the source layout is
remembered, so writing an unmodified scene reproduces the file byte for byte.
Writing is deterministic: no timestamps, no host details.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray

from psplat.io.scene import SH_REST_BY_DEGREE, PlyLayout, Scene

# PLY scalar type keyword -> numpy type code, without byte order.
PLY_TYPES: dict[str, str] = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}  # fmt: skip
# numpy type code -> the PLY keyword psplat writes for it.
_KEYWORD_FOR_CODE = {
    "i1": "char", "u1": "uchar", "i2": "short", "u2": "ushort",
    "i4": "int", "u4": "uint", "f4": "float", "f8": "double",
}  # fmt: skip
_BYTE_ORDER = {"binary_little_endian": "<", "binary_big_endian": ">", "ascii": "<"}

_POSITION = ("x", "y", "z")
_NORMALS = ("nx", "ny", "nz")
_DC = ("f_dc_0", "f_dc_1", "f_dc_2")
_SCALE = ("scale_0", "scale_1", "scale_2")
_ROT = ("rot_0", "rot_1", "rot_2", "rot_3")
_REQUIRED = (*_POSITION, *_DC, "opacity", *_SCALE, *_ROT)

# Rows copied per chunk when reading or writing, so a multi-million splat
# scene never needs a second full-size buffer.
_READ_CHUNK = 1 << 18
_WRITE_CHUNK = 1 << 18


class PlyFormatError(ValueError):
    """A PLY file could not be read as a Gaussian splat scene."""


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: tuple[tuple[str, str], ...]  # (name, PLY type keyword)


@dataclass(frozen=True)
class PlyHeader:
    format: str
    elements: tuple[PlyElement, ...]
    comments: tuple[str, ...]
    obj_info: tuple[str, ...]
    size: int  # bytes up to and including the end_header line

    def element(self, name: str) -> PlyElement:
        for el in self.elements:
            if el.name == name:
                return el
        raise PlyFormatError(f"PLY has no {name!r} element")


def read_header(f: BinaryIO) -> PlyHeader:
    """Parse a PLY header from the start of ``f``."""
    if f.readline().rstrip(b"\r\n") != b"ply":
        raise PlyFormatError("not a PLY file (missing 'ply' magic line)")
    fmt = ""
    comments: list[str] = []
    obj_info: list[str] = []
    elements: list[PlyElement] = []
    current: tuple[str, int, list[tuple[str, str]]] | None = None
    while True:
        raw = f.readline()
        if not raw:
            raise PlyFormatError("PLY header has no end_header line")
        line = raw.decode("ascii", errors="replace").rstrip("\r\n")
        keyword, _, rest = line.partition(" ")
        if keyword == "format":
            fmt, _, version = rest.partition(" ")
            if fmt not in _BYTE_ORDER or version.strip() != "1.0":
                raise PlyFormatError(f"unsupported PLY format {rest!r}")
        elif keyword == "comment":
            comments.append(rest)
        elif keyword == "obj_info":
            obj_info.append(rest)
        elif keyword == "element":
            if current is not None:
                elements.append(PlyElement(current[0], current[1], tuple(current[2])))
            name, _, count = rest.partition(" ")
            current = (name, int(count), [])
        elif keyword == "property":
            if current is None:
                raise PlyFormatError("PLY property declared before any element")
            parts = rest.split()
            if parts[0] == "list":
                raise PlyFormatError(
                    f"list property {parts[-1]!r} in element {current[0]!r} is not supported; "
                    "Gaussian splat PLYs use scalar properties only"
                )
            if len(parts) != 2 or parts[0] not in PLY_TYPES:
                raise PlyFormatError(f"unsupported PLY property line {line!r}")
            current[2].append((parts[1], parts[0]))
        elif keyword == "end_header":
            break
        elif line.strip():
            raise PlyFormatError(f"unexpected PLY header line {line!r}")
    if current is not None:
        elements.append(PlyElement(current[0], current[1], tuple(current[2])))
    if not fmt:
        raise PlyFormatError("PLY header has no format line")
    return PlyHeader(fmt, tuple(elements), tuple(comments), tuple(obj_info), f.tell())


def read_ply_header(path: str | Path) -> PlyHeader:
    with Path(path).open("rb") as f:
        return read_header(f)


def _dtype(properties: Iterable[tuple[str, str]], order: str) -> np.dtype[np.void]:
    return np.dtype([(name, order + PLY_TYPES[kind]) for name, kind in properties])


def read_ply(path: str | Path) -> Scene:
    """Read a Gaussian splat PLY into a :class:`Scene`."""
    path = Path(path)
    with path.open("rb") as f:
        header = read_header(f)
    vertex = header.element("vertex")
    names = [name for name, _ in vertex.properties]
    if "packed_position" in names:
        raise PlyFormatError(
            f"{path.name} is a compressed PLY (SuperSplat). Decompress it first, for example "
            "with: npx @playcanvas/splat-transform in.compressed.ply out.ply"
        )
    missing = [name for name in _REQUIRED if name not in names]
    if missing:
        raise PlyFormatError(
            f"{path.name} is not a Gaussian splat PLY; missing {', '.join(missing)}"
        )

    # Everything before the vertex element must be skipped by size.
    offset = header.size
    for el in header.elements:
        if el.name == "vertex":
            break
        if header.format == "ascii":
            raise PlyFormatError("ASCII PLY with elements before 'vertex' is not supported")
        offset += el.count * _dtype(el.properties, "<").itemsize
    others = [el.name for el in header.elements if el.name != "vertex"]
    if others:
        warnings.warn(
            f"ignoring PLY element(s) {', '.join(others)}; only 'vertex' is read", stacklevel=2
        )

    return _read_scene(path, header, vertex, offset)


def _read_body(
    path: Path,
    header: PlyHeader,
    vertex: PlyElement,
    offset: int,
    float_groups: dict[str, list[str]],
    raw_columns: list[str],
) -> tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.generic]]]:
    """Read the vertex body in one sequential pass.

    ``float_groups`` maps a group name to property names; each comes back as
    an ``(n, len(names))`` float32 array. ``raw_columns`` come back one by one
    in their file dtype. Rows are copied in chunks, so memory beyond the
    outputs stays bounded and the file is read front to back exactly once,
    which is what makes multi-million splat files fast to load.
    """
    n = vertex.count
    names = [name for name, _ in vertex.properties]
    index = {name: i for i, name in enumerate(names)}
    kinds = {name: PLY_TYPES[kind] for name, kind in vertex.properties}
    groups = {g: np.empty((n, len(cols)), np.float32) for g, cols in float_groups.items()}
    raws = {name: np.empty(n, kinds[name]) for name in raw_columns}
    if n == 0:
        return groups, raws

    if header.format == "ascii":
        with path.open("rb") as f:
            f.seek(offset)
            text = f.read().decode("ascii")
        table = np.loadtxt(io.StringIO(text), dtype=np.float64, max_rows=n, ndmin=2)
        if table.shape != (n, len(names)):
            raise PlyFormatError(f"ASCII PLY body has shape {table.shape}")
        for g, cols in float_groups.items():
            groups[g][:] = table[:, [index[c] for c in cols]]
        for name in raw_columns:
            raws[name][:] = table[:, index[name]]
        return groups, raws

    order = _BYTE_ORDER[header.format]
    record = _dtype(vertex.properties, order)
    expected = offset + n * record.itemsize
    if path.stat().st_size < expected:
        raise PlyFormatError(f"{path.name} is truncated: expected at least {expected} bytes")

    dense = set(kinds.values()) == {"f4"}
    mm: np.memmap[tuple[int, ...], np.dtype[np.generic]]
    if dense:
        # The common case: every property is a 4-byte float, so the body is
        # an (n, p) matrix and a group is a column selection.
        mm = np.memmap(path, dtype=order + "f4", mode="r", offset=offset, shape=(n, len(names)))
    else:
        mm = np.memmap(path, dtype=record, mode="r", offset=offset, shape=(n,))
    selectors = {g: _columns([index[c] for c in cols]) for g, cols in float_groups.items()}
    try:
        for start in range(0, n, _READ_CHUNK):
            stop = min(start + _READ_CHUNK, n)
            block: Any = mm[start:stop]  # rows as a view; structured or dense
            for g, cols in float_groups.items():
                if dense:
                    groups[g][start:stop] = block[:, selectors[g]]
                else:
                    for j, c in enumerate(cols):
                        groups[g][start:stop, j] = block[c]
            for name in raw_columns:
                raws[name][start:stop] = block[:, index[name]] if dense else block[name]
            del block
    finally:
        del mm
    return groups, raws


def _columns(indices: list[int]) -> slice | list[int]:
    """A slice when the columns are contiguous (much faster to copy), else a list."""
    if indices and indices == list(range(indices[0], indices[0] + len(indices))):
        return slice(indices[0], indices[0] + len(indices))
    return indices


def _rest_names(names: Iterable[str]) -> list[str]:
    rest = [name for name in names if name.startswith("f_rest_")]
    expected = [f"f_rest_{i}" for i in range(len(rest))]
    if sorted(rest, key=lambda s: int(s[7:]) if s[7:].isdigit() else -1) != expected:
        raise PlyFormatError("f_rest properties must be f_rest_0 .. f_rest_{M-1}")
    if len(rest) % 3 or len(rest) // 3 not in SH_REST_BY_DEGREE.values():
        raise PlyFormatError(f"{len(rest)} f_rest properties is not a valid SH degree")
    return expected


def _read_scene(path: Path, header: PlyHeader, vertex: PlyElement, offset: int) -> Scene:
    names = [name for name, _ in vertex.properties]
    rest = _rest_names(names)
    known = {*_REQUIRED, *rest}
    float_groups = {
        "means": list(_POSITION),
        "scales": list(_SCALE),
        "quats": list(_ROT),
        "sh0": list(_DC),
        "opacity": ["opacity"],
        "rest": rest,
    }
    extras = [name for name in names if name not in known]
    groups, raws = _read_body(path, header, vertex, offset, float_groups, extras)
    n, k = vertex.count, len(rest) // 3
    # f_rest is channel-major on disk: (n, 3 channels, k coefficients) -> (n, k, 3).
    # Transposed in row chunks; one whole-array strided transpose is slow.
    rest_values = groups.pop("rest")
    shn = np.empty((n, k, 3), np.float32)
    for start in range(0, n if k else 0, _READ_CHUNK):
        stop = min(start + _READ_CHUNK, n)
        shn[start:stop] = rest_values[start:stop].reshape(-1, 3, k).transpose(0, 2, 1)
    del rest_values
    return Scene(
        means=groups["means"],
        scales=groups["scales"],
        quats=groups["quats"],
        opacities=groups["opacity"][:, 0].copy(),
        sh0=groups["sh0"],
        shN=shn,
        extra={name: raws[name] for name in extras},
        ply_layout=PlyLayout(vertex.properties, header.comments, header.obj_info),
    )


# Writing ----------------------------------------------------------------------


def canonical_properties(scene: Scene, *, normals: bool = True) -> list[tuple[str, str]]:
    """The standard 3DGS property order for ``scene``, with extras appended.

    ``normals`` includes ``nx ny nz`` (zeros unless the scene carries them);
    several importers expect the full 62-property Inria layout.
    """
    props = [(name, "float") for name in _POSITION]
    include_normals = normals or any(name in scene.extra for name in _NORMALS)
    if include_normals:
        props += [(name, _keyword(scene.extra.get(name), "float")) for name in _NORMALS]
    props += [(name, "float") for name in _DC]
    props += [(f"f_rest_{i}", "float") for i in range(scene.shN.shape[1] * 3)]
    props += [("opacity", "float")]
    props += [(name, "float") for name in (*_SCALE, *_ROT)]
    props += [
        (name, _keyword(values, "float"))
        for name, values in scene.extra.items()
        if name not in _NORMALS
    ]
    return props


def _keyword(values: NDArray[np.generic] | None, default: str) -> str:
    if values is None:
        return default
    code = values.dtype.kind + str(values.dtype.itemsize)
    if code not in _KEYWORD_FOR_CODE:
        raise PlyFormatError(f"cannot write dtype {values.dtype} to PLY")
    return _KEYWORD_FOR_CODE[code]


def _scene_columns(scene: Scene) -> dict[str, NDArray[np.generic]]:
    n, k = scene.count, scene.shN.shape[1]
    cols: dict[str, NDArray[np.generic]] = {}
    for i, name in enumerate(_POSITION):
        cols[name] = scene.means[:, i]
    for i, name in enumerate(_DC):
        cols[name] = scene.sh0[:, i]
    rest = scene.shN.transpose(0, 2, 1).reshape(n, 3 * k)
    for i in range(3 * k):
        cols[f"f_rest_{i}"] = rest[:, i]
    cols["opacity"] = scene.opacities
    for i, name in enumerate(_SCALE):
        cols[name] = scene.scales[:, i]
    for i, name in enumerate(_ROT):
        cols[name] = scene.quats[:, i]
    for name in _NORMALS:
        cols[name] = np.zeros(n, np.float32)
    cols.update(scene.extra)
    return cols


def _layout_fits(scene: Scene, src: PlyLayout) -> bool:
    """True if ``src`` names every attribute ``scene`` has, and nothing else.

    Normals are the one exception: the writer can always supply zeros.
    """
    names = [name for name, _ in src.properties]
    must = {*_REQUIRED, *(f"f_rest_{i}" for i in range(3 * scene.shN.shape[1])), *scene.extra}
    return len(set(names)) == len(names) and must <= set(names) <= must | set(_NORMALS)


def write_ply(
    scene: Scene,
    path: str | Path,
    *,
    layout: str = "source",
    comments: Iterable[str] | None = None,
    normals: bool = True,
) -> None:
    """Write ``scene`` as a binary little-endian Gaussian splat PLY.

    ``layout="source"`` reuses the property order, types and comments of the
    file the scene was read from, when that layout still fits the scene; this
    is what makes an unmodified round trip byte-identical. ``layout="canonical"``
    always writes the standard 3DGS order. ``comments`` replaces the header
    comments. Writes go to a temporary file that is renamed into place, so a
    failed write never leaves a half-written PLY behind.
    """
    if layout not in ("source", "canonical"):
        raise ValueError("layout must be 'source' or 'canonical'")
    columns = _scene_columns(scene)
    props = canonical_properties(scene, normals=normals)
    header_comments: tuple[str, ...] = ()
    obj_info: tuple[str, ...] = ()
    src = scene.ply_layout
    if layout == "source" and src is not None and _layout_fits(scene, src):
        props = list(src.properties)
        header_comments, obj_info = src.comments, src.obj_info
    if comments is not None:
        header_comments = tuple(comments)

    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in header_comments]
    lines += [f"obj_info {o}" for o in obj_info]
    lines.append(f"element vertex {scene.count}")
    lines += [f"property {kind} {name}" for name, kind in props]
    lines.append("end_header")
    header = ("\n".join(lines) + "\n").encode("ascii")

    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    all_float = all(PLY_TYPES[kind] == "f4" for _, kind in props)
    with tmp.open("wb") as f:
        f.write(header)
        for start in range(0, scene.count, _WRITE_CHUNK):
            stop = min(start + _WRITE_CHUNK, scene.count)
            if all_float:
                dense = np.empty((stop - start, len(props)), dtype="<f4")
                for i, (name, _) in enumerate(props):
                    dense[:, i] = columns[name][start:stop]
                f.write(dense.tobytes())
            else:
                records = np.empty(stop - start, dtype=_dtype(props, "<"))
                for name, _ in props:
                    records[name] = columns[name][start:stop]
                f.write(records.tobytes())
    tmp.replace(path)
