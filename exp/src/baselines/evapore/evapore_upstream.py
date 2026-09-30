"""Import shim for the upstream EVAPORE code, and their candidate generator.

Upstream: https://github.com/oscarmorand/EVAPORE, branch **shapeMI**
(the ShapeMI / paper version), commit
``05af7d028bd6f3cf9772100f8b344e20cfc8783d``, **GPL-3.0**.  Cloned read-only to
``exp/third_party/EVAPORE``.  At the time of cloning ``main`` and ``shapeMI``
have identical trees.

GPL-3.0 note
------------
Their code is *imported*, never copied into this project.  Anything that links
their modules is covered by the GPL; keep that boundary at this package.

Three things stand between their ``src/`` and a Windows + torch 2.6 run, all
handled here so the adapter can just call :func:`upstream`:

S1. ``utils/path.py`` raises at **import time** unless ``EVAPORE_DATA_DIR`` and
    ``EVAPORE_CHECKPOINTS_DIR`` are set.  We point them at scratch paths.
S2. ``pylena`` (the Dahu minimal-path backend) publishes **manylinux wheels
    only** -- there is no Windows build -- and the ``__init__.py`` of
    ``utils.reconstruction`` imports the Dahu module eagerly, so *any* import
    from that package fails.  The notebooks never use Dahu (04/05/06 hard-code
    ``EuclideanPathReconstructionMethod``), so we install a stub module.
S3. ``path_neural_networks/models/__init__.py`` imports their Lightning module,
    and ``pytorch_lightning`` / ``torchmetrics`` are not in this environment.
    Their *components* (path sampler, encoder, classifier, U-Net) are plain
    ``torch.nn.Module``\\ s and use neither, so we stub both packages
    (``LightningModule = nn.Module``) purely to let the package ``__init__``
    execute.  The stubs are never used for training -- our adapter drives the
    real modules with a plain PyTorch loop.

None of this patches a file in ``third_party``.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["EVAPORE_ROOT", "EVAPORE_COMMIT", "EVAPORE_BRANCH", "EVAPORE_LICENSE",
           "ensure_upstream", "upstream", "generate_candidates"]

EVAPORE_ROOT = (Path(__file__).resolve().parents[3] / "third_party" / "EVAPORE")
EVAPORE_SRC = EVAPORE_ROOT / "src"
EVAPORE_BRANCH = "shapeMI"
EVAPORE_COMMIT = "05af7d028bd6f3cf9772100f8b344e20cfc8783d"
EVAPORE_LICENSE = "GPL-3.0"

_READY = False
_MODS: Dict[str, Any] = {}


class _Stub(types.ModuleType):
    """A module whose every attribute is fabricated on demand.

    A ``Capitalised`` attribute becomes a permissive ``nn.Module`` subclass, so
    upstream code that *subclasses* it (``class BinaryPRAUC(Metric)``,
    ``class ReducedPipelineLitModule(pl.LightningModule)``) still imports; a
    lowercase attribute becomes a no-op callable.
    """

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        import torch
        import torch.nn as nn

        if k[:1].isupper():
            def _init(self, *a, **kw):
                nn.Module.__init__(self)

            obj = type(k, (nn.Module,), {
                "__init__": _init,
                "update": lambda self, *a, **kw: None,
                "compute": lambda self: torch.tensor(0.0),
                "reset": lambda self, *a, **kw: None,
                "forward": lambda self, *a, **kw: torch.tensor(0.0),
            })
        else:
            # a lowercase attribute may be either a function
            # (``torchmetrics.functional.precision_recall_curve``) or a
            # subpackage (``pylena.morpho``), so hand back something that is
            # both callable and further attribute-addressable.
            obj = _Stub(f"{self.__name__}.{k}")
            obj.__path__ = []
            _give_spec(obj)
            sys.modules.setdefault(obj.__name__, obj)
        setattr(self, k, obj)
        return obj

    def __call__(self, *a, **kw):
        return None


def _give_spec(mod: types.ModuleType) -> None:
    """Attach a real ``ModuleSpec`` to a stub inserted straight into
    ``sys.modules`` (bypassing the import machinery, which would otherwise do
    this for us).

    Without it ``mod.__spec__`` stays the bare ``None`` that
    ``types.ModuleType`` initialises, and some torch versions'
    ``torch._dynamo.trace_rules`` (imported transitively by
    ``torchvision.ops`` on import) call ``importlib.util.find_spec`` on every
    module name it knows about; CPython's ``find_spec`` raises
    ``ValueError(f'{name}.__spec__ is None')`` for exactly that case. This
    was never hit on the Windows dev box (older torchvision import order),
    but reproduces on Linux/torch 2.5.x -- see S4 remote-offload notes.
    """
    import importlib.machinery

    mod.__spec__ = importlib.machinery.ModuleSpec(
        mod.__name__, _StubLoader(), is_package=True)


class _StubLoader:
    """Materialise any submodule of a stubbed package as a :class:`_Stub`."""

    def create_module(self, spec):
        m = _Stub(spec.name)
        m.__path__ = []          # mark as a package so submodules resolve
        return m

    def exec_module(self, module):
        return None


class _StubFinder:
    """Meta-path finder answering for whole stubbed package trees."""

    def __init__(self, prefixes):
        self.prefixes = tuple(prefixes)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root not in self.prefixes:
            return None
        import importlib.machinery
        return importlib.machinery.ModuleSpec(fullname, _StubLoader(),
                                              is_package=True)


def _install_stubs() -> None:
    """Install stand-ins for the four packages upstream imports but we lack.

    ``pylena`` has no Windows wheel (S2); ``pytorch_lightning``,
    ``torchmetrics``, ``hydra``/``omegaconf`` are pulled in only by the package
    ``__init__`` chain that leads to their Lightning module (S3).  The
    components we actually use are plain ``torch.nn.Module``\ s that import
    none of them.  A meta-path finder covers every submodule at once.
    """
    import torch.nn as nn

    missing = []
    for name in ("pylena", "pytorch_lightning", "torchmetrics", "hydra",
                 "omegaconf"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except Exception:
            missing.append(name)

    if not missing:
        return

    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.append(_StubFinder(missing))

    for name in missing:
        mod = _Stub(name)
        mod.__path__ = []
        _give_spec(mod)
        sys.modules[name] = mod

    # the one binding that must be exact: their LitModules subclass this, and
    # we want them to be ordinary nn.Modules so state_dicts stay compatible.
    if "pytorch_lightning" in missing:
        sys.modules["pytorch_lightning"].LightningModule = nn.Module
        sys.modules["pytorch_lightning"].LightningDataModule = object
    if "torchmetrics" in missing:
        sys.modules["torchmetrics"].Metric = nn.Module


def ensure_upstream(data_dir: Optional[str] = None,
                    ckpt_dir: Optional[str] = None) -> Dict[str, Any]:
    """Make the upstream package importable and return the pieces we use."""
    global _READY
    if _READY:
        return _MODS

    if not EVAPORE_SRC.is_dir():
        raise RuntimeError(
            f"EVAPORE not cloned at {EVAPORE_ROOT}.  Run:\n"
            f"  git clone https://github.com/oscarmorand/EVAPORE.git "
            f"{EVAPORE_ROOT} && git -C {EVAPORE_ROOT} checkout {EVAPORE_BRANCH}")

    scratch = EVAPORE_ROOT.parent.parent / "runs" / "repair" / "_evapore_scratch"
    os.environ.setdefault("EVAPORE_DATA_DIR", str(data_dir or scratch / "data"))   # S1
    os.environ.setdefault("EVAPORE_CHECKPOINTS_DIR",
                          str(ckpt_dir or EVAPORE_ROOT / "checkpoints"))
    _install_stubs()

    if str(EVAPORE_SRC) not in sys.path:
        sys.path.insert(0, str(EVAPORE_SRC))

    import importlib

    from graph.graph_creation import img_to_graph
    from graph.graph_oversampling import OversampleNodesTransform
    from graph.graph_wrapper import GraphWrapper
    from path_neural_networks.utils.paths_creation import (
        cut_mask_from_negative_edges_for_all, get_query_edges,
    )
    euc = importlib.import_module(
        "utils.reconstruction.path_reconstruction.euclidean_path_reconstruction")
    rad = importlib.import_module(
        "utils.reconstruction.radius_reconstruction.smallest_radius_reconstruction")

    _MODS.update(
        img_to_graph=img_to_graph,
        OversampleNodesTransform=OversampleNodesTransform,
        GraphWrapper=GraphWrapper,
        get_query_edges=get_query_edges,
        cut_mask_from_negative_edges_for_all=cut_mask_from_negative_edges_for_all,
        EuclideanPathReconstructionMethod=euc.EuclideanPathReconstructionMethod,
        SmallestRadiusReconstructionMethod=rad.SmallestRadiusReconstructionMethod,
    )
    _READY = True
    return _MODS


def upstream() -> Dict[str, Any]:
    return ensure_upstream()


def load_components():
    """Their nn.Module components (imported, not reimplemented).

    ``MultiScaleSquarePathSampling``, ``SamplingMaxAggregation``,
    ``ConvMaxPoolingPathEncoder``, ``FCNPathClassifier``, ``UNet``.
    """
    ensure_upstream()
    import importlib

    ps = importlib.import_module("path_neural_networks.models.path_samplers")
    pe = importlib.import_module(
        "path_neural_networks.models.path_encoders.conv_pooling_encoder")
    pc = importlib.import_module(
        "path_neural_networks.models.path_classifiers.fcn_path_classifier")
    un = importlib.import_module("image_segmentation.models.unet")

    enc = (getattr(pe, "ConvMaxPoolingPathEncoder", None)
           or getattr(pe, "ConvPoolingEncoder"))
    return dict(
        MultiScaleSquarePathSampling=ps.MultiScaleSquarePathSampling,
        SamplingMaxAggregation=ps.SamplingMaxAggregation,
        PathEncoder=enc,
        FCNPathClassifier=pc.FCNPathClassifier,
        UNet=un.UNet,
    )


# --------------------------------------------------------------------------
# their inference-time candidate generator
# --------------------------------------------------------------------------


def generate_candidates(
    mask: np.ndarray,
    max_dist: float = 100.0,
    oversampling_max_dist: float = 50.0,
    n_closest: int = 5,
    closing_radius: int = 1,
    clean: bool = True,
) -> Dict[str, Any]:
    """Candidate reconnection paths for a binary mask, **their** way.

    A direct transcription of ``process_case`` from their
    ``notebooks/06_test_path_classification_model.ipynb`` (cell 33) -- the only
    place upstream wires the importable pieces into a mask-only, GT-free
    generator; it exists in no ``.py`` file, so it has to be restated here,
    calling their functions at every step:

    1. ``img_to_graph`` -- skeletonise, optional closing, ``skan`` pixel graph
       collapsed to a branch graph with ``pos`` / ``radius`` per node;
    2. ``OversampleNodesTransform`` -- insert nodes every
       ``oversampling_max_dist`` px so endpoints are not only at branch points;
    3. ``get_query_edges`` -- for every degree-1 endpoint of every non-largest
       component, the ``n_closest`` nearest nodes in *other* components within
       ``max_dist`` (their nearest-endpoint heuristic; no A*);
    4. ``EuclideanPathReconstructionMethod`` -- rasterise a straight line
       between each pair;
    5. ``cut_mask_from_negative_edges_for_all`` -- clip each line to the part
       that actually lies **off** the mask, i.e. the gap.

    Returns ``dict(edges, path_centerlines, full_path_centerlines, graph,
    nodes_radius)`` with centerlines as lists of ``[row, col]`` ints.
    ``nodes_radius`` maps graph node id -> skan radius, which is what their
    ``RadiusReconstructionMethod`` family consumes when painting the tube.
    """
    u = ensure_upstream()
    m = np.asarray(mask).squeeze() > 0

    graph = u["img_to_graph"](m, clean=clean, closing_radius=closing_radius,
                              return_pixel_graph=False)
    gw = u["OversampleNodesTransform"](oversampling_max_dist,
                                       remove_original_edges=True)(
        u["GraphWrapper"](graph))
    G = gw.get_graph()

    qe = u["get_query_edges"](G, n_closest=int(n_closest), max_dist=float(max_dist))
    nodes_radius = {int(i): float(d["radius"]) for i, d in G.nodes(data=True)
                    if "radius" in d}

    if qe.numel() == 0:
        return dict(edges=[], path_centerlines=[], full_path_centerlines=[],
                    graph=G, nodes_radius=nodes_radius)

    edges = qe.t().tolist()
    paths = u["EuclideanPathReconstructionMethod"]().reconstruct(
        map=None, graph=G, new_edges=qe)
    paths = [[[int(x), int(y)] for (x, y) in p] for p in paths]
    res = u["cut_mask_from_negative_edges_for_all"](
        paths, m, edges=edges, return_old_centerlines=True)
    return dict(edges=res["edges"], path_centerlines=res["new_centerlines"],
                full_path_centerlines=res["old_centerlines"], graph=G,
                nodes_radius=nodes_radius)
