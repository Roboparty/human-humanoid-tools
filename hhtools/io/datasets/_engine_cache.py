"""Small helper that caches :class:`SmplxEngine` instances across dataset adapter calls.

Constructing the engine takes O(1-3s) because it needs to load weights, so we re-use it
between adjacent calls for the same ``(family, gender, num_betas)`` combination.  This module
is intentionally lazy so that ``import hhtools.io.datasets`` does not pull in ``smplx`` or
``torch``.
"""

from __future__ import annotations

from functools import lru_cache

from hhtools.bodymodels.params import SmplMotionParams


@lru_cache(maxsize=8)
def _get_cached_engine(family: str, gender: str, num_betas: int):  # noqa: ANN202
    from hhtools.bodymodels.engine import SmplxEngine

    return SmplxEngine(family, gender=gender, num_betas=num_betas)


def engine_for_params(params: SmplMotionParams):  # noqa: ANN201
    """Return a cached :class:`SmplxEngine` matching the SMPL family declared by *params*."""
    betas_dim = int(params.betas.reshape(-1).shape[0])
    num_betas = min(betas_dim, 10)  # smplx defaults
    return _get_cached_engine(params.surface_model, params.gender, num_betas)


def motion_from_params(
    params: SmplMotionParams,
    *,
    name: str,
    source_format: str | None = None,
    return_mesh: bool = False,
    progress_callback=None,
):
    """Use the licensed engine when available, otherwise return a skeleton preview."""
    try:
        engine = engine_for_params(params)
    except (FileNotFoundError, ImportError, OSError) as error:
        from hhtools.bodymodels.fallback import motion_from_fallback

        return motion_from_fallback(
            params,
            name=name,
            source_format=source_format,
            reason=error,
            progress_callback=progress_callback,
        )
    return engine.to_motion(
        params,
        name=name,
        source_format=source_format,
        return_mesh=return_mesh,
        progress_callback=progress_callback,
    )


def clear_engine_cache() -> None:
    _get_cached_engine.cache_clear()


__all__ = ["clear_engine_cache", "engine_for_params", "motion_from_params"]
