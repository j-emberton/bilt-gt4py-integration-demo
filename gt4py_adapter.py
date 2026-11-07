"""Small GT4Py Next adapter for Bilt's aligned two-dimensional tensor values.

This deliberately uses Bilt's private evaluation context to recover axis order.
It belongs in this demo, not in Bilt core or a stable integration package.
"""

import gt4py.next as gtx
import numpy as np
from bilt import dim
from bilt.core._evaluation_context import current_evaluation

x = dim("x")
y = dim("y")

I = gtx.Dimension("I")  # noqa: E741
J = gtx.Dimension("J")


@gtx.field_operator
def five_point_laplacian(
    source: gtx.Field[[I, J], gtx.float64],  # pyrefly: ignore[invalid-param-spec, not-a-type]  # ty: ignore[invalid-type-form]
    inv_dx2: gtx.float64,
    inv_dy2: gtx.float64,
) -> gtx.Field[[I, J], gtx.float64]:  # pyrefly: ignore[invalid-param-spec, not-a-type]  # ty: ignore[invalid-type-form]
    """Compute the discrete Laplacian over the physical grid cells."""
    return (source(I - 1) - 2.0 * source + source(I + 1)) * inv_dx2 + (  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]
        source(J - 1) - 2.0 * source + source(J + 1)  # pyrefly: ignore[unsupported-operation]  # ty: ignore[unsupported-operator]
    ) * inv_dy2


def periodic_laplacian(values: np.ndarray, inv_dx2: float, inv_dy2: float) -> np.ndarray:
    """Evaluate a periodic GT4Py stencil and restore the original Bilt axis order."""
    domain = current_evaluation().canonical_order.dims
    spatial_dims = (x, y)
    if len(domain) != len(spatial_dims) or set(domain) != set(spatial_dims):
        msg = "This demonstration only supports an (x, y) Cartesian field."
        raise ValueError(msg)

    # Transform inputs are arrays aligned to the current Bilt evaluation order.
    to_gt4py = (domain.index(x), domain.index(y))
    physical = np.asarray(values).transpose(to_gt4py)

    # The one-cell periodic halo is an implementation detail, not model state.
    padded = np.pad(physical, pad_width=1, mode="wrap")
    source = gtx.as_field([I, J], padded)  # pyrefly: ignore[bad-argument-type]
    output = gtx.as_field([I, J], np.zeros_like(padded))  # pyrefly: ignore[bad-argument-type]
    five_point_laplacian(
        source,
        inv_dx2,
        inv_dy2,
        out=output[1:-1, 1:-1],
        offset_provider={},
    )

    # Return an ordinary aligned array; Bilt wraps it back into a Tensor.
    from_gt4py = tuple(to_gt4py.index(axis) for axis in range(2))
    return np.asarray(output[1:-1, 1:-1].asnumpy()).transpose(from_gt4py)
