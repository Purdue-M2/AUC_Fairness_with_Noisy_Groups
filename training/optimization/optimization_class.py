import torch

"""Helper functions for performing constrained optimization."""

def project_multipliers_wrt_euclidean_norm(multipliers, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors with nonnegative elements that
    sum to at most "radius".
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto the
        feasible region w.r.t. the Euclidean norm.
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    if not torch.is_floating_point(multipliers):
        raise TypeError("multipliers must have a floating-point dtype")
    multipliers_dims = multipliers.shape
    if multipliers_dims is None:
        raise ValueError("multipliers must have a known rank")
    if len(multipliers_dims) != 1:
        raise ValueError("multipliers must be rank 1 (it is rank %d)" % len(multipliers_dims))
    dimension = multipliers_dims[0]
    if dimension is None:
        raise ValueError("multipliers must have a fully-known shape")

    iteration = 0
    inactive = torch.ones_like(multipliers, dtype=multipliers.dtype)
    old_inactive = torch.zeros_like(multipliers, dtype=multipliers.dtype)

    while True:
        iteration += 1
        scale = min(0.0, (radius - torch.sum(multipliers)).item() /
                    max(1.0, torch.sum(inactive)).item())
        multipliers = multipliers + (scale * inactive)
        new_inactive = (multipliers > 0).to(multipliers.dtype)
        multipliers = multipliers * new_inactive

        not_done = (iteration < dimension)
        not_converged = torch.any(inactive != old_inactive)

        if not (not_done and not_converged):
            break

        old_inactive = inactive
        inactive = new_inactive

    return multipliers

def project_multipliers_wrt_euclidean_norm_soft(multipliers, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors with nonnegative elements that
    sum to at most "radius".
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto the
        feasible region w.r.t. the Euclidean norm.
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    if not torch.is_floating_point(multipliers):
        raise TypeError("multipliers must have a floating-point dtype")
    multipliers_dims = multipliers.shape
    if multipliers_dims is None:
        raise ValueError("multipliers must have a known rank")
    if len(multipliers_dims) != 1:
        raise ValueError("multipliers must be rank 1 (it is rank %d)" % len(multipliers_dims))
    dimension = multipliers_dims[0]
    if dimension is None:
        raise ValueError("multipliers must have a fully-known shape")

    iteration = 0
    inactive = torch.ones_like(multipliers, dtype=multipliers.dtype)
    old_inactive = torch.zeros_like(multipliers, dtype=multipliers.dtype)

    while True:
        iteration += 1
        scale = min(0.0, (radius - torch.sum(multipliers)) /
                    max(1.0, torch.sum(inactive)))
        multipliers = multipliers + (scale * inactive)
        new_inactive = (multipliers > 0).to(multipliers.dtype)
        multipliers = multipliers * new_inactive

        not_done = (iteration < dimension)
        not_converged = torch.any(inactive != old_inactive)

        if not (not_done and not_converged):
            break

        old_inactive = inactive
        inactive = new_inactive

    return multipliers

def project_multipliers_wrt_euclidean_norm_handlefloat(multipliers, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors with nonnegative elements that
    sum to at most "radius".
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto the
        feasible region w.r.t. the Euclidean norm.
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    if not torch.is_floating_point(multipliers):
        raise TypeError("multipliers must have a floating-point dtype")
    multipliers_dims = multipliers.shape
    if multipliers_dims is None:
        raise ValueError("multipliers must have a known rank")
    if len(multipliers_dims) != 1:
        raise ValueError("multipliers must be rank 1 (it is rank %d)" % len(multipliers_dims))
    dimension = multipliers_dims[0]
    if dimension is None:
        raise ValueError("multipliers must have a fully-known shape")

    iteration = 0
    inactive = torch.ones_like(multipliers, dtype=multipliers.dtype)
    old_inactive = torch.zeros_like(multipliers, dtype=multipliers.dtype)

    while True:
        iteration += 1
        scale = min(0.0, (radius - torch.sum(multipliers)) /
                    max(1.0, torch.sum(inactive)))
        multipliers = multipliers + (scale * inactive)
        new_inactive = (multipliers > 0).to(multipliers.dtype)
        multipliers = multipliers * new_inactive

        not_done = (iteration < dimension)
        not_converged = torch.any(inactive != old_inactive)

        if not (not_done and not_converged):
            break

        old_inactive = inactive
        inactive = new_inactive

    return multipliers

def project_multipliers_to_L1_ball(multipliers, center, radius):
    """Projects its argument onto the feasible region.
    The feasible region is the set of all vectors in the L1 ball with the given center multipliers and given radius.
    Args:
        multipliers: rank-1 `Tensor`, the Lagrange multipliers to project.
        radius: float, the radius of the feasible region.
        center: rank-1 `Tensor`, the Lagrange multipliers as the center.
    Returns:
        The rank-1 `Tensor` that results from projecting "multipliers" onto a L1 norm ball w.r.t. the Euclidean norm.
        The returned rank-1 `Tensor` is in a simplex.
    Raises:
        TypeError: if the "multipliers" `Tensor` is not floating-point.
        ValueError: if the "multipliers" `Tensor` does not have a fully-known shape,
          or is not one-dimensional.
    """
    assert radius >= 0
    # Compute the offset from the center and the distance
    offset = multipliers - center
    # print()
    dist = torch.abs(offset)
    
    # Project multipliers on the simplex
    new_dist = project_multipliers_wrt_euclidean_norm(dist, radius=radius)
    signs = torch.sign(offset)
    new_offset = signs * new_dist
    projection = center + new_offset
    projection = torch.clamp(projection, min=0.0)
    
    return projection
    
import torch

def project_by_dykstra(weights, project_groups_fn, project_simplex_fn, num_iterations=20):
    """Applies Dykstra's projection algorithm for monotonicity/trust constraints."""
    if num_iterations == 0:
        return weights

    def body(iteration, weights, last_change):
        """Body of the loop for Dykstra's projection algorithm."""
        last_change = last_change.copy()
    
        # Project onto group linear equality constraints
        rolled_back_weights = weights - last_change["Aw=b"]
        weights = project_groups_fn(rolled_back_weights)
        last_change["Aw=b"] = weights - rolled_back_weights

        # Project onto simplex linear equality constraints
        rolled_back_weights = weights - last_change["1w=1"]
        weights = project_simplex_fn(rolled_back_weights)
        last_change["1w=1"] = weights - rolled_back_weights

        # Project onto nonnegativity constraints
        rolled_back_weights = weights - last_change["w>=0"]
        weights = torch.relu(weights)
        last_change["w>=0"] = weights - rolled_back_weights

        return iteration + 1, weights, last_change

    def cond(iteration, weights, last_change):
        return iteration < num_iterations

    # Initialize last_change
    last_change = {"Aw=b": torch.zeros_like(weights), "1w=1": torch.zeros_like(weights), "w>=0": torch.zeros_like(weights)}

    # Apply Dykstra's algorithm
    iteration = 0
    while cond(iteration, weights, last_change):
        iteration, weights, last_change = body(iteration, weights, last_change)

    return weights
