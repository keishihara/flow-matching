# adapted from https://github.com/facebookresearch/flow_matching

from abc import ABC
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn
from torchdiffeq import odeint


def gradient(
    output: Tensor,
    x: Tensor,
    grad_outputs: Tensor | None = None,
    create_graph: bool = False,
) -> Tensor:
    """
    Compute the gradient of the inner product of output and grad_outputs w.r.t :math:`x`.

    Args:
        output (Tensor): [N, D] Output of the function.
        x (Tensor): [N, d_1, d_2, ... ] input
        grad_outputs (Optional[Tensor]): [N, D] Gradient of outputs, if `None`,
            then will use a tensor of ones
        create_graph (bool): If True, graph of the derivative will be constructed, allowing
            to compute higher order derivative products. Defaults to False.
    Returns:
        Tensor: [N, d_1, d_2, ... ]. the gradient w.r.t x.
    """

    if grad_outputs is None:
        grad_outputs = torch.ones_like(output).detach()
    grad = torch.autograd.grad(output, x, grad_outputs=grad_outputs, create_graph=create_graph)[0]
    return grad


class ModelWrapper(ABC, nn.Module):
    """
    This class is used to wrap around another model, adding custom forward pass logic.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        r"""
        This method defines how inputs should be passed through the wrapped model.
        Here, we're assuming that the wrapped model takes both :math:`x` and :math:`t` as input,
        along with any additional keyword arguments.

        Optional things to do here:
            - check that t is in the dimensions that the model is expecting.
            - add a custom forward pass logic.
            - call the wrapped model.

        | given x, t
        | returns the model output for input x at time t, with extra information `extra`.

        Args:
            x (Tensor): input data to the model (batch_size, ...).
            t (Tensor): time (batch_size).
            **extras: additional information forwarded to the model, e.g., text condition.

        Returns:
            Tensor: model output.
        """
        return self.model(x=x, t=t, **extras)


class TimeBroadcastWrapper(ModelWrapper):
    """Wrap a model that expects `model(x_t=..., t=(batch, 1))`.

    This wrapper makes ODE solvers compatible with models that require a batched time input.
    """

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:
            if t.shape[0] == 1:
                t = t.expand(x.shape[0])
            assert t.shape[0] == x.shape[0]
            t = t[:, None]
        else:
            assert t.ndim == 2
            if t.shape[0] == 1:
                t = t.expand(x.shape[0], t.shape[1])
            assert t.shape[0] == x.shape[0]
            assert t.shape[1] == 1

        return self.model(x_t=x, t=t.float(), **extras)


class ODESolver:
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(self, velocity_model: ModelWrapper | Callable):
        super().__init__()
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        step_size: float | None,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor | None = None,
        return_intermediates: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> Tensor | Sequence[Tensor]:
        r"""Solve the ODE with the velocity field.

        Example:

        .. code-block:: python

            import torch
            from flow_matching.utils import ModelWrapper
            from flow_matching.solver import ODESolver

            class DummyModel(ModelWrapper):
                def __init__(self):
                    super().__init__(None)

                def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                    return torch.ones_like(x) * 3.0 * t**2

            velocity_model = DummyModel()
            solver = ODESolver(velocity_model=velocity_model)
            x_init = torch.tensor([0.0, 0.0])
            step_size = 0.001
            time_grid = torch.tensor([0.0, 1.0])

            result = solver.sample(x_init=x_init, step_size=step_size, time_grid=time_grid)

        Args:
            x_init (Tensor): initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): The process is solved in the interval [min(time_grid, max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to torch.tensor([0.0, 1.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Defaults to False.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tensor, Sequence[Tensor]]: The last timestep when return_intermediates=False, otherwise all values specified in time_grid.
        """

        if time_grid is None:
            time_grid = torch.tensor([0.0, 1.0], device=x_init.device)
        else:
            time_grid = time_grid.to(x_init.device)

        def ode_func(t, x):
            return self.velocity_model(x=x, t=t, **model_extras)

        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            # Approximate ODE solution with numerical ODE solver
            sol = odeint(
                ode_func,
                x_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        if return_intermediates:
            return sol
        else:
            return sol[-1]

    def compute_likelihood(
        self,
        x_1: Tensor,
        log_p0: Callable[[Tensor], Tensor],
        step_size: float | None,
        method: str = "euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Tensor | None = None,
        return_intermediates: bool = False,
        exact_divergence: bool = False,
        enable_grad: bool = False,
        **model_extras,
    ) -> tuple[Tensor, Tensor] | tuple[Sequence[Tensor], Tensor]:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

        Works similarly to sample, but solves the ODE in reverse to compute the log-likelihood. The velocity model must be differentiable with respect to x.
        The function assumes log_p0 is the log probability of the source distribution at :math:`t=0`.

        Args:
            x_1 (Tensor): target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 (Callable[[Tensor], Tensor]): Log probability function of the source distribution.
            step_size (Optional[float]): The step size. Must be None for adaptive step solvers.
            method (str): A method supported by torchdiffeq. Defaults to "euler". Other commonly used solvers are "dopri5", "midpoint" and "heun3". For a complete list, see torchdiffeq.
            atol (float): Absolute tolerance, used for adaptive step solvers.
            rtol (float): Relative tolerance, used for adaptive step solvers.
            time_grid (Tensor): If step_size is None then time discretization is set by the time grid. Must start at 1.0 and end at 0.0, otherwise the likelihood computation is not valid. Defaults to torch.tensor([1.0, 0.0]).
            return_intermediates (bool, optional): If True then return intermediate time steps according to time_grid. Otherwise only return the final sample. Defaults to False.
            exact_divergence (bool): Whether to compute the exact divergence or use the Hutchinson estimator.
            enable_grad (bool, optional): Whether to compute gradients during sampling. Defaults to False.
            **model_extras: Additional input for the model.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[Sequence[Tensor], Tensor]]: Samples at time_grid and log likelihood values of given x_1.
        """

        if time_grid is None:
            time_grid = torch.tensor([1.0, 0.0], device=x_1.device)
        else:
            time_grid = time_grid.to(x_1.device)

        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        # Fix the random projection for the Hutchinson divergence estimator
        if not exact_divergence:
            z = (torch.randn_like(x_1).to(x_1.device) < 0) * 2.0 - 1.0

        def ode_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        def dynamics_func(t, states):
            xt = states[0]
            with torch.set_grad_enabled(True):
                xt.requires_grad_()
                ut = ode_func(xt, t)

                if exact_divergence:
                    # Compute exact divergence
                    div = 0
                    for i in range(ut.flatten(1).shape[1]):
                        div += gradient(ut[:, i], xt, create_graph=True)[:, i]
                else:
                    # Compute Hutchinson divergence estimator E[z^T D_x(ut) z]
                    ut_dot_z = torch.einsum("ij,ij->i", ut.flatten(start_dim=1), z.flatten(start_dim=1))
                    grad_ut_dot_z = gradient(ut_dot_z, xt)
                    div = torch.einsum(
                        "ij,ij->i",
                        grad_ut_dot_z.flatten(start_dim=1),
                        z.flatten(start_dim=1),
                    )

            return ut.detach(), div.detach()

        y_init = (x_1, torch.zeros(x_1.shape[0], device=x_1.device))
        ode_opts = {"step_size": step_size} if step_size is not None else {}

        with torch.set_grad_enabled(enable_grad):
            sol, log_det = odeint(
                dynamics_func,
                y_init,
                time_grid,
                method=method,
                options=ode_opts,
                atol=atol,
                rtol=rtol,
            )

        x_source = sol[-1]
        source_log_p = log_p0(x_source)

        if return_intermediates:
            return sol, source_log_p + log_det[-1]
        else:
            return sol[-1], source_log_p + log_det[-1]
