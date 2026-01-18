from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Module

from flow_matching.datasets import TOY_DATASETS, SyntheticDataset, ToyDatasetName
from flow_matching.solver import ODESolver, TimeBroadcastWrapper


def sample(
    ode_model: TimeBroadcastWrapper,
    source_samples: Tensor,
    num_output_steps: int,
    num_inference_steps: int,
    method: str,
    return_intermediates: bool,
    **model_kwargs,
) -> Tensor:
    device = next(ode_model.parameters()).device
    x_init = source_samples.to(device)
    t_eval = torch.linspace(0, 1, num_output_steps, device=device)
    solver = ODESolver(ode_model)
    step_size = 1.0 / num_inference_steps
    return solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=method,
        time_grid=t_eval,
        return_intermediates=return_intermediates,
        **model_kwargs,
    )


@dataclass
class ScriptArguments:
    dataset: ToyDatasetName = "checkerboard"
    output_dir: Path = Path("outputs")
    num_samples: int = 500_000
    grid_size: int = 15
    num_output_steps: int = field(
        default=101,
        metadata={
            "help": "Number of output time points in [0, 1] to save/visualize (i.e., GIF frames). This does not control internal solver steps."
        },
    )
    num_inference_steps: int = field(
        default=20,
        metadata={
            "help": "Number of fixed steps for integrating t in [0, 1] (dt = 1 / num_inference_steps). Note: NFE depends on `method` (e.g., midpoint uses 2 model evals per step)."
        },
    )
    method: str = "midpoint"
    fps: int = 20


class Mlp(Module):
    def __init__(self, dim: int = 2, time_dim: int = 1, h: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim + time_dim, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
            nn.Linear(h, dim),
        )

    def forward(
        self,
        x_t: Float[Tensor, "batch dim"],
        t: Float[Tensor, "batch time_dim"],
    ) -> Float[Tensor, "batch dim"]:
        return self.layers(torch.cat([x_t, t], dim=1))


def main(args: ScriptArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfm_path = args.output_dir / "cfm" / args.dataset / "ckpt.pth"
    reflow_path = args.output_dir / "reflow" / args.dataset / "ckpt.pth"
    if not cfm_path.is_file():
        raise FileNotFoundError(f"CFM checkpoint {cfm_path} not found.")
    if not reflow_path.is_file():
        raise FileNotFoundError(f"Reflow checkpoint {reflow_path} not found.")

    cfm = Mlp(dim=2, time_dim=1, h=512)
    cfm.load_state_dict(torch.load(cfm_path, weights_only=True))
    cfm.to(device)
    cfm.eval()
    wrapped_cfm = TimeBroadcastWrapper(cfm)

    reflow = Mlp(dim=2, time_dim=1, h=512)
    reflow.load_state_dict(torch.load(reflow_path, weights_only=True))
    reflow.to(device)
    reflow.eval()
    wrapped_reflow = TimeBroadcastWrapper(reflow)

    dataset: SyntheticDataset = TOY_DATASETS[args.dataset](device=device)

    x_init = torch.randn(args.num_samples, 2).to(device)
    samples_cfm = sample(
        wrapped_cfm,
        x_init,
        num_output_steps=args.num_output_steps,
        num_inference_steps=args.num_inference_steps,
        method=args.method,
        return_intermediates=True,
    )
    samples_reflow = sample(
        wrapped_reflow,
        x_init,
        num_output_steps=args.num_output_steps,
        num_inference_steps=args.num_inference_steps,
        method=args.method,
        return_intermediates=True,
    )

    samples_cfm = samples_cfm.detach().cpu().numpy()
    samples_reflow = samples_reflow.detach().cpu().numpy()

    # Create a grid for the density and vector field
    x_range, y_range = dataset.get_square_range()
    x = np.linspace(x_range[0], x_range[1], args.grid_size)
    y = np.linspace(y_range[0], y_range[1], args.grid_size)
    xv, yv = np.meshgrid(x, y)
    grid = np.stack([xv, yv], axis=-1).reshape(-1, 2)  # Shape: (grid_size^2, 2)

    grid_tensor = torch.tensor(grid, dtype=torch.float32, device=device)
    t_eval = torch.linspace(0, 1, args.num_output_steps, device=device)

    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    def update(frame):
        for ax in axes.flatten():
            ax.clear()

        # Current time step
        t = t_eval[frame]

        # Plot CFM samples
        axes[0, 0].hist2d(
            samples_cfm[frame, :, 0],
            samples_cfm[frame, :, 1],
            bins=300,
            range=[x_range, y_range],
            cmap="viridis",
        )
        axes[0, 0].set_title(f"Flow Matching (t = {t.item():.2f})", fontsize=16)
        axes[0, 0].set_xlim(x_range)
        axes[0, 0].set_ylim(y_range)
        axes[0, 0].set_aspect("equal")
        axes[0, 0].axis("off")

        # Plot Reflow samples
        axes[0, 1].hist2d(
            samples_reflow[frame, :, 0],
            samples_reflow[frame, :, 1],
            bins=300,
            range=[x_range, y_range],
            cmap="viridis",
        )
        axes[0, 1].set_title(f"Reflow (2-Rectified Flow) (t = {t.item():.2f})", fontsize=16)
        axes[0, 1].set_xlim(x_range)
        axes[0, 1].set_ylim(y_range)
        axes[0, 1].set_aspect("equal")
        axes[0, 1].axis("off")

        # Plot CFM vector field
        vectors_cfm = wrapped_cfm(x=grid_tensor, t=t).detach().cpu().numpy()
        vectors_cfm = vectors_cfm.reshape(args.grid_size, args.grid_size, 2)
        magnitudes_cfm = np.linalg.norm(vectors_cfm, axis=2)
        axes[1, 0].quiver(
            xv,
            yv,
            vectors_cfm[:, :, 0],
            vectors_cfm[:, :, 1],
            magnitudes_cfm,
            angles="xy",
            scale_units="xy",
            scale=10.0,
            cmap=cm.coolwarm,
            alpha=0.8,
            width=0.01,
        )

        # axes[1, 0].set_title(f"CFM Vector Field (t = {t.item():.2f})", fontsize=16)
        axes[1, 0].set_xlim(x_range)
        axes[1, 0].set_ylim(y_range)
        axes[1, 0].set_aspect("equal")
        axes[1, 0].axis("off")

        # Plot Reflow vector field
        vectors_reflow = wrapped_reflow(x=grid_tensor, t=t).detach().cpu().numpy()
        vectors_reflow = vectors_reflow.reshape(args.grid_size, args.grid_size, 2)
        magnitudes_reflow = np.linalg.norm(vectors_reflow, axis=2)
        axes[1, 1].quiver(
            xv,
            yv,
            vectors_reflow[:, :, 0],
            vectors_reflow[:, :, 1],
            magnitudes_reflow,
            angles="xy",
            scale_units="xy",
            scale=10.0,
            cmap=cm.coolwarm,
            alpha=0.8,
            width=0.01,
        )
        # axes[1, 1].set_title(f"Reflow Vector Field (t = {t.item():.2f})", fontsize=16)
        axes[1, 1].set_xlim(x_range)
        axes[1, 1].set_ylim(y_range)
        axes[1, 1].set_aspect("equal")
        axes[1, 1].axis("off")

    # Adjust layout to reduce white space
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.05)

    ani = animation.FuncAnimation(fig, update, frames=args.num_output_steps)

    print("Saving animation...")
    filename = args.output_dir / "comparisons" / f"cfm_reflow_{args.dataset}.gif"
    filename.parent.mkdir(parents=True, exist_ok=True)
    ani.save(filename, writer="pillow", fps=args.fps)
    print(f"Saved animation to {filename}")


if __name__ == "__main__":
    main(tyro.cli(ScriptArguments))
