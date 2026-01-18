from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import Module
from tqdm.auto import tqdm

from flow_matching import visualization
from flow_matching.datasets import TOY_DATASETS, SyntheticDataset, ToyDatasetName
from flow_matching.solver import TimeBroadcastWrapper
from flow_matching.utils import set_seed


@dataclass
class ScriptArguments:
    dataset: ToyDatasetName = "checkerboard"
    output_dir: Path = Path("outputs")
    learning_rate: float = 1e-3
    batch_size: int = 4096
    iterations: int = 20000
    log_every: int = 2000
    hidden_dim: int = 512
    seed: int = 42


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
        h = torch.cat([x_t, t], dim=1)
        return self.layers(h)


def main(args: ScriptArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = args.output_dir / "cfm" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    dataset: SyntheticDataset = TOY_DATASETS[args.dataset](device=device)

    flow = Mlp(dim=dataset.dim, time_dim=1, h=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), args.learning_rate)

    # Training

    losses = []
    for global_step in tqdm(range(args.iterations), desc="Training", dynamic_ncols=True):
        x_1 = dataset.sample(args.batch_size)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(x_1.size(0), 1, device=device)

        # Compute the Conditional Flow Matching objective
        # Check eq. (22) and (23) in the paper: https://arxiv.org/abs/2210.02747
        # where, we set \sigma_{\min} = 0
        x_t = (1 - t) * x_0 + t * x_1  # \phi_t(x_0)
        dx_t = x_1 - x_0  # u_t(x|x_t)

        optimizer.zero_grad()
        loss = F.mse_loss(flow(x_t=x_t, t=t), dx_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (global_step + 1) % args.log_every == 0:
            tqdm.write(f"| step: {global_step + 1:6d} | loss: {loss.item():8.4f} |")

    flow.eval()
    torch.save(flow.state_dict(), output_dir / "ckpt.pth")

    visualization.plot_loss_curve(losses=losses, output_path=output_dir / "losses.png")

    # Sampling with ODE solver and visualization

    wrapped_model = TimeBroadcastWrapper(flow)

    visualization.plot_ode_sampling_evolution(
        flow=wrapped_model,
        dataset=dataset,
        output_dir=output_dir,
        filename=f"ode_sampling_evolution_{args.dataset}.png",
    )

    visualization.save_vector_field_and_samples_as_gif(
        flow=wrapped_model,
        dataset=dataset,
        output_dir=output_dir,
        filename=f"vector_field_and_samples_{args.dataset}.gif",
    )

    visualization.plot_likelihood(
        flow=wrapped_model,
        dataset=dataset,
        output_dir=output_dir,
        filename=f"likelihood_{args.dataset}.png",
    )


if __name__ == "__main__":
    main(tyro.cli(ScriptArguments))
