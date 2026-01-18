import copy
from dataclasses import dataclass, field
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
from flow_matching.solver import ModelWrapper, ODESolver, TimeBroadcastWrapper
from flow_matching.utils import set_seed


@dataclass
class ScriptArguments:
    dataset: ToyDatasetName = "checkerboard"
    pretrained_model: Path | None = None
    output_dir: Path = Path("outputs")
    learning_rate: float = 1e-3
    batch_size: int = 4096
    iterations: int = 2000
    log_every: int = 200
    hidden_dim: int = 512
    seed: int = 42
    num_inference_steps: int = field(
        default=20,
        metadata={
            "help": "Number of fixed steps for integrating t in [0, 1] (dt = 1 / num_inference_steps). Note: NFE depends on `method` (e.g., midpoint uses 2 model evals per step)."
        },
    )
    method: str = "midpoint"

    def __post_init__(self) -> None:
        if self.pretrained_model is None:
            self.pretrained_model = Path("outputs") / "cfm" / self.dataset / "ckpt.pth"

        if not self.pretrained_model.is_file():
            raise FileNotFoundError(f"Pretrained model checkpoint {self.pretrained_model} not found.")


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


def integrate_source_samples(
    flow: ModelWrapper,
    source_samples: Tensor,
    num_inference_steps: int,
    method: str,
) -> Tensor:
    device = next(flow.parameters()).device
    x_init = source_samples.to(device)
    t_span = torch.tensor([0.0, 1.0], device=device)
    step_size = 1.0 / num_inference_steps
    solver = ODESolver(flow)
    samples = solver.sample(
        x_init=x_init,
        step_size=step_size,
        method=method,
        time_grid=t_span,
        return_intermediates=False,
    )
    return samples


def main(args: ScriptArguments) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    output_dir = args.output_dir / "reflow" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    dataset: SyntheticDataset = TOY_DATASETS[args.dataset](device=device)
    data_shape = dataset.sample(4).shape[1:]  # n does not matter here, just need to account for mixture which requires n % 4 == 0

    pretrained_flow = Mlp(dim=dataset.dim, time_dim=1, h=args.hidden_dim).to(device)
    pretrained_flow.load_state_dict(torch.load(args.pretrained_model, weights_only=True))
    pretrained_flow.eval()

    reflow = copy.deepcopy(pretrained_flow)
    reflow.train()

    optimizer = torch.optim.AdamW(reflow.parameters(), args.learning_rate)
    pretrained_flow = TimeBroadcastWrapper(pretrained_flow)

    # Training

    losses = []
    for global_step in tqdm(range(args.iterations), desc="Training", dynamic_ncols=True):
        x_0 = torch.randn([args.batch_size, *data_shape], dtype=torch.float32, device=device)
        t = torch.rand(args.batch_size, 1, dtype=torch.float32, device=device)

        x_1 = integrate_source_samples(
            pretrained_flow,
            x_0,
            num_inference_steps=args.num_inference_steps,
            method=args.method,
        )

        # Reflow objective (distillation): build (x_0, x_1) pairs by integrating a pretrained flow,
        # then regress the straight-line displacement field u_t(x_t) = x_1 - x_0 along x_t = (1 - t) x_0 + t x_1.
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        optimizer.zero_grad()
        loss = F.mse_loss(reflow(x_t=x_t, t=t), dx_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (global_step + 1) % args.log_every == 0:
            tqdm.write(f"| step: {global_step + 1:6d} | loss: {loss.item():8.4f} |")

    reflow.eval()
    torch.save(reflow.state_dict(), output_dir / "ckpt.pth")
    visualization.plot_loss_curve(losses=losses, output_path=output_dir / "losses.png")

    # Sampling with ODE solver and visualization

    reflow = TimeBroadcastWrapper(reflow)

    visualization.plot_ode_sampling_evolution(
        flow=reflow,
        dataset=dataset,
        output_dir=output_dir,
        filename=f"ode_sampling_evolution_{args.dataset}.png",
    )

    visualization.save_vector_field_and_samples_as_gif(
        flow=reflow,
        dataset=dataset,
        output_dir=output_dir,
        filename=f"vector_field_and_samples_{args.dataset}.gif",
    )

    visualization.plot_likelihood(
        flow=reflow,
        dataset=dataset,
        output_dir=output_dir,
        filename=f"likelihood_{args.dataset}.png",
    )


if __name__ == "__main__":
    main(tyro.cli(ScriptArguments))
