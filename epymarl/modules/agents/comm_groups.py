"""Pairwise communication groups for the CommNet / TarMAC agents."""
import torch as th


def build_group_mask(
    comm_mask: th.Tensor, mode: str = "bits", symmetric: bool = False
) -> th.Tensor:
    """Build a pairwise communication mask.

    Two input conventions are supported, selected by ``mode``:

    - ``"bits"`` (default): ``comm_mask`` holds per-agent connectivity bits with
      shape [B, N] or [B, N, 1]. The pairwise group is the outer-AND of the bits,
      i.e. agents i and j communicate iff both bits are set (a hub topology). This
      is the original behaviour used by the aerial-ordering environment and is left
      byte-for-byte unchanged.
    - ``"adjacency"``: ``comm_mask`` is already a pairwise adjacency matrix with
      shape [B, N, N]; mask[b, i, j] means agent i can attend to agent j. Used
      directly (optionally symmetrised for undirected graphs).

    Returns:
        Tensor[bool] with shape [B, N, N]; the self-loop diagonal is always set.
    """
    if mode == "adjacency":
        pairwise = comm_mask > 0.5
        if symmetric:
            pairwise = pairwise | pairwise.transpose(-1, -2)
        n_agents = pairwise.size(-1)
    else:
        if comm_mask.dim() == 3:
            comm_mask = comm_mask.squeeze(-1)
        comm_mask = comm_mask > 0.5
        pairwise = comm_mask.unsqueeze(2) & comm_mask.unsqueeze(1)
        n_agents = comm_mask.size(1)

    eye = th.eye(n_agents, device=pairwise.device, dtype=th.bool).unsqueeze(0)
    return pairwise | eye
