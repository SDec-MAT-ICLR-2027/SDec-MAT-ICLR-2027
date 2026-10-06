"""The four methods.

Everything outside this file is identical across methods within a task, so this
is the whole of what is being compared.

`entry` names the training script; `flags` are the only method-specific command
line arguments. `train_joint_nav.py` normalises the two communication arms to
`algorithm_name=mat_comm` with `comm_mask_kind=pairwise`,
`encoder_critic_attn=soft_group_bias` and `critic_global_state=1`; they differ
only in `decoder_self_only`.
"""

ARMS = {
    # The proposed method: group-masked encoder plus an autoregressive decoder
    # whose conditioning set is the agent's one-hop neighbours that precede it.
    "sdec_mat": {
        "label": "SDec-MAT",
        "entry": "mat",
        "flags": {"algorithm_name": "mat_comm_soft"},
    },
    # The decentralised-execution control: the same communication-masked
    # encoder, so observations are still shared, but no action conditioning.
    "mat_dec": {
        "label": "MAT-Dec",
        "entry": "mat",
        "flags": {"algorithm_name": "mat_comm_dec"},
    },
    # Per-agent MLP actor on its own observation, centralised critic.
    "mappo": {
        "label": "MAPPO",
        "entry": "mat",
        "flags": {"algorithm_name": "mappo", "dec_actor": True},
    },
    # Graph neural network over the entity graph; lives in the other code base.
    "informarl": {
        "label": "InforMARL",
        "entry": "graph",
        "flags": {
            "algorithm_name": "mappo",
            "use_recurrent_policy": True,   # store_false flag: present => disabled
            "gnn_layer_N": 2,
        },
    },
}
