"""Adapter that runs InforMARL (graph rMAPPO) on the MAC benchmark envs.

Nothing in InforMARL's algorithm code is modified. The adapter only provides:
  env_wrapper.py  -- MAC env (built through epymarl/envs gymma, exactly as the
                     EPyMARL baselines build it) exposed with InforMARL's graph
                     env interface; the graph is the env's live comm mask.
  parallel_envs.py-- parallel env copies (subprocesses) that also carry action masks.
  runner.py       -- GMPERunner subclass: passes action masks, logs EPyMARL-style
                     episode stats and test runs to results/informarl/tb_logs.
  train_mac.py    -- entry point.
"""
