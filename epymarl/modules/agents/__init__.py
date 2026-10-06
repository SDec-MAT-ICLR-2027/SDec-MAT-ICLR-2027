from .rnn_agent import RNNAgent
from .commnet_agent import CommNetAgent
from .tarmac_agent import TarMACAgent


REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["commnet"] = CommNetAgent
REGISTRY["tarmac"] = TarMACAgent
