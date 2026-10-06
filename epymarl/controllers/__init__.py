REGISTRY = {}

from .basic_controller import BasicMAC
from .comm_mac import CommMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["comm_mac"] = CommMAC
