"""
Agents Module

Contains intelligent agents for mining optimization including:
- PoUW agent for proof of useful work coordination
- Intelligent mining agent for dynamic decision making
- Multi-agent systems for distributed optimization
- Deep reinforcement learning agents (DSAC)
"""

# Import with optional dependency handling
import warnings

def _safe_import(module_name, items):
    """Safely import items from a module."""
    imported = {}
    try:
        module = __import__(f"pouw_dem.core.agents.{module_name}", fromlist=items if isinstance(items, list) else [items])
        if isinstance(items, list):
            for item in items:
                try:
                    imported[item] = getattr(module, item)
                except AttributeError:
                    warnings.warn(f"Could not import {item} from {module_name}")
                    imported[item] = None
        else:
            imported[items] = getattr(module, items)
    except ImportError as e:
        warnings.warn(f"Could not import from {module_name}: {e}")
        if isinstance(items, list):
            for item in items:
                imported[item] = None
        else:
            imported[items] = None
    return imported

# PoUW Agent imports
pouw_imports = _safe_import("PoUWAgent", ["PoUWAgent", "TaskType", "Priority", "GridTask", "MinerState"])
PoUWAgent = pouw_imports.get("PoUWAgent")
TaskType = pouw_imports.get("TaskType")
Priority = pouw_imports.get("Priority")
GridTask = pouw_imports.get("GridTask")
MinerState = pouw_imports.get("MinerState")

# Other agent imports
IntelligentMiningAgent = _safe_import("intelligent_mining_agent", "IntelligentMiningAgent").get("IntelligentMiningAgent")
Multi_Agent = _safe_import("MultiAgent", "Multi_Agent").get("Multi_Agent")
DSAC = _safe_import("Agent", "DSAC").get("DSAC")

# Environment imports
env_imports = _safe_import("house", ["ProgressusEnv", "registration"])
ProgressusEnv = env_imports.get("ProgressusEnv")
registration = env_imports.get("registration")

# Buffer and model imports
ReplayBuffer = _safe_import("buffer", "ReplayBuffer").get("ReplayBuffer")
model_imports = _safe_import("model", ["Critics", "Actors"])
Critics = model_imports.get("Critics")
Actors = model_imports.get("Actors")

# Network imports
network_imports = _safe_import("networks", ["Actor", "Critic", "DDQN_Net"])
Actor = network_imports.get("Actor")
Critic = network_imports.get("Critic")
DDQN_Net = network_imports.get("DDQN_Net")

# Utility imports
util_imports = _safe_import("utils", ["save", "collect_random"])
save = util_imports.get("save")
collect_random = util_imports.get("collect_random")

# Gumbel softmax imports
gumbel_imports = _safe_import("gumbel_softmax", ["gumbel_sample", "gumbel_softmax_sample", "gumbel_softmax", "onehot_action"])
gumbel_sample = gumbel_imports.get("gumbel_sample")
gumbel_softmax_sample = gumbel_imports.get("gumbel_softmax_sample")
gumbel_softmax = gumbel_imports.get("gumbel_softmax")
onehot_action = gumbel_imports.get("onehot_action")

__all__ = [
    # Main agents
    "PoUWAgent",
    "IntelligentMiningAgent",
    "Multi_Agent",
    "DSAC",
    
    # Environment
    "ProgressusEnv",
    "registration",
    
    # Agent components
    "TaskType",
    "Priority",
    "GridTask",
    "MinerState",
    
    # Neural network components
    "Critics",
    "Actors",
    "Actor",
    "Critic",
    "DDQN_Net",
    
    # Utilities
    "ReplayBuffer",
    "save",
    "collect_random",
    
    # Gumbel softmax utilities
    "gumbel_sample",
    "gumbel_softmax_sample",
    "gumbel_softmax",
    "onehot_action",
]