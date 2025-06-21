# PoUW-DEM Mining

Privacy-preserving, grid-optimizing mining pool for MARA Holdings using Proof of Useful Work (PoUW) and Federated Deep Reinforcement Learning (FDRL).

## Overview

This system dynamically allocates computational resources between Bitcoin mining and grid optimization tasks based on profitability and grid needs. Key features:

- **Federated Learning**: Privacy-preserving multi-agent coordination
- **Dynamic Task Scheduling**: Real-time switching between mining and grid tasks
- **Grid Integration**: Connected to ERCOT for real-time grid data
- **Blockchain Recording**: Allocation decisions recorded on Polygon
- **Smart Contracts**: Deployed contracts for coordination and rewards

## Architecture

```
pouw_dem/
├── api/              # External API integrations (ERCOT, mining pools)
├── blockchain/       # Smart contracts and Web3 integration
├── core/            # Core functionality
│   ├── agents/      # FDRL agents with DSAC algorithm
│   ├── grid_simulation/  # MATPOWER integration
│   └── security/    # SGX secure enclave interface
├── training/        # Model training scripts
├── deployment/      # Production deployment
├── models/          # Trained models
└── web/            # Dashboard UI
```

## Quick Start

1. Install dependencies:
```bash
pip install -e .
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

3. Run the system:
```bash
python -m pouw_dem.deployment.run_full_system
```

4. Access dashboard at http://localhost:5000

## Key Components

### FDRL Agents
- Discrete Soft Actor-Critic (DSAC) algorithm
- Experience replay for stable training
- Federated averaging for privacy-preserving coordination
- Achieved 99.7-122% improvement over baseline

### Grid Integration
- Real-time ERCOT ECRS market data
- OAuth2 authentication with ROPC flow
- Renewable energy threshold (30%) for task prioritization
- MATPOWER for power flow simulation

### Mining Pool Integration
- Connected to SlushPool API
- Support for major pools (Foundry, AntPool, F2Pool)
- Dynamic hashrate allocation

### Smart Contracts (Polygon Mainnet)
- GridOptimizationRewards: 0x4fE7f9feCC3470119c0561f322E1AA76a3C8D8e6
- FederatedCoordinator: 0x53B7c73F60E1cCbaa7774079da42bCc3eAac4293
- GridDataOracle: 0xd5166447fc08C41f3F8C46df073f99c1B6eE6aC2
- TaskAllocation: 0x9a826233C6A616c73f8f2199439cE09d9d8fc2bD
- RewardToken: 0xF593210F049711e3279f8917Ff2988E616257F09

## API Endpoints

- `GET /api/allocation` - Current resource allocation
- `GET /api/metrics` - Performance metrics
- `GET /api/agent-status` - Agent training status
- `POST /api/retrain` - Trigger agent retraining

## License

Proprietary - MARA Holdings
