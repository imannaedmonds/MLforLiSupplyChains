# MLforLiSupplyChains

# LiPOMDPs: Modeling Lithium Resource Management Using POMDPs

## Overview

LiPOMDPs is a computational framework for modeling the US path to lithium self-sufficiency using Partially Observable Markov Decision Processes (POMDPs). This project addresses the strategic challenges of lithium resource management, balancing domestic vs. foreign mining decisions while considering environmental impacts, economic costs, and uncertainty in resource availability.

The framework allows for the modeling and comparison of various lithium sourcing policies including:
- MCTS (Monte Carlo Tree Search)
- POMCPOW (Partially Observable Monte Carlo Planning with Observation Widening)
- Heuristic policies (ExploreNSteps, ImportOnly, EfficiencyPolicy, etc.)

This implementation enables the generation of Pareto-optimal curves analyzing the trade-offs between:
- NPV (Net Present Value) vs. Emission Costs
- Total Volume (Domestic + Imported) vs. Total Emissions
- Different lithium mining and exploration strategies under uncertain conditions

## Installation

### Prerequisites

- Julia 1.6 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/imannaedmonds/MLforLiSupplyChains.git 
cd LiPOMDPs
```

2. Install the required packages. Start Julia and run:
```julia
using Pkg
Pkg.activate(".")  # Activate the project environment
Pkg.instantiate()  # Install the packages specified in Project.toml
```

## Project Structure

```
LiPOMDPs/
├── Project.toml          # Julia project dependencies
├── src/                  # Core model implementation
│   ├── LiPOMDPs.jl       # Main module definition
│   ├── model.jl          # POMDP model definition
│   ├── policies.jl       # Policy implementations
│   ├── pomdp.jl          # Core POMDP functionality
│   ├── steps.jl          # Simulation steps
│   └── utils.jl          # Utility functions
├── scripts/              # Experiment scripts
│   └── experiments/      # Various experiment configurations
└── data/                 # Directory for input data (if needed)
```

## Running the Experiments

There are several experiment scripts available in the `scripts/experiments/` directory:

### Basic Policy Evaluation

To run the basic policy evaluation:

```julia
include("scripts/experiments/pricing_deterministic_vs_stochastic.jl")
```

This script compares policy performance under different conditions (deterministic vs. stochastic pricing).

### Pareto Curve Generation

To generate Pareto curves comparing different policies:

```julia
include("scripts/experiments/combination_of_pareto_curves.jl")
```

This comprehensive script compares MCTS, POMCPOW, ExploreNSteps, and ImportOnly policies across emissions and volume dimensions.

### NPV vs. Emission Cost Analysis

To generate NPV vs. emission cost trade-offs:

```julia
include("scripts/experiments/npv_vs_emission_cost_varied.jl")
```

This script analyzes how different carbon pricing affects the NPV vs. emission cost trade-offs across different policies.

### Heuristic Policy Analysis

To evaluate specific heuristic policies:

```julia
# For ExploreNSteps policy
include("scripts/experiments/heuristic_explore_n_steps.jl")

# For ImportOnly policy
include("scripts/experiments/heuristic_import_only.jl")

# For combined heuristic comparison
include("scripts/experiments/heuristic_combined.jl")
```

## Model Parameters

The LiPOMDP model can be configured with various parameters:

- `t_goal`: Time goal (default: 10)
- `σ_obs`: Standard deviation of the observation noise (default: 0.1)
- `Vₜ_goal`: Volume goal (default: 8)
- `γ`: Discount factor (default: 0.98)
- `time_horizon`: Simulation time horizon (default: 30)
- `n_deposits`: Number of deposits to model (default: 4)
- `stochastic_price`: Enable stochastic pricing (default: false)
- `alpha`: Trade-off parameter between emissions and volume (default: 1)

You can modify these parameters when initializing the POMDP:

```julia
# Example: Initialize a POMDP with custom parameters
pomdp = initialize_lipomdp(
    t_goal=15,
    α_obs=0.2, 
    stochastic_price=true
)
```

## Customizing Policies

You can create and evaluate your own policies by extending the base policy types. For example:

```julia
# Create a custom policy
my_custom_policy = ExploreNStepsPolicy(
    pomdp=my_pomdp,
    explore_steps=15,
    curr_steps=1
)

# Evaluate your policy
results = evaluate_policy(my_pomdp, my_custom_policy, 10, 30)
```

## Output and Visualization

Most experiment scripts generate plots that are saved as PNG files. These visualizations typically include:

- Pareto curves showing trade-offs between different objectives
- Bar charts comparing policy performance
- Line plots showing performance over time

The plots are automatically saved in the working directory with descriptive filenames.

## Contact

For questions, issues, or collaboration inquiries, please open an issue on this repository or contact the contributors directly.