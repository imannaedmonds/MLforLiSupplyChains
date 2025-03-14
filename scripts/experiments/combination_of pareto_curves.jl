using Random
using POMDPs
using POMDPTools
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW
using Distributions
using Parameters
using Plots
using Statistics
using ProgressBars

"""
This script creates a comprehensive comparison of policy types for the LiPOMDP:
- MCTS
- POMCPOW
- ExploreNSteps
- ImportOnly

It generates a single Pareto curve plot with all policies for easy comparison.
"""

rng = MersenneTwister(1)

function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean=sample_mean, se=sample_se)
end

# Experiment function to evaluate a planner
function experiment(planner, eval_pomdp, n_reps=20, max_steps=30; initial_belief=nothing)
    reward_tot_all = []
    reward_disc_all = []
    emission_tot_all = []
    emission_disc_all = []
    domestic_tot_all = []
    imported_tot_all = []

    for t in tqdm(1:n_reps)
        reward_tot = 0.0
        reward_disc = 0.0
        emission_tot = 0.0
        emission_disc = 0.0
        vol_tot = 0.0    # mined domestically
        imported_tot = 0.0  # imported/mined internationally
        disc = 1.0

        # Use custom initial belief if provided, otherwise use default
        if initial_belief !== nothing
            up = updater(planner) 
            step_iter = stepthrough(eval_pomdp, planner, up, initial_belief, "s,a,o,r", max_steps=max_steps)
        else
            step_iter = stepthrough(eval_pomdp, planner, "s,a,o,r", max_steps=max_steps)
        end

        for (s, a, o, r) in step_iter
            # Compute reward and discounted reward
            reward_tot += r
            reward_disc += r * disc

            # Compute emissions and discount emissions
            e = get_action_emission(eval_pomdp, a)
            emission_tot += e
            emission_disc += e * disc

            # Track domestic vs imported mining
            if a.a == "MINE1" || a.a == "MINE2"
                vol_tot += 1
            elseif a.a == "MINE3" || a.a == "MINE4"
                imported_tot += 1
            end

            disc *= discount(eval_pomdp)
        end
        
        # Store results from this repetition
        push!(reward_tot_all, reward_tot)
        push!(reward_disc_all, reward_disc)
        push!(emission_tot_all, emission_tot)
        push!(emission_disc_all, emission_disc)
        push!(domestic_tot_all, vol_tot)
        push!(imported_tot_all, imported_tot)
    end

    # Compute metrics for all repetitions
    results = Dict(
        "Total Reward" => compute_metrics(reward_tot_all),
        "Disc. Reward" => compute_metrics(reward_disc_all),
        "Total Emissions" => compute_metrics(emission_tot_all),
        "Disc. Emissions" => compute_metrics(emission_disc_all),
        "Total Domestic" => compute_metrics(domestic_tot_all),
        "Total Imported" => compute_metrics(imported_tot_all)
    )

    return results
end

# MCTS generation function
function compute_mcts_results(param_values, param_name="alpha"; 
                             stochastic_price=false, max_steps=30, n_reps=20)
    results = Dict()
    
    for param_value in tqdm(param_values)
        # Initialize POMDP
        if param_name == "alpha"
            pomdp = initialize_lipomdp(alpha=param_value, stochastic_price=stochastic_price, compute_tradeoff=true)
        else
            pomdp = initialize_lipomdp(stochastic_price=stochastic_price, compute_tradeoff=true)
        end
        
        train_up = LiBeliefUpdater(pomdp)
        
        # Create belief MDP for MCTS
        mdp = GenerativeBeliefMDP(pomdp, train_up, terminal_behavior=ContinueTerminalBehavior(pomdp, train_up))
        
        # Set up a fallback policy for MCTS
        rollout_policy = EfficiencyPolicyWithUncertainty(pomdp, 1.0, [true, true, true, true])
        
        # Configure MCTS solver - adjust parameters based on param_name
        if param_name == "depth"
            depth = param_value
        else
            depth = 10  # default
        end
        
        if param_name == "iterations"
            n_iterations = param_value
        else
            n_iterations = 100  # default
        end
        
        mcts_solver = DPWSolver(
            depth=depth,
            n_iterations=n_iterations,
            estimate_value=RolloutEstimator(rollout_policy),
            enable_action_pw=false,
            enable_state_pw=true,
            k_state=4.0,
            alpha_state=0.1
        )
        
        # Solve the MDP
        mcts_planner = solve(mcts_solver, mdp)
        
        # Run experiment
        results[param_value] = experiment(mcts_planner, pomdp, n_reps, max_steps)
    end
    
    return results
end

# POMCPOW generation function
function compute_pomcpow_results(param_values, param_name="alpha"; 
                                stochastic_price=false, max_steps=30, n_reps=20)
    results = Dict()
    
    for param_value in tqdm(param_values)
        # Initialize POMDP
        if param_name == "alpha"
            pomdp = initialize_lipomdp(alpha=param_value, stochastic_price=stochastic_price, compute_tradeoff=true)
        else
            pomdp = initialize_lipomdp(stochastic_price=stochastic_price, compute_tradeoff=true)
        end
        
        # Configure solver parameters based on param_name
        if param_name == "tree_queries"
            tree_queries = param_value
        else
            tree_queries = 100  # default
        end
        
        if param_name == "max_depth"
            max_depth = param_value
        else
            max_depth = 10 # default
        end
        
        # Configure POMCPOW solver
        solver = POMCPOW.POMCPOWSolver(
            tree_queries=tree_queries,
            estimate_value=estimate_value,
            k_observation=4.0,
            alpha_observation=0.1,
            max_depth=max_depth,
            enable_action_pw=false,
            init_N=10
        )
        
        # Solve the POMDP
        pomcpow_planner = solve(solver, pomdp)
        
        # Run experiment
        results[param_value] = experiment(pomcpow_planner, pomdp, n_reps, max_steps)
    end
    
    return results
end

# ExploreNSteps generation function
function compute_explore_n_steps_results(step_values; stochastic_price=false, max_steps=30, n_reps=20)
    results = Dict()
    
    for num_steps in tqdm(step_values)
        # Initialize POMDP
        pomdp = initialize_lipomdp(stochastic_price=stochastic_price, compute_tradeoff=true)
        
        # Create policy
        policy = ExploreNStepsPolicy(pomdp=pomdp, explore_steps=num_steps, curr_steps=1)
        
        # Run experiment
        results[num_steps] = experiment(policy, pomdp, n_reps, max_steps)
    end
    
    return results
end

# ImportOnly generation function
function compute_import_only_results(step_values; stochastic_price=false, max_steps=30, n_reps=20)
    results = Dict()
    
    for num_steps in tqdm(step_values)
        # Initialize POMDP
        pomdp = initialize_lipomdp(stochastic_price=stochastic_price, compute_tradeoff=true)
        updater = LiBeliefUpdater(pomdp)
        
        # Initialize custom belief
        initial_belief = initialize_belief_import_only(updater)
        
        # Create policy
        policy = ImportOnlyPolicy(pomdp=pomdp, explore_steps=num_steps)
        
        # Run experiment with custom belief
        results[num_steps] = experiment(policy, pomdp, n_reps, max_steps, initial_belief=initial_belief)
    end
    
    return results
end

# Removed Random policy generation function

# Process results for plotting
function process_results_for_plot(results)
    param_values = sort(collect(keys(results)))
    
    xs = [results[param]["Total Emissions"].mean for param in param_values]
    ys = [results[param]["Total Domestic"].mean + results[param]["Total Imported"].mean for param in param_values]
    xerr = [results[param]["Total Emissions"].se for param in param_values]
    yerr = [sqrt(results[param]["Total Domestic"].se^2 + results[param]["Total Imported"].se^2) for param in param_values]
    
    # Sort by emissions for better curve visualization
    idxs = sortperm(xs)
    xs = xs[idxs]
    ys = ys[idxs]
    xerr = xerr[idxs]
    yerr = yerr[idxs]
    
    return xs, ys, xerr, yerr
end

# Plot combined Pareto curve for all policies
function plot_comprehensive_pareto(mcts_results, pomcpow_results, explore_results, import_results)
    # Process all results
    mcts_xs, mcts_ys, mcts_xerr, mcts_yerr = process_results_for_plot(mcts_results)
    pomcpow_xs, pomcpow_ys, pomcpow_xerr, pomcpow_yerr = process_results_for_plot(pomcpow_results)
    explore_xs, explore_ys, explore_xerr, explore_yerr = process_results_for_plot(explore_results)
    import_xs, import_ys, import_xerr, import_yerr = process_results_for_plot(import_results)
    
    # Create the plot
    p = plot(
        xlabel="Total Emissions",
        ylabel="Total Volume (Domestic + Imported)",
        title="Comprehensive Pareto Comparison of all Policies",
        legend=:topright,
        grid=true,
        gridalpha=0.3,
        size=(1200, 800),
        background_color=:white,
        foreground_color=:black,
        margin=10Plots.mm
    )
    
    # Plot MCTS data
    scatter!(p, 
        mcts_xs, 
        mcts_ys, 
        xerr=mcts_xerr, 
        yerr=mcts_yerr,
        label="MCTS",
        markercolor=:blue,
        markersize=7,
        markershape=:circle
    )
    
    plot!(p, mcts_xs, mcts_ys,
        linecolor=:blue,
        linestyle=:solid,
        alpha=0.8,
        linewidth=2,
        label=false
    )
    
    # Plot POMCPOW data
    scatter!(p, 
        pomcpow_xs, 
        pomcpow_ys, 
        xerr=pomcpow_xerr, 
        yerr=pomcpow_yerr,
        label="POMCPOW",
        markercolor=:red,
        markersize=7,
        markershape=:diamond
    )
    
    plot!(p, pomcpow_xs, pomcpow_ys,
        linecolor=:red,
        linestyle=:solid,
        alpha=0.8,
        linewidth=2,
        label=false
    )
    
    # Plot ExploreNSteps data
    scatter!(p, 
        explore_xs, 
        explore_ys, 
        xerr=explore_xerr, 
        yerr=explore_yerr,
        label="ExploreNSteps",
        markercolor=:orange,
        markersize=6,
        markershape=:rect
    )
    
    plot!(p, explore_xs, explore_ys,
        linecolor=:orange,
        linestyle=:dash,
        alpha=0.8,
        linewidth=2,
        label=false
    )
    
    # Plot ImportOnly data
    scatter!(p, 
        import_xs, 
        import_ys, 
        xerr=import_xerr, 
        yerr=import_yerr,
        label="ImportOnly",
        markercolor=:green,
        markersize=6,
        markershape=:utriangle
    )
    
    plot!(p, import_xs, import_ys,
        linecolor=:green,
        linestyle=:dash,
        alpha=0.8,
        linewidth=2,
        label=false
    )
    
    # Random policy display removed
    
    # Add a legend with policy descriptions
    annotate!(p, [
        (minimum(mcts_xs), maximum(mcts_ys) * 1.05, text("MCTS: Utilizes search tree with rollouts", :left, 8)),
        (minimum(mcts_xs), maximum(mcts_ys) * 1.03, text("POMCPOW: Monte Carlo Tree Search for POMDPs", :left, 8)),
        (minimum(mcts_xs), maximum(mcts_ys) * 1.01, text("ExploreNSteps: Simple heuristic with N exploration steps", :left, 8)),
        (minimum(mcts_xs), maximum(mcts_ys) * 0.99, text("ImportOnly: Prioritizes foreign deposits", :left, 8))
    ])
    
    # Save the figure
    savefig(p, "comprehensive_policy_comparison.png")
    return p
end

function main()
    println("Starting comprehensive policy comparison for LiPOMDP...")
    
    # Define parameter values for each policy type
    alpha_values = collect(LinRange(0, 1, 10))  # For MCTS and POMCPOW
    step_values = 5:5:50  # For ExploreNSteps and ImportOnly
    
    # Common settings for all experiments
    stochastic_price = false
    max_steps = 30
    n_reps = 10  # Reduced for faster runtime
    
    # Generate results for all policy types
    println("\nGenerating MCTS results...")
    mcts_results = compute_mcts_results(alpha_values, "alpha", 
                                       stochastic_price=stochastic_price,
                                       max_steps=max_steps, 
                                       n_reps=n_reps)
    
    println("\nGenerating POMCPOW results...")
    pomcpow_results = compute_pomcpow_results(alpha_values, "alpha", 
                                            stochastic_price=stochastic_price,
                                            max_steps=max_steps, 
                                            n_reps=n_reps)
    
    println("\nGenerating ExploreNSteps results...")
    explore_results = compute_explore_n_steps_results(step_values, 
                                                   stochastic_price=stochastic_price,
                                                   max_steps=max_steps, 
                                                   n_reps=n_reps)
    
    println("\nGenerating ImportOnly results...")
    import_results = compute_import_only_results(step_values, 
                                               stochastic_price=stochastic_price,
                                               max_steps=max_steps, 
                                               n_reps=n_reps)
    
    # Random policy generation removed
    
    # Create comprehensive Pareto plot
    println("\nCreating comprehensive Pareto plot...")
    p = plot_comprehensive_pareto(mcts_results, pomcpow_results, explore_results, import_results, random_results)
    
    println("\nAnalysis complete! Results saved to 'comprehensive_policy_comparison.png'")
    
    # Return all results
    results = Dict(
        "mcts" => mcts_results,
        "pomcpow" => pomcpow_results,
        "explore" => explore_results,
        "import" => import_results,
        "random" => random_results
    )
    
    return results, p
end

# Run the main function if executed as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end