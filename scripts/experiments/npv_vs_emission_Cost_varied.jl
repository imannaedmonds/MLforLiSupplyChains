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
This script creates a Pareto curve comparing NPV versus emission costs
for different carbon prices. It uses the CO2_cost vector defined in the
LiPOMDP struct to compare policies:
- MCTS
- POMCPOW
- ExploreNSteps
- ImportOnly
"""
rng = MersenneTwister(1)

function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean=sample_mean, se=sample_se)
end

# Modified experiment function to track NPV and emission costs separately
function experiment(planner, eval_pomdp, n_reps=20, max_steps=30; initial_belief=nothing)
    npv_all = []
    emission_cost_all = []
    domestic_tot_all = []
    imported_tot_all = []

    for t in tqdm(1:n_reps)
        npv_tot = 0.0
        emission_cost_tot = 0.0
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
            # Calculate NPV and emission costs separately
            npv = compute_npv(eval_pomdp, s, a)
            npv_tot += npv * disc
            
            emission_cost = -compute_emission_cost(eval_pomdp, s, a)  # Negate to get positive cost
            emission_cost_tot += emission_cost * disc

            # Track domestic vs imported mining
            if a.a == "MINE1" || a.a == "MINE2"
                vol_tot += 1
            elseif a.a == "MINE3" || a.a == "MINE4"
                imported_tot += 1
            end

            disc *= discount(eval_pomdp)
        end
        
        # Store results from this repetition
        push!(npv_all, npv_tot)
        push!(emission_cost_all, emission_cost_tot)
        push!(domestic_tot_all, vol_tot)
        push!(imported_tot_all, imported_tot)
    end

    # Compute metrics for all repetitions
    results = Dict(
        "NPV" => compute_metrics(npv_all),
        "Emission Cost" => compute_metrics(emission_cost_all),
        "Total Domestic" => compute_metrics(domestic_tot_all),
        "Total Imported" => compute_metrics(imported_tot_all)
    )

    return results
end

# Function to compute policy results for all carbon prices
function compute_policy_results(policy_generator; 
                               stochastic_price=false, max_steps=30, n_reps=20,
                               initial_belief_generator=nothing,
                               policy_name="Policy")
    carbon_prices = [80, 200, 400, 600]
    results = Dict()
    
    for (idx, carbon_price) in enumerate(carbon_prices)
        println("\nTesting $policy_name with carbon price: $carbon_price")
        
        # Create a POMDP with a single carbon price
        # We'll modify the LiPOMDP to use just one price from the vector
        pomdp = initialize_lipomdp(
            stochastic_price=stochastic_price, 
            compute_tradeoff=true,
            CO2_cost=[carbon_price, carbon_price, carbon_price, carbon_price]  # Use the same price for all sites
        )
        
        # Generate initial belief if a generator is provided
        initial_belief = nothing
        if initial_belief_generator !== nothing
            initial_belief = initial_belief_generator(pomdp)
        end
        
        # Create policy using the generator function
        policy = policy_generator(pomdp)
        
        # Run experiment
        if initial_belief !== nothing
            results[carbon_price] = experiment(policy, pomdp, n_reps, max_steps, initial_belief=initial_belief)
        else
            results[carbon_price] = experiment(policy, pomdp, n_reps, max_steps)
        end
    end
    
    return results
end

# MCTS generator function
function mcts_generator(pomdp)
    train_up = LiBeliefUpdater(pomdp)
    mdp = GenerativeBeliefMDP(pomdp, train_up, terminal_behavior=ContinueTerminalBehavior(pomdp, train_up))
    rollout_policy = EfficiencyPolicyWithUncertainty(pomdp, 1.0, [true, true, true, true])
    
    mcts_solver = DPWSolver(
        depth=10,
        n_iterations=100,
        estimate_value=RolloutEstimator(rollout_policy),
        enable_action_pw=false,
        enable_state_pw=true,
        k_state=4.0,
        alpha_state=0.1
    )
    
    return solve(mcts_solver, mdp)
end

# POMCPOW generator function
function pomcpow_generator(pomdp)
    solver = POMCPOW.POMCPOWSolver(
        tree_queries=1000,
        estimate_value=estimate_value,
        k_observation=4.0,
        alpha_observation=0.1,
        max_depth=15,
        enable_action_pw=false,
        init_N=10
    )
    
    return solve(solver, pomdp)
end

# ExploreNSteps generator function
function explore_generator(pomdp)
    return ExploreNStepsPolicy(pomdp=pomdp, explore_steps=20, curr_steps=1)
end

# ImportOnly generator function
function import_generator(pomdp)
    return ImportOnlyPolicy(pomdp=pomdp, explore_steps=20)
end

# ImportOnly belief generator function
function import_belief_generator(pomdp)
    updater = LiBeliefUpdater(pomdp)
    return initialize_belief_import_only(updater)
end

# Process results for NPV vs Emission Cost plotting
function process_npv_emission_results(results)
    carbon_prices = sort(collect(keys(results)))
    
    xs = [results[price]["Emission Cost"].mean for price in carbon_prices]
    ys = [results[price]["NPV"].mean for price in carbon_prices]
    xerr = [results[price]["Emission Cost"].se for price in carbon_prices]
    yerr = [results[price]["NPV"].se for price in carbon_prices]
    
    # Sort by emission cost for better curve visualization
    idxs = sortperm(xs)
    xs = xs[idxs]
    ys = ys[idxs]
    xerr = xerr[idxs]
    yerr = yerr[idxs]
    sorted_prices = carbon_prices[idxs]
    
    return xs, ys, xerr, yerr, sorted_prices
end

# Plot combined NPV vs Emission Cost Pareto curve for all policies
function plot_combined_npv_emission_pareto(results_dict)
    # First, let's print the values to understand the data range
    println("\nPolicy data ranges:")
    for (policy_name, results) in results_dict
        xs, ys, xerr, yerr, sorted_prices = process_npv_emission_results(results)
        println("$policy_name:")
        for i in 1:length(xs)
            println("  Price: $(sorted_prices[i]), NPV: $(ys[i]), Emission Cost: $(xs[i])")
        end
    end
    
    # Create main plot
    p = plot(
        xlabel="Emission Cost",
        ylabel="Net Present Value (NPV)",
        title="NPV vs Emission Cost: Carbon Price Comparison",
        legend=:topright,
        grid=true,
        gridalpha=0.3,
        size=(1200, 800),
        background_color=:white,
        foreground_color=:black,
        margin=10Plots.mm,
        xscale=:log10  # Use log scale for x-axis to spread out values
    )
    
    # Colors for different policies
    colors = [:blue, :red, :green, :orange]
    markers = [:circle, :diamond, :rect, :utriangle]
    
    # Plot each policy
    i = 1
    for (policy_name, results) in results_dict
        xs, ys, xerr, yerr, carbon_prices = process_npv_emission_results(results)
        
        # Make sure we don't have zero or negative values for log scale
        for j in 1:length(xs)
            if xs[j] <= 0
                xs[j] = 1.0  # Small positive value for log scale
            end
        end
        
        # Increase marker size for ImportOnly policy
        marker_size = policy_name == "ImportOnly" ? 12 : 7
        
        scatter!(p, 
            xs, 
            ys, 
            xerr=xerr, 
            yerr=yerr,
            label=policy_name,
            markercolor=colors[i],
            markersize=marker_size,
            markershape=markers[i]
        )
        
        plot!(p, xs, ys,
            linecolor=colors[i],
            linestyle=:solid,
            alpha=0.8,
            linewidth=2,
            label=false
        )
        
        # Add annotations for carbon prices
        for j in 1:length(xs)
            annotate!(p, [(xs[j], ys[j] + 0.05*ys[j], text("$(carbon_prices[j])", 8, colors[i]))])
        end
        
        i = i % length(colors) + 1
    end
    
    # Get plot limits for placing annotations
    x_min, x_max = xlims(p)
    y_min, y_max = ylims(p)
    
    # Add a legend with policy descriptions
    # Position annotations near the top of the plot
    y_top = y_max * 0.95
    y_step = (y_max - y_min) * 0.02  # 2% of plot height
    
    savefig(p, "npv_emission_carbon_price_pareto.png")
    
    return p
end

function main()
    println("Starting NPV vs Emission Cost analysis with varying carbon prices...")
    
    # Common settings
    stochastic_price = false
    max_steps = 150
    n_reps = 50 # Reduced for faster runtime
    
    # Generate results for different policies
    results_dict = Dict()

    println("\nGenerating ExploreNSteps results with varying carbon prices...")
    explore_results = compute_policy_results(
        explore_generator,
        stochastic_price=stochastic_price,
        max_steps=max_steps, 
        n_reps=n_reps,
        policy_name="ExploreNSteps"
    )
    results_dict["ExploreNSteps"] = explore_results
    
    println("\nGenerating ImportOnly results with varying carbon prices...")
    import_results = compute_policy_results(
        import_generator,
        stochastic_price=stochastic_price,
        max_steps=max_steps, 
        n_reps=n_reps,
        initial_belief_generator=import_belief_generator,
        policy_name="ImportOnly"
    )
    results_dict["ImportOnly"] = import_results
    
    println("\nGenerating MCTS results with varying carbon prices...")
    mcts_results = compute_policy_results(
        mcts_generator,
        stochastic_price=stochastic_price,
        max_steps=max_steps, 
        n_reps=n_reps,
        policy_name="MCTS"
    )
    results_dict["MCTS"] = mcts_results
    
    println("\nGenerating POMCPOW results with varying carbon prices...")
    pomcpow_results = compute_policy_results(
        pomcpow_generator,
        stochastic_price=stochastic_price,
        max_steps=max_steps, 
        n_reps=n_reps,
        policy_name="POMCPOW"
    )
    results_dict["POMCPOW"] = pomcpow_results
    
    
    # Create combined Pareto plot
    plot_combined_npv_emission_pareto(results_dict)
    
    println("\nNPV vs Emission Cost analysis complete!")
    println("Check the generated 'npv_emission_carbon_price_pareto.png' file for the Pareto curve.")
    
    return results_dict
end

# Run the main function
main()