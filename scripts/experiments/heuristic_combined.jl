using Random
using POMDPs
using POMDPTools
using LiPOMDPs
using MCTS
using DiscreteValueIteration
using POMCPOW
using Distributions
using Parameters
using ARDESPOT
using Plots
using Statistics
using ProgressBars

"""
This script compares two different lithium sourcing policies:
1. ImportOnlyPolicy - prioritizes foreign deposits
2. ExploreNStepsPolicy - balances exploration and exploitation

It generates Pareto curves for both policies and plots them on the same graph
for easy comparison of their emission-volume trade-offs.
"""

rng = MersenneTwister(1)

function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean=sample_mean, se=sample_se)
end

# Unified experiment function that can handle both with and without custom initial belief
function experiment(planner, eval_pomdp, n_reps=100, max_steps=30; initial_belief=nothing)
    reward_tot_all = []
    reward_disc_all = []
    emission_tot_all = []
    emission_disc_all = []
    domestic_tot_all = []
    imported_tot_all = []

    for t = tqdm(1:n_reps)
        reward_tot = 0.0
        reward_disc = 0.0
        emission_tot = 0.0
        emission_disc = 0.0
        vol_tot = 0.0 #mined domestically
        imported_tot = 0.0 #imported/mined internationally
        disc = 1.0

        # Use custom initial belief if provided, otherwise use default
        if initial_belief !== nothing
            up = updater(planner) 
            step_iter = stepthrough(eval_pomdp, planner, up, initial_belief, "s,a,o,r", max_steps=max_steps)
        else
            step_iter = stepthrough(eval_pomdp, planner, "s,a,o,r", max_steps=max_steps)
        end

        for (s, a, o, r) in step_iter
            #compute reward and discounted reward
            reward_tot += r
            reward_disc += r * disc

            #compute emissions and discount emissions
            e = get_action_emission(eval_pomdp, a)
            emission_tot += e
            emission_disc += e * disc

            if a.a == "MINE1" || a.a == "MINE2"
                vol_tot += 1
            elseif a.a == "MINE3" || a.a == "MINE4"
                imported_tot += 1
            end

            disc *= discount(eval_pomdp)
        end
        push!(reward_tot_all, reward_tot)
        push!(reward_disc_all, reward_disc)
        push!(emission_tot_all, emission_tot)
        push!(emission_disc_all, emission_disc)
        push!(domestic_tot_all, vol_tot)
        push!(imported_tot_all, imported_tot)
    end

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

function display_results(results)
    for metric in keys(results)
        println(metric, ": ", results[metric])
    end
end

# Function to compute results for ExploreNStepsPolicy
function compute_explore_n_steps_results(num_steps=1, stochastic_price=false, max_steps=100)
    # Initialize POMDP
    pomdp = initialize_lipomdp(stochastic_price=stochastic_price, compute_tradeoff=true)

    # Create the ExploreNStepsPolicy planner
    policy = ExploreNStepsPolicy(pomdp=pomdp, explore_steps=num_steps, curr_steps=1)

    # Run experiment with the policy
    results = experiment(policy, pomdp, 1000, max_steps)
    return results
end

# Function to compute results for ImportOnlyPolicy
function compute_import_only_results(num_steps=1, stochastic_price=false, max_steps=100)
    # Initialize POMDP
    pomdp = initialize_lipomdp(stochastic_price=stochastic_price, compute_tradeoff=true)
    updater = LiBeliefUpdater(pomdp) 
    
    # Initialize custom belief that favors import
    initial_belief = initialize_belief_import_only(updater)
    
    # Create the ImportOnlyPolicy planner
    policy = ImportOnlyPolicy(pomdp=pomdp, explore_steps=num_steps)
    
    # Run experiment with the policy and custom belief
    results = experiment(policy, pomdp, 1000, max_steps; initial_belief=initial_belief)
    return results
end

# Function to plot combined Pareto curves
function plot_combined_pareto(explore_results, import_results)
    num_steps = 1:5:100  # Steps to evaluate

    # Extract data for ExploreNStepsPolicy
    explore_xs = [explore_results[steps]["emissions"][1] for steps in num_steps]
    explore_ys = [explore_results[steps]["volume"][1] for steps in num_steps]
    explore_xerror = [explore_results[steps]["emissions"][2] for steps in num_steps]
    explore_yerror = [explore_results[steps]["volume"][2] for steps in num_steps]

    # Sort values for ExploreNStepsPolicy
    explore_indices = sortperm(explore_xs)
    explore_xs = explore_xs[explore_indices]
    explore_ys = explore_ys[explore_indices]
    explore_xerror = explore_xerror[explore_indices]
    explore_yerror = explore_yerror[explore_indices]

    # Extract data for ImportOnlyPolicy
    import_xs = [import_results[steps]["emissions"][1] for steps in num_steps]
    import_ys = [import_results[steps]["volume"][1] for steps in num_steps]
    import_xerror = [import_results[steps]["emissions"][2] for steps in num_steps]
    import_yerror = [import_results[steps]["volume"][2] for steps in num_steps]

    # Sort values for ImportOnlyPolicy
    import_indices = sortperm(import_xs)
    import_xs = import_xs[import_indices]
    import_ys = import_ys[import_indices]
    import_xerror = import_xerror[import_indices]
    import_yerror = import_yerror[import_indices]

    # Create the combined plot
    p = plot(
        xlabel="Total Emissions",
        ylabel="Total Volume",
        title="Pareto Tradeoff: Policy Comparison",
        legend=:topright,
        grid=true,
        gridalpha=0.3,
        linewidth=2.5,
        size=(1000, 600),
        background_color=:white,
        foreground_color=:black,
    )

    # Add ExploreNStepsPolicy data
    scatter!(p, 
        explore_xs, 
        explore_ys, 
        xerr=explore_xerror, 
        yerr=explore_yerror,
        label="Explore N Steps Policy",
        markercolor=:orange,
        markersize=6
    )

    # Add ImportOnlyPolicy data
    scatter!(p, 
        import_xs, 
        import_ys, 
        xerr=import_xerror, 
        yerr=import_yerror,
        label="Import Only Policy",
        markercolor=:yellow,
        markersize=6
    )

    # Connect points with lines
    plot!(p, explore_xs, explore_ys,
        linecolor=:black,
        linestyle=:solid,
        alpha=0.7,
        label=false
    )

    plot!(p, import_xs, import_ys,
        linecolor=:black,
        linestyle=:dash,
        alpha=0.7,
        label=false
    )

    # Add text annotations for selected points
    annotate_points = [1, 30, 95]  # Beginning, middle, end
    
    for step_idx in annotate_points
        # Find the closest step value
        closest_step = num_steps[argmin(abs.(num_steps .- step_idx))]
        
        # Annotate ExploreNStepsPolicy point
        explore_idx = findfirst(x -> x == closest_step, num_steps[explore_indices])
        if explore_idx !== nothing
            annotate!(p, [(explore_xs[explore_idx], explore_ys[explore_idx] + 1, 
                      text("n=$(closest_step)", :red, :top, 8))])
        end
        
        # Annotate ImportOnlyPolicy point
        import_idx = findfirst(x -> x == closest_step, num_steps[import_indices])
        if import_idx !== nothing
            annotate!(p, [(import_xs[import_idx], import_ys[import_idx] - 1, 
                      text("n=$(closest_step)", :blue, :bottom, 8))])
        end
    end

    # Save the figure
    savefig(p, "combined_pareto_curve.png")
    return p
end

function main()
    max_steps = 100  # Maximum steps to simulate
    num_steps_values = 1:5:100  # Values to test

    # Store results for both policies
    explore_results = Dict()
    import_results = Dict()

    # Generate results for ExploreNStepsPolicy
    println("\nGenerating Pareto curve for Explore N Steps Policy...")
    for num_steps in tqdm(num_steps_values)
        results = compute_explore_n_steps_results(num_steps, false, max_steps)
        
        explore_results[num_steps] = Dict(
            "emissions" => results["Total Emissions"],
            "volume" => (
                results["Total Domestic"].mean + results["Total Imported"].mean,
                sqrt(results["Total Domestic"].se^2 + results["Total Imported"].se^2)
            )
        )
    end

    # Generate results for ImportOnlyPolicy
    println("\nGenerating Pareto curve for Import Only Policy...")
    for num_steps in tqdm(num_steps_values)
        results = compute_import_only_results(num_steps, false, max_steps)
        
        import_results[num_steps] = Dict(
            "emissions" => results["Total Emissions"],
            "volume" => (
                results["Total Domestic"].mean + results["Total Imported"].mean,
                sqrt(results["Total Domestic"].se^2 + results["Total Imported"].se^2)
            )
        )
    end

    # Create the combined Pareto plot
    println("\nCreating combined Pareto plot...")
    p = plot_combined_pareto(explore_results, import_results)
    
    println("\nAnalysis complete! Results saved to 'combined_pareto_curve.png'")
    
    # Compare the best points of each policy
    best_explore_vol = maximum([explore_results[steps]["volume"][1] for steps in num_steps_values])
    best_import_vol = maximum([import_results[steps]["volume"][1] for steps in num_steps_values])
    
    println("\nBest volume achieved:")
    println("- Explore N Steps Policy: $(round(best_explore_vol, digits=2))")
    println("- Import Only Policy: $(round(best_import_vol, digits=2))")
    
    return p, explore_results, import_results
end

main()