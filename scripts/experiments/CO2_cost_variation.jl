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
This script generates Pareto curves showing the tradeoff between NPV and emission costs
by varying the CO2_cost parameter in the LiPOMDP model.

CO2 costs tested: [80, 200, 400, 600]
"""

rng = MersenneTwister(1)

function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean=sample_mean, se=sample_se)
end

# Experiment function to evaluate policies with CO2 cost variations
function experiment(planner, eval_pomdp, n_reps=20, max_steps=30; initial_belief=nothing)
    # Arrays to store results
    npv_all = []
    emission_cost_all = []
    co2_emitted_all = []
    domestic_all = []
    imported_all = []

    for t in tqdm(1:n_reps)
        # Variables to track total values
        npv_tot = 0.0
        emission_cost_tot = 0.0
        co2_emitted_tot = 0.0
        domestic_tot = 0.0
        imported_tot = 0.0
        disc = 1.0  # Discount factor

        # Set up the step iterator
        if initial_belief !== nothing
            up = updater(planner) 
            step_iter = stepthrough(eval_pomdp, planner, up, initial_belief, "s,a,o,r", max_steps=max_steps)
        else
            step_iter = stepthrough(eval_pomdp, planner, "s,a,o,r", max_steps=max_steps)
        end

        # Simulate steps
        for (s, a, o, r) in step_iter
            # Calculate NPV separately for tracking
            npv = compute_npv(eval_pomdp, s, a)
            npv_tot += npv * disc
            
            # Calculate emission cost separately for tracking
            emission_cost = compute_emission_cost(eval_pomdp, s, a)
            emission_cost_tot += emission_cost * disc
            
            # Track actual CO2 emissions (not the cost)
            action_type = get_action_type(a)
            site_num = get_site_number(a)
            
            # Calculate new emissions
            new_emission = 0
            if action_type == "MINE" && !s.have_mined[site_num]
                new_emission = eval_pomdp.CO2_emissions
            end
            
            # Add existing emissions
            for i in 1:eval_pomdp.n_deposits
                if s.have_mined[i]
                    new_emission += eval_pomdp.CO2_emissions
                end
            end
            
            co2_emitted_tot += new_emission * disc
            
            # Track mining actions
            if a.a == "MINE1" || a.a == "MINE2"
                domestic_tot += 1
            elseif a.a == "MINE3" || a.a == "MINE4"
                imported_tot += 1
            end
            
            disc *= discount(eval_pomdp)
        end
        
        # Store results
        push!(npv_all, npv_tot)
        push!(emission_cost_all, emission_cost_tot)
        push!(co2_emitted_all, co2_emitted_tot)
        push!(domestic_all, domestic_tot)
        push!(imported_all, imported_tot)
    end

    # Compute metrics
    results = Dict(
        "NPV" => compute_metrics(npv_all),
        "Emission Cost" => compute_metrics(emission_cost_all),
        "CO2 Emitted" => compute_metrics(co2_emitted_all),
        "Domestic Mining" => compute_metrics(domestic_all),
        "Imported Mining" => compute_metrics(imported_all),
        "Total Mining" => compute_metrics(domestic_all .+ imported_all)
    )

    return results
end

# Policy generator functions
function create_pomcpow_planner(pomdp)
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

function create_mcts_planner(pomdp)
    up = LiBeliefUpdater(pomdp)
    mdp = GenerativeBeliefMDP(pomdp, up, terminal_behavior=ContinueTerminalBehavior(pomdp, up))
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

function create_explore_n_steps_planner(pomdp, n_steps=20)
    return ExploreNStepsPolicy(pomdp=pomdp, explore_steps=n_steps, curr_steps=1)
end

function create_import_only_planner(pomdp, n_steps=20)
    return ImportOnlyPolicy(pomdp=pomdp, explore_steps=n_steps)
end

function create_import_only_belief(pomdp)
    updater = LiBeliefUpdater(pomdp)
    return initialize_belief_import_only(updater)
end

# Function to plot NPV vs Emission Cost Pareto curve
function plot_co2_cost_pareto(results_dict)
    # Create the plot
    p = plot(
        xlabel="Emission Cost",
        ylabel="Net Present Value (NPV)",
        title="NPV vs Emission Cost Pareto Curve (CO₂ Cost Variation)",
        legend=:topright,
        grid=true,
        gridalpha=0.3,
        size=(900, 600),
        background_color=:white,
        foreground_color=:black,
        margin=10Plots.mm
    )
    
    # Colors for different policies
    policy_colors = Dict(
        "POMCPOW" => :blue,
        "MCTS" => :red,
        "ExploreNSteps" => :green,
        "ImportOnly" => :orange
    )
    
    # Process and plot each policy
    for (policy_name, results) in results_dict
        # Extract CO2 cost values and sort them
        co2_costs = sort(collect(keys(results)))
        
        # Extract data points
        xs = []  # Emission costs (x-axis)
        ys = []  # NPV values (y-axis)
        xerr = []  # Error bars for emission costs
        yerr = []  # Error bars for NPV
        
        for cost in co2_costs
            push!(xs, -results[cost]["Emission Cost"].mean)  # Negate to get positive cost
            push!(ys, results[cost]["NPV"].mean)
            push!(xerr, results[cost]["Emission Cost"].se)
            push!(yerr, results[cost]["NPV"].se)
        end
        
        # Sort by emission cost for better curve
        sorted_indices = sortperm(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]
        xerr = xerr[sorted_indices]
        yerr = yerr[sorted_indices]
        sorted_costs = co2_costs[sorted_indices]
        
        # Plot scatter points with error bars
        scatter!(p, 
            xs, 
            ys, 
            xerr=xerr, 
            yerr=yerr,
            label=policy_name,
            markercolor=policy_colors[policy_name],
            markersize=7,
            markershape=:circle
        )
        
        # Connect the points with a line
        plot!(p, xs, ys,
            linecolor=policy_colors[policy_name],
            linestyle=:solid,
            alpha=0.7,
            linewidth=2,
            label=false
        )
        
        # Add CO2 cost annotations to points
        for (i, cost) in enumerate(sorted_costs)
            annotate!(p, [(xs[i], ys[i] + maximum(ys)/50, text("$cost", 8, policy_colors[policy_name]))])
        end
    end
    
    # Save the figure
    savefig(p, "npv_emission_co2cost_pareto.png")
    return p
end

# Function to plot NPV vs CO2 Emitted
function plot_co2_emitted_pareto(results_dict)
    # Create the plot
    p = plot(
        xlabel="CO₂ Emissions",
        ylabel="Net Present Value (NPV)",
        title="NPV vs CO₂ Emissions (Cost Variation)",
        legend=:topright,
        grid=true,
        gridalpha=0.3,
        size=(900, 600),
        background_color=:white,
        foreground_color=:black,
        margin=10Plots.mm
    )
    
    # Colors for different policies
    policy_colors = Dict(
        "POMCPOW" => :blue,
        "MCTS" => :red,
        "ExploreNSteps" => :green,
        "ImportOnly" => :orange
    )
    
    # Process and plot each policy
    for (policy_name, results) in results_dict
        # Extract CO2 cost values and sort them
        co2_costs = sort(collect(keys(results)))
        
        # Extract data points
        xs = []  # CO2 emissions (x-axis)
        ys = []  # NPV values (y-axis)
        xerr = []  # Error bars for CO2 emissions
        yerr = []  # Error bars for NPV
        
        for cost in co2_costs
            push!(xs, -results[cost]["CO2 Emitted"].mean)  # Negate to get positive emissions
            push!(ys, results[cost]["NPV"].mean)
            push!(xerr, results[cost]["CO2 Emitted"].se)
            push!(yerr, results[cost]["NPV"].se)
        end
        
        # Sort by CO2 emissions for better curve
        sorted_indices = sortperm(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]
        xerr = xerr[sorted_indices]
        yerr = yerr[sorted_indices]
        sorted_costs = co2_costs[sorted_indices]
        
        # Plot scatter points with error bars
        scatter!(p, 
            xs, 
            ys, 
            xerr=xerr, 
            yerr=yerr,
            label=policy_name,
            markercolor=policy_colors[policy_name],
            markersize=7,
            markershape=:circle
        )
        
        # Connect the points with a line
        plot!(p, xs, ys,
            linecolor=policy_colors[policy_name],
            linestyle=:solid,
            alpha=0.7,
            linewidth=2,
            label=false
        )
        
        # Add CO2 cost annotations to points
        for (i, cost) in enumerate(sorted_costs)
            annotate!(p, [(xs[i], ys[i] + maximum(ys)/50, text("Cost: $cost", 8, policy_colors[policy_name]))])
        end
    end
    
    # Save the figure
    savefig(p, "npv_co2_emitted_pareto.png")
    return p
end

function main()
    # Settings
    n_reps = 20  # Number of repetitions
    max_steps = 30  # Maximum steps per simulation
    stochastic_price = false  # Use deterministic prices
    
    # Define CO2 cost values to test
    co2_costs = [80, 200, 400, 600]  # As specified
    
    # Store results for each policy
    results_dict = Dict(
        "POMCPOW" => Dict(),
        "MCTS" => Dict(),
        "ExploreNSteps" => Dict(),
        "ImportOnly" => Dict()
    )
    
    # For each policy type, test with different CO2 costs
    for policy_type in keys(results_dict)
        println("\nTesting $policy_type with different CO2 costs:")
        
        for cost in co2_costs
            println("  Testing CO2 cost = $cost")
            
            # Create POMDP with this CO2 cost (all 4 mines use the same cost)
            pomdp = initialize_lipomdp(
                alpha=0.5,  # Equal weight to NPV and emissions
                stochastic_price=stochastic_price,
                compute_tradeoff=true,
                CO2_cost=[cost, cost, cost, cost]
            )
            
            # Create and evaluate the appropriate planner
            if policy_type == "POMCPOW"
                planner = create_pomcpow_planner(pomdp)
                results = experiment(planner, pomdp, n_reps, max_steps)
            elseif policy_type == "MCTS"
                planner = create_mcts_planner(pomdp)
                results = experiment(planner, pomdp, n_reps, max_steps)
            elseif policy_type == "ExploreNSteps"
                planner = create_explore_n_steps_planner(pomdp, 20)
                results = experiment(planner, pomdp, n_reps, max_steps)
            elseif policy_type == "ImportOnly"
                initial_belief = create_import_only_belief(pomdp)
                planner = create_import_only_planner(pomdp, 20)
                results = experiment(planner, pomdp, n_reps, max_steps, initial_belief=initial_belief)
            end
            
            # Store results
            results_dict[policy_type][cost] = results
        end
    end
    
    # Create Pareto curve plots
    println("\nGenerating Pareto curve plots...")
    p1 = plot_co2_cost_pareto(results_dict)
    p2 = plot_co2_emitted_pareto(results_dict)
    
    println("\nPareto curves generated and saved as:")
    println("  - 'npv_emission_co2cost_pareto.png'")
    println("  - 'npv_co2_emitted_pareto.png'")
    
    return results_dict, p1, p2
end

# Run the main function
main()