# Non-distributed version - completely sequential execution
# Explicitly avoid using any Distributed functionality

using Random
using Statistics
using Dates
using StatsBase  # For countmap

# Try to load other required packages
required_packages = [
    "POMDPs", "POMDPTools", "LiPOMDPs", "MCTS", 
    "DiscreteValueIteration", "POMCPOW", "Distributions", 
    "Parameters"
]

for pkg in required_packages
    try
        @eval using $(Symbol(pkg))
    catch e
        println("Warning: Unable to load package $pkg: $e")
    end
end

# Conditionally load Plots if available (for later use)
has_plotting = false
try
    @eval using Plots
    has_plotting = true
catch e
    println("Note: Plotting package not available. Will save results as CSV only.")
end

# Fix random seed for reproducibility
const RNG = MersenneTwister(1)

# Test if we can initialize LiPOMDP correctly
try
    println("\nTesting initialize_lipomdp function...")
    
    # Try different ways of calling it
    try
        # Method 1: Try positional arguments
        test_pomdp = initialize_lipomdp(0.5, true, true, [300, 300, 300, 300])
        println("Success: initialize_lipomdp works with positional arguments")
        global INIT_METHOD = :positional
    catch e1
        println("Failed with positional arguments: $e1")
        
        try
            # Method 2: Try kwargs with semicolon
            test_pomdp = initialize_lipomdp(; alpha=0.5, stochastic_price=true, compute_tradeoff=true, CO2_cost=[300, 300, 300, 300])
            println("Success: initialize_lipomdp works with kwargs using semicolon")
            global INIT_METHOD = :kwargs_semicolon
        catch e2
            println("Failed with kwargs using semicolon: $e2")
            
            try
                # Method 3: Try using direct parameter construction
                test_pomdp = LiPOMDP(alpha=0.5, stochastic_price=true, compute_tradeoff=true, CO2_cost=[300, 300, 300, 300])
                println("Success: LiPOMDP constructor works directly")
                global INIT_METHOD = :constructor
            catch e3
                println("Failed with direct constructor: $e3")
                
                println("\nWARNING: Could not find a working method to initialize LiPOMDP.")
                println("Please check the LiPOMDPs package documentation for the correct way to initialize.")
                global INIT_METHOD = :unknown
            end
        end
    end
catch e
    println("Error during initialize_lipomdp testing: $e")
    global INIT_METHOD = :unknown
end

# Function to initialize a LiPOMDP based on discovered method
function create_lipomdp(alpha, stochastic_price, compute_tradeoff, co2_cost)
    if INIT_METHOD == :positional
        return initialize_lipomdp(alpha, stochastic_price, compute_tradeoff, co2_cost)
    elseif INIT_METHOD == :kwargs_semicolon
        return initialize_lipomdp(; alpha=alpha, stochastic_price=stochastic_price, 
                                compute_tradeoff=compute_tradeoff, CO2_cost=co2_cost)
    elseif INIT_METHOD == :constructor
        return LiPOMDP(alpha=alpha, stochastic_price=stochastic_price, 
                     compute_tradeoff=compute_tradeoff, CO2_cost=co2_cost)
    else
        error("No working method found to initialize LiPOMDP")
    end
end

# Compute mean and standard error from samples
function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean=sample_mean, se=sample_se)
end

# Run experiment to evaluate policies
function experiment(planner, pomdp, n_reps=20, max_steps=30; initial_belief=nothing)
    # Arrays to store results
    npv_all = Float64[]
    emission_cost_all = Float64[]
    co2_emitted_all = Float64[]
    domestic_all = Float64[]
    imported_all = Float64[]
    action_trajectories = [String[] for _ in 1:max_steps]

    try
        for t in 1:n_reps
            # Show progress
            print("\rRunning simulation $(t)/$(n_reps) [$(round(Int, 100*t/n_reps))%]")
            
            # Variables to track total values
            npv_tot = 0.0
            emission_cost_tot = 0.0
            co2_emitted_tot = 0.0
            domestic_tot = 0.0
            imported_tot = 0.0
            disc = 1.0  # Discount factor
            step_idx = 1

            # Set up the step iterator
            if initial_belief !== nothing
                up = updater(planner) 
                step_iter = stepthrough(pomdp, planner, up, initial_belief, "s,a,o,r", max_steps=max_steps)
            else
                step_iter = stepthrough(pomdp, planner, "s,a,o,r", max_steps=max_steps)
            end

            # Simulate steps
            for (s, a, o, r) in step_iter
                # Calculate NPV separately for tracking
                npv = compute_npv(pomdp, s, a)
                npv_tot += npv * disc
                
                # Calculate emission cost separately for tracking
                emission_cost = compute_emission_cost(pomdp, s, a)
                emission_cost_tot += emission_cost * disc
                
                # Track actual CO2 emissions (not the cost)
                action_type = get_action_type(a)
                site_num = get_site_number(a)
                
                # Calculate new emissions
                new_emission = 0
                if action_type == "MINE" && !s.have_mined[site_num]
                    new_emission = pomdp.CO2_emissions
                end
                
                # Add existing emissions
                for i in 1:pomdp.n_deposits
                    if s.have_mined[i]
                        new_emission += pomdp.CO2_emissions
                    end
                end
                
                co2_emitted_tot += new_emission * disc
                
                # Track mining actions
                if a.a == "MINE1" || a.a == "MINE2"
                    domestic_tot += 1
                elseif a.a == "MINE3" || a.a == "MINE4"
                    imported_tot += 1
                end

                # Record action for this step
                if step_idx <= max_steps
                    push!(action_trajectories[step_idx], a.a)
                end

                step_idx += 1
                disc *= discount(pomdp)
            end

            # Store results for this run
            push!(npv_all, npv_tot)
            push!(emission_cost_all, emission_cost_tot)
            push!(co2_emitted_all, co2_emitted_tot)
            push!(domestic_all, domestic_tot)
            push!(imported_all, imported_tot)
        end
        println("\nSimulations complete!")
    catch e
        println("\nWarning: Simulation error: $e")
        println("Processing partial results ($(length(npv_all)) completed runs)...")
    end

    # Compute metrics across all runs
    if isempty(npv_all)
        error("No simulation data collected")
    end
    
    results = Dict(
        "NPV" => compute_metrics(npv_all),
        "Emission Cost" => compute_metrics(emission_cost_all),
        "CO2 Emitted" => compute_metrics(co2_emitted_all),
        "Domestic Mining" => compute_metrics(domestic_all),
        "Imported Mining" => compute_metrics(imported_all),
        "Total Mining" => compute_metrics(domestic_all .+ imported_all)
    )

    return results, action_trajectories
end

# Policy creation functions
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

# CSV data export functions
function save_results_to_csv(results_dict, filename_prefix)
    try
        # Save overall results
        open("$(filename_prefix)_summary.csv", "w") do io
            println(io, "Policy,Alpha,Metric,Mean,SE")
            for (policy, alpha_dict) in results_dict
                for (alpha, metrics) in alpha_dict
                    for (metric_name, metric_value) in metrics
                        mean_val = metric_value.mean
                        se_val = metric_value.se
                        println(io, "$policy,$alpha,$metric_name,$mean_val,$se_val")
                    end
                end
            end
        end
        
        # Save Pareto curve data for NPV vs Emission Cost
        open("$(filename_prefix)_emission_pareto.csv", "w") do io
            println(io, "Policy,Alpha,EmissionCost,NPV,EmissionCost_SE,NPV_SE")
            for (policy, alpha_dict) in results_dict
                for alpha in sort(collect(keys(alpha_dict)))
                    emission_cost = -alpha_dict[alpha]["Emission Cost"].mean
                    npv = alpha_dict[alpha]["NPV"].mean
                    emission_cost_se = alpha_dict[alpha]["Emission Cost"].se
                    npv_se = alpha_dict[alpha]["NPV"].se
                    println(io, "$policy,$alpha,$emission_cost,$npv,$emission_cost_se,$npv_se")
                end
            end
        end
        
        # Save Pareto curve data for NPV vs CO2 Emitted
        open("$(filename_prefix)_co2_pareto.csv", "w") do io
            println(io, "Policy,Alpha,CO2Emitted,NPV,CO2Emitted_SE,NPV_SE")
            for (policy, alpha_dict) in results_dict
                for alpha in sort(collect(keys(alpha_dict)))
                    co2_emitted = -alpha_dict[alpha]["CO2 Emitted"].mean
                    npv = alpha_dict[alpha]["NPV"].mean
                    co2_emitted_se = alpha_dict[alpha]["CO2 Emitted"].se
                    npv_se = alpha_dict[alpha]["NPV"].se
                    println(io, "$policy,$alpha,$co2_emitted,$npv,$co2_emitted_se,$npv_se")
                end
            end
        end
        
        println("Results saved to CSV files with prefix: $filename_prefix")
        return true
    catch e
        println("Warning: Failed to save results to CSV: $e")
        return false
    end
end

function save_actions_to_csv(action_logs_dict, filename_prefix)
    try
        for (policy_name, alphas) in action_logs_dict
            for alpha in keys(alphas)
                action_trajectories = action_logs_dict[policy_name][alpha]
                
                open("$(filename_prefix)_$(policy_name)_alpha$(alpha)_actions.csv", "w") do io
                    println(io, "Step,Action,Count")
                    for (step, actions) in enumerate(action_trajectories)
                        # Count occurrences of each action
                        action_counts = Dict{String, Int}()
                        for action in actions
                            action_counts[action] = get(action_counts, action, 0) + 1
                        end
                        
                        # Write to file
                        for (action, count) in action_counts
                            println(io, "$step,$action,$count")
                        end
                    end
                end
            end
        end
        
        println("Action data saved to CSV files with prefix: $filename_prefix")
        return true
    catch e
        println("Warning: Failed to save action data to CSV: $e")
        return false
    end
end

# Try to create plots if possible
function try_create_plots(results_dict, filename_prefix)
    if !has_plotting
        println("Plotting not available. Results saved as CSV only.")
        return nothing, nothing
    end
    
    println("\nAttempting to generate plots...")
    
    try
        # Try different backends
        backend_found = false
        for backend in [:pyplot, :gr, :plotlyjs]
            try
                @eval Plots.$(backend)()
                println("Using $backend for plotting")
                backend_found = true
                break
            catch
                println("Backend $backend not available")
            end
        end
        
        if !backend_found
            println("No suitable plotting backend found. Saving data as CSV only.")
            return nothing, nothing
        end
        
        # Print debug info about data
        println("\nData available for plotting:")
        for (policy, alphas) in results_dict
            println("Policy: $policy has $(length(alphas)) alpha values")
            for alpha in sort(collect(keys(alphas)))
                if haskey(alphas[alpha], "NPV") && haskey(alphas[alpha], "CO2 Emitted")
                    npv = alphas[alpha]["NPV"].mean
                    co2 = alphas[alpha]["CO2 Emitted"].mean
                    println("  α=$alpha: NPV=$npv, CO2=$co2")
                else
                    println("  α=$alpha: Missing NPV or CO2 data")
                end
            end
        end
        
        # Create simple plots
        p1 = plot(title="NPV vs Emission Cost", xlabel="Emission Cost", ylabel="NPV", legend=:topright)
        p2 = plot(title="NPV vs CO2 Emitted", xlabel="CO2 Emitted", ylabel="NPV", legend=:topright)
        
        colors = [:blue, :red, :green, :orange]
        
        # Auto-detect axis limits
        all_xs_emission = Float64[]
        all_ys_npv = Float64[]
        all_xs_co2 = Float64[]
        
        for (i, policy_name) in enumerate(keys(results_dict))
            xs_emission = Float64[]
            ys_emission = Float64[]
            xs_co2 = Float64[]
            ys_co2 = Float64[]
            
            for alpha in sort(collect(keys(results_dict[policy_name])))
                # Try both negative and positive values since we're not sure of the sign convention
                emission_cost = results_dict[policy_name][alpha]["Emission Cost"].mean
                co2_emitted = results_dict[policy_name][alpha]["CO2 Emitted"].mean
                npv = results_dict[policy_name][alpha]["NPV"].mean
                
                # Store original values for axis scaling
                push!(all_xs_emission, emission_cost)
                push!(all_xs_co2, co2_emitted)
                push!(all_ys_npv, npv)
                
                # For plotting
                push!(xs_emission, emission_cost)
                push!(ys_emission, npv)
                push!(xs_co2, co2_emitted)
                push!(ys_co2, npv)
            end
            
            # Only plot if we have data
            if !isempty(xs_emission)
                # Plot emission cost curve
                plot!(p1, xs_emission, ys_emission, 
                      label=policy_name, linecolor=colors[i], marker=:circle, markercolor=colors[i])
                
                # Plot CO2 emitted curve  
                plot!(p2, xs_co2, ys_co2, 
                      label=policy_name, linecolor=colors[i], marker=:circle, markercolor=colors[i])
            end
        end
        
        # Set appropriate axis limits if we have data
        if !isempty(all_xs_emission)
            margin = 0.1  # 10% margin
            
            # For emission plot
            xmin_e = minimum(all_xs_emission)
            xmax_e = maximum(all_xs_emission)
            x_range_e = xmax_e - xmin_e
            x_margin_e = max(x_range_e * margin, 1.0)  # At least 1.0 margin
            
            # For CO2 plot
            xmin_c = minimum(all_xs_co2)
            xmax_c = maximum(all_xs_co2)
            x_range_c = xmax_c - xmin_c
            x_margin_c = max(x_range_c * margin, 1.0)  # At least 1.0 margin
            
            # For NPV (y-axis on both plots)
            ymin = minimum(all_ys_npv)
            ymax = maximum(all_ys_npv)
            y_range = ymax - ymin
            y_margin = max(y_range * margin, 1.0)  # At least 1.0 margin
            
            # Apply limits
            plot!(p1, xlim=(xmin_e - x_margin_e, xmax_e + x_margin_e),
                 ylim=(ymin - y_margin, ymax + y_margin))
            plot!(p2, xlim=(xmin_c - x_margin_c, xmax_c + x_margin_c),
                 ylim=(ymin - y_margin, ymax + y_margin))
        end
        
        savefig(p1, "$(filename_prefix)_emission_pareto.png")
        savefig(p2, "$(filename_prefix)_co2_pareto.png")
        println("Plots saved successfully!")
        
        return p1, p2
    catch e
        println("Warning: Failed to generate plots: $e")
        println("Results have been saved to CSV files.")
        return nothing, nothing
    end
end

# Main function - completely sequential approach
function main()
    n_reps = 20  # Number of repetitions
    max_steps = 30  # Maximum steps per simulation
    stochastic_price = true
    
    # Define alpha values to test
    alpha_values = [0.1, 0.5, 1.0]
    
    # Store results for each policy
    results_dict = Dict(
        "POMCPOW" => Dict(),
        "MCTS" => Dict(),
        "ExploreNSteps" => Dict(),
        "ImportOnly" => Dict()
    )

    # Store action trajectories for visualization
    action_logs_dict = Dict(
        "POMCPOW" => Dict{Float64, Vector{Vector{String}}}(),
        "MCTS" => Dict{Float64, Vector{Vector{String}}}(),
        "ExploreNSteps" => Dict{Float64, Vector{Vector{String}}}(),
        "ImportOnly" => Dict{Float64, Vector{Vector{String}}}()
    )
    
    # For each policy type, test with different alpha values
    for policy_type in keys(results_dict)
        println("\nTesting $policy_type with different alpha values:")

        for alpha in alpha_values
            println("  Testing alpha = $alpha")
            
            try
                # Create POMDP with this alpha and consistent CO2 costs using our helper function
                pomdp = create_lipomdp(
                    alpha,
                    stochastic_price,
                    true,  # compute_tradeoff
                    [300, 300, 300, 300]  # CO2_cost
                )
                
                # Create and evaluate the appropriate planner
                local results, action_trajectories
                
                if policy_type == "POMCPOW"
                    planner = create_pomcpow_planner(pomdp)
                    results, action_trajectories = experiment(planner, pomdp, n_reps, max_steps)
                elseif policy_type == "MCTS"
                    planner = create_mcts_planner(pomdp)
                    results, action_trajectories = experiment(planner, pomdp, n_reps, max_steps)
                elseif policy_type == "ExploreNSteps"
                    planner = create_explore_n_steps_planner(pomdp, 20)
                    results, action_trajectories = experiment(planner, pomdp, n_reps, max_steps)
                elseif policy_type == "ImportOnly"
                    # Handle ImportOnly specially
                    try
                        # First try to use the initialize_belief_import_only function if it exists
                        initial_belief = initialize_belief_import_only(LiBeliefUpdater(pomdp))
                        println("    Using specialized initialize_belief_import_only")
                    catch e
                        # If it doesn't exist, use regular initialize_belief
                        println("    Warning: initialize_belief_import_only not found, using standard initialize_belief")
                        initial_belief = initialize_belief(LiBeliefUpdater(pomdp))
                    end
                    
                    planner = create_import_only_planner(pomdp, 20)
                    results, action_trajectories = experiment(planner, pomdp, n_reps, max_steps, initial_belief=initial_belief)
                end
                
                # Store results and action logs 
                results_dict[policy_type][alpha] = results
                action_logs_dict[policy_type][alpha] = action_trajectories
                
            catch e
                println("Error processing $policy_type with alpha=$alpha: $e")
                println("Continuing with next configuration...")
            end
        end
    end

    # Save results to CSV files 
    println("\nSaving results to CSV...")
    timestamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    filename_prefix = "pareto_results_$(timestamp)"
    
    save_results_to_csv(results_dict, filename_prefix)
    save_actions_to_csv(action_logs_dict, filename_prefix)
    
    # Try to create plots
    p1, p2 = try_create_plots(results_dict, filename_prefix)
    
    return results_dict, action_logs_dict, p1, p2
end

# Run the main function
results_dict, action_logs_dict, p1, p2 = main()