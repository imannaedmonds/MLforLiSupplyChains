using Distributed
nworkers() == 2 && addprocs(4)  # Add worker if only running with single worker

@everywhere begin
    using Random, POMDPs, POMDPTools, LiPOMDPs, MCTS, DiscreteValueIteration
    using POMCPOW, Distributions, Parameters, Plots, Statistics, ProgressBars, StatsBase

    const RNG = MersenneTwister(1)

    # Compute mean and standard error from samples
    compute_metrics(samples) = (
        mean = mean(samples),
        se = std(samples) / sqrt(length(samples))
    )

    # Run experiment to evaluate policies with alpha variations
    function experiment(planner, pomdp, n_reps=20, max_steps=30; initial_belief=nothing)
        # Initialize result arrays
        npv_all, emission_cost_all, co2_emitted_all = Float64[], Float64[], Float64[]
        domestic_all, imported_all = Float64[], Float64[]
        action_trajectories = [String[] for _ in 1:max_steps]

        for t in ProgressBars.tqdm(1:n_reps)
            npv_tot, emission_cost_tot, co2_emitted_tot = 0.0, 0.0, 0.0
            domestic_tot, imported_tot = 0.0, 0.0
            disc, step_idx = 1.0, 1

            # Set up step iterator with or without initial belief
            step_iter = initial_belief !== nothing ? 
                stepthrough(pomdp, planner, updater(planner), initial_belief, "s,a,o,r", max_steps=max_steps) :
                stepthrough(pomdp, planner, "s,a,o,r", max_steps=max_steps)

            # Simulate steps
            for (s, a, o, r) in step_iter
                # Calculate and track metrics
                npv_tot += compute_npv(pomdp, s, a) * disc
                emission_cost_tot += compute_emission_cost(pomdp, s, a) * disc
                
                # Track CO2 emissions
                action_type, site_num = get_action_type(a), get_site_number(a)
                new_emission = 0
                
                # New emissions from current action
                if action_type == "MINE" && !s.have_mined[site_num]
                    new_emission = pomdp.CO2_emissions
                end
                
                # Existing emissions
                for i in 1:pomdp.n_deposits
                    s.have_mined[i] && (new_emission += pomdp.CO2_emissions)
                end
                
                co2_emitted_tot += new_emission * disc
                
                # Track mining actions by type
                if a.a ∈ ["MINE1", "MINE2"]
                    domestic_tot += 1
                elseif a.a ∈ ["MINE3", "MINE4"]
                    imported_tot += 1
                end

                # Record action for this step
                step_idx <= max_steps && push!(action_trajectories[step_idx], a.a)
                
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

        # Compute metrics across all runs
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

    # Create planners with appropriate parameters
    function create_pomcpow_planner(pomdp)
        solver = POMCPOW.POMCPOWSolver(
            tree_queries=1000, estimate_vxalue=estimate_value,
            k_observation=4.0, alpha_observation=0.1, max_depth=15, 
            enable_action_pw=false, init_N=10
        )
        return solve(solver, pomdp)
    end

    function create_mcts_planner(pomdp)
        up = LiBeliefUpdater(pomdp)
        mdp = GenerativeBeliefMDP(pomdp, up, terminal_behavior=ContinueTerminalBehavior(pomdp, up))
        rollout_policy = EfficiencyPolicyWithUncertainty(pomdp, 1.0, [true, true, true, true])
        
        mcts_solver = DPWSolver(
            depth=10, n_iterations=100, estimate_value=RolloutEstimator(rollout_policy),
            enable_action_pw=false, enable_state_pw=true, k_state=4.0, alpha_state=0.1
        )
        
        return solve(mcts_solver, mdp)
    end

    # Simple policy creation functions
    create_explore_n_steps_planner(pomdp, n_steps=20) = 
        ExploreNStepsPolicy(pomdp=pomdp, explore_steps=n_steps, curr_steps=1)
    
    create_import_only_planner(pomdp, n_steps=20) = 
        ImportOnlyPolicy(pomdp=pomdp, explore_steps=n_steps)
    
    create_import_only_belief(pomdp) = 
        initialize_belief_import_only(LiBeliefUpdater(pomdp))
end

# Plotting functions
function plot_pareto_curve(results_dict, metric, title, filename)
    policy_colors = Dict("POMCPOW" => :blue, "MCTS" => :red,
                        "ExploreNSteps" => :green, "ImportOnly" => :orange)
    
    p = plot(
        xlabel="$metric", ylabel="Net Present Value (NPV)",
        title="NPV vs $metric (Alpha Variation)",
        legend=:topright, grid=true, gridalpha=0.3, size=(900, 600),
        background_color=:white, foreground_color=:black, margin=10Plots.mm
    )
    
    for (policy_name, results) in results_dict
        alphas = sort(collect(keys(results)))
        xs, ys, xerr, yerr = Float64[], Float64[], Float64[], Float64[]
        
        for alpha in alphas
            push!(xs, -results[alpha][metric].mean)  # Negate for positive values
            push!(ys, results[alpha]["NPV"].mean)
            push!(xerr, results[alpha][metric].se)
            push!(yerr, results[alpha]["NPV"].se)
        end
        
        # Sort for better curve visualization
        sorted_indices = sortperm(xs)
        xs, ys = xs[sorted_indices], ys[sorted_indices]
        xerr, yerr = xerr[sorted_indices], yerr[sorted_indices]
        sorted_alphas = alphas[sorted_indices]
        
        # Plot points, lines, and annotations
        scatter!(p, xs, ys, xerr=xerr, yerr=yerr, label=policy_name, 
                markercolor=policy_colors[policy_name], markersize=7, markershape=:circle)
        
        plot!(p, xs, ys, linecolor=policy_colors[policy_name], 
             linestyle=:solid, alpha=0.7, linewidth=2, label=false)
        
        for (i, alpha) in enumerate(sorted_alphas)
            annotate!(p, [(xs[i], ys[i] + maximum(ys)/50, text("α=$(alpha)", 8, policy_colors[policy_name]))])
        end
    end
    
    savefig(p, filename)
    println("Saved Pareto curve to: $filename")
    return p
end

# Plot histogram of actions
function plot_action_histograms(action_trajectories, alpha, policy_name)
    for (t, actions) in enumerate(action_trajectories)
        counts = countmap(actions)
        p = bar(collect(keys(counts)), collect(values(counts)),
            xlabel="Action", ylabel="Frequency",
            title="Policy: $policy_name | α=$alpha | Step $t",
            legend=false, size=(600, 400), rotation=45
        )
        filename = "histogram_$(policy_name)_alpha$(alpha)_step$t.png"
        savefig(p, filename)
    end
end

# Main function
function main()
    n_reps, max_steps = 10, 20
    alpha_values = [0.1, 0.5, 1.0]
    
    # Initialize results dictionaries
    policy_types = ["POMCPOW", "MCTS", "ExploreNSteps", "ImportOnly"]
    results_dict = Dict(policy => Dict() for policy in policy_types)
    action_logs_dict = Dict(policy => Dict{Float64, Vector{Vector{String}}}() for policy in policy_types)
    
    # For each policy type, test with different alpha values
    for policy_type in policy_types
        println("\nTesting $policy_type with different alpha values:")

        # Use pmap for better parallelization
        results = pmap(alpha_values) do alpha
            println("  Testing alpha = $alpha for $policy_type")
            
            # Create POMDP with this alpha
            pomdp = initialize_lipomdp(
                alpha=alpha, stochastic_price=true, compute_tradeoff=true,
                CO2_cost=[300, 300, 300, 300]  # Medium CO2 cost for all mines
            )
            
            # Create planner and run experiment
            local planner
            if policy_type == "POMCPOW"
                planner = create_pomcpow_planner(pomdp)
            elseif policy_type == "MCTS"
                planner = create_mcts_planner(pomdp)
            elseif policy_type == "ExploreNSteps"
                planner = create_explore_n_steps_planner(pomdp, 20)
            elseif policy_type == "ImportOnly"
                initial_belief = create_import_only_belief(pomdp)
                planner = create_import_only_planner(pomdp, 20)
                experiment_results, action_traj = experiment(planner, pomdp, n_reps, max_steps, initial_belief=initial_belief)
                return (alpha, experiment_results, action_traj)
            end
            
            experiment_results, action_traj = experiment(planner, pomdp, n_reps, max_steps)
            return (alpha, experiment_results, action_traj)
        end
        
        # Store results after parallel computation is complete
        for (alpha, res, actions) in results
            results_dict[policy_type][alpha] = res
            action_logs_dict[policy_type][alpha] = actions
        end
    end

    # Generate histograms and plots
    println("\nGenerating visualizations...")
    
    # Create histograms
    for (policy_name, alphas) in action_logs_dict
        for alpha in keys(alphas)
            plot_action_histograms(action_logs_dict[policy_name][alpha], alpha, policy_name)
        end
    end
    
    # Create Pareto curves
    p1 = plot_pareto_curve(results_dict, "Emission Cost", "Emission Cost", "p_npv_emission_alpha_pareto.png")
    p2 = plot_pareto_curve(results_dict, "CO2 Emitted", "CO₂ Emissions", "p_npv_co2_emitted_alpha_pareto.png")
    
    return results_dict, p1, p2
end

# Run the main function
results_dict, p1, p2 = main()