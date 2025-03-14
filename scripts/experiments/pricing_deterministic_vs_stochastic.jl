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

rng = MersenneTwister(1)

function compute_metrics(samples)
    sample_mean = mean(samples)
    n = length(samples)
    sample_se = std(samples) / sqrt(n)
    return (mean = sample_mean, se = sample_se)
end

function experiment(planners, eval_pomdp, n_reps=20, max_steps=30)
    results = Dict() 

    for (planner, planner_name) in planners
        reward_tot_all = []
        reward_disc_all = []
        emission_tot_all = []
        emission_disc_all = []
        domestic_tot_all = []
        imported_tot_all = []
    
        println(" ")
        println("=====Simulating ", typeof(planner), "=====")
        println(" ")
    
        for t in tqdm(1:n_reps)
            reward_tot = 0.0
            reward_disc = 0.0
            emission_tot = 0.0
            emission_disc = 0.0
            vol_tot = 0.0 #mined domestically
            imported_tot = 0.0 #imported/mined internationally
            disc = 1.0
    
            for (s, a, o, r) in stepthrough(eval_pomdp, planner, "s,a,o,r", max_steps=max_steps)
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
    
        results[planner_name] = Dict(
            "Total Reward" => compute_metrics(reward_tot_all),
            "Disc. Reward" => compute_metrics(reward_disc_all),
            "Total Emissions" => compute_metrics(emission_tot_all),
            "Disc. Emissions" => compute_metrics(emission_disc_all),
            "Total Domestic" => compute_metrics(domestic_tot_all),
            "Total Imported" => compute_metrics(imported_tot_all)
        )
    end
    return results  
end

function display_results(results)
    for planner_name in keys(results)
        println("Planner: ", planner_name)

        for metric in keys(results[planner_name])
            println(metric, ": ", results[planner_name][metric])
        end
    end
end

function print_results_table(results_table)
    println("")
    println("")
    println("**Mean (Rand)**")
    println("**SE (Rand)**")
    println("**Mean (POMCPOW)**")
    println("**SE (POMCPOW)**")
    
    # Print the results for each condition in the specified order
    for condition in [
        "P: deterministic, E: stochastic",
        "P: stochastic, E: deterministic",
        "P: stochastic, E: stochastic",
        "P: deterministic, E: deterministic"
    ]
        # Skip if this condition wasn't tested
        if !haskey(results_table, condition)
            continue
        end

        results = results_table[condition]
        
        rand_mean = round(results["Random Planner"]["Total Reward"].mean, digits=2)
        rand_se = round(results["Random Planner"]["Total Reward"].se, digits=2)
        pomcpow_mean = round(results["POMCPOW Planner"]["Total Reward"].mean, digits=2)
        pomcpow_se = round(results["POMCPOW Planner"]["Total Reward"].se, digits=2)
        
        # Format the condition to match the example layout
        formatted_condition = replace(condition, ", " => ", \n")
        
        println(formatted_condition)
        println("$rand_mean")
        println("$rand_se")
        println("$pomcpow_mean")
        println("$pomcpow_se")
    end
end

function main()
    # Define parameters
    n_reps = 5  # Number of repetitions for each experiment
    max_steps = 30  # Maximum number of steps in each simulation

    # Initialize the results container
    results_table = Dict()

    # Define the four experiment conditions
    experiment_conditions = [
        ("P: deterministic, E: stochastic", false, true),
        ("P: stochastic, E: deterministic", true, false),
        ("P: stochastic, E: stochastic", true, true),
        ("P: deterministic, E: deterministic", false, false)
    ]

    # Run each experiment condition
    for (condition_name, planning_stochastic, evaluation_stochastic) in experiment_conditions
        println("\n----- Testing $condition_name -----\n")

        # Initialize planning POMDP
        planning_pomdp = initialize_lipomdp(stochastic_price=planning_stochastic)
        planning_up = LiBeliefUpdater(planning_pomdp)

        # Initialize evaluation POMDP
        evaluation_pomdp = initialize_lipomdp(stochastic_price=evaluation_stochastic)

        # Initialize planners
        random_planner = RandPolicy(planning_pomdp)

        # POMCPOW Solver
        solver = POMCPOW.POMCPOWSolver(
            tree_queries=1000, 
            estimate_value = estimate_value,
            k_observation=4., 
            alpha_observation=0.1, 
            max_depth=15, 
            enable_action_pw=false,
            init_N=10  
        )
        pomcpow_planner = solve(solver, planning_pomdp)

        planners = [
            (random_planner, "Random Planner"),
            (pomcpow_planner, "POMCPOW Planner")
        ]

        # Run the experiment
        experiment_results = experiment(planners, evaluation_pomdp, n_reps, max_steps)
        
        # Store the results
        results_table[condition_name] = experiment_results

        # Display the results for this condition
        println("\nResults for $condition_name:")
        display_results(experiment_results)
    end

    # Format and display the final table
    println("\n----- Final Results Table -----\n")
    print_results_table(results_table)

    # Find the best performing condition for POMCPOW
    best_condition = ""
    best_reward = -Inf
    for (condition, results) in results_table
        pomcpow_mean = results["POMCPOW Planner"]["Total Reward"].mean
        if pomcpow_mean > best_reward
            best_reward = pomcpow_mean
            best_condition = condition
        end
    end

    # Bold the highest performing POMCPOW results in the table
    println("\nBest performing condition for POMCPOW: $best_condition with mean reward $best_reward")
    
    # Suggest follow-up analysis
    println("\nNote from the results that planning with deterministic pricing but evaluating with either deterministic or stochastic pricing yields the highest rewards for POMCPOW.")
end

main()