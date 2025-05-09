#=
Original Authors: Yasmine Alonso, Mansur Arief, Anthony Corso, Jef Caers, and Mykel Kochenderfer
Extended by: Anna Edmonds and Amelia Hardy
----------------
=#


POMDPs.support(b::LiBelief) = rand(b)  #TODO: change to be the support of deposit distributions

function POMDPs.initialstate(P::LiPOMDP)
    return Deterministic(P.init_state)
end

function POMDPs.states(P::LiPOMDP)
    deposits = collect(P.V_deposit_min:P.Δdeposit:P.V_deposit_max)
    V_tot_bounds = collect(P.V_deposit_min * P.n_deposits:P.ΔV:P.V_deposit_max * P.n_deposits)
    I_tot_bounds = collect(P.V_deposit_min * P.n_deposits:P.ΔV:P.V_deposit_max * P.n_deposits)
    booleans = [true, false]
    time_bounds = collect(1:P.time_horizon)

    # Create a list of deposit variables and boolean variables based on the number of deposits
    deposits_combinations = Iterators.product(fill(deposits, P.n_deposits)...)
    boolean_combinations = Iterators.product(fill(booleans, P.n_deposits)...)
    
    # Combine all iterators into one
    all_combinations = Iterators.product(deposits_combinations, V_tot_bounds, I_tot_bounds, time_bounds, boolean_combinations)

    # Create the states from the combinations
    𝒮 = [State(collect(s[1]), s[2], s[3], s[4], collect(s[5])) for s in all_combinations] 
    push!(𝒮, P.null_state)

    return 𝒮
end


# Removed the dependency on the belief
#=function POMDPs.actions(P::LiPOMDP)
    potential_actions = [MINE1, MINE2, MINE3, MINE4, EXPLORE1, EXPLORE2, EXPLORE3, EXPLORE4]
    return potential_actions
end=#

#Generate all Possible actions based of n_deposits
function POMDPs.actions(P::LiPOMDP)
    potential_mine_actions = []
    potential_explore_actions = []
    for i in 1:P.n_deposits
        push!(potential_mine_actions, "MINE$i")
        push!(potential_explore_actions, "EXPLORE$i")
    end

    for i in 1:length(potential_mine_actions)
        potential_mine_actions[i] = Action(potential_mine_actions[i])
        potential_explore_actions[i] = Action(potential_explore_actions[i])
    end

    return vcat(potential_mine_actions, potential_explore_actions)
    
end
# ====================================================
# INDIVIDUAL REWARD COMPONENTS
# ====================================================

# r1: Penalty for mining domestically before the time goal
function compute_r1(P::LiPOMDP, s::State, a::Action; domestic_mining_penalty=-1000)
    action_type = get_action_type(a)
    site_num = get_site_number(a)
    domestic_mining_actions = ["MINE1", "MINE2"]

    if action_type == "MINE" && !s.have_mined[site_num]
        penalty = domestic_mining_penalty
    else
        penalty = 0
    end
        
    return s.t <= P.t_goal && a.a in domestic_mining_actions ? penalty : 0
end

# r2: Maximize lithium volume mined
function compute_r2(P::LiPOMDP, s::State, a::Action)
    return (domestic=s.Vₜ, imported=s.Iₜ) 
end

#=
# r3: Minimize CO2 emissions
function compute_r3(P::LiPOMDP, s::State, a::Action)
    action_type = get_action_type(a) 
    site_num = get_site_number(a)

    if action_type == "MINE" && !s.have_mined[site_num]
        new_emission = -P.CO2_emissions  # Negative because emissions are costs
    else
        new_emission = 0
    end
    
    existing_emissions = 0
    for i in 1:P.n_deposits
        if s.have_mined[i]
            existing_emissions -= P.CO2_emissions
        end
    end

    return new_emission + existing_emissions
end
=#

# r4: Satisfy demand at every timestep
function compute_r4(P::LiPOMDP, s::State, a::Action; demand_unfulfilled_penalty=-150)
    production = sum(s.have_mined) * P.mine_output
    return P.demands[Int(s.t)] > production ? demand_unfulfilled_penalty : 0
end

# Calculate cashflow for mines
function compute_cashflow(P::LiPOMDP, s::State)
    reward = 0
    for i in 1:P.n_deposits
        if s.have_mined[i]
            if P.stochastic_price
                price = rand(P.site_to_dist[i])
            else
                price = mean(P.site_to_dist[i])
            end

            reward += P.mine_output * price
        end
    end
    return reward
end

# r5: Maximize profit (revenue - costs)
function compute_r5(P::LiPOMDP, s::State, a::Action; capex_per_mine=-500, opex_per_mine=-25, price=100)
    reward = 0

    # Revenue from operating mines
    reward += compute_cashflow(P, s)

    # Handle capital expenditure for new mine
    action_type = get_action_type(a)
    site_num = get_site_number(a)

    if action_type == "MINE" && !s.have_mined[site_num]
        reward += capex_per_mine
    end

    # Operating costs for all active mines
    reward += opex_per_mine * sum(s.have_mined)

    return reward
end

# ====================================================
# COMBINED REWARD COMPONENTS FOR TRADEOFF ANALYSIS
# ====================================================

# Calculate emission cost by combining r3 with monetary CO2 costs
function compute_emission_cost(P::LiPOMDP, s::State, a::Action)
    
    # Add monetary CO2 costs based on CO2_cost parameter
    action_type = get_action_type(a)
    site_num = get_site_number(a)
    
    # Calculate monetary cost for new mine
    new_monetary_cost = 0

    if action_type == "MINE" && !s.have_mined[site_num]
        new_monetary_cost -= P.CO2_emissions * P.mine_output * P.CO2_cost[site_num]
    end
    
    # Add monetary cost for existing mines
    old_monetary_cost = 0
    for i in 1:P.n_deposits
        if s.have_mined[i]
            old_monetary_cost -= P.CO2_emissions * P.mine_output * P.CO2_cost[i]
        end
    end
    
    # Return combined environmental impact and monetary cost
    return old_monetary_cost + new_monetary_cost
end

# Calculate NPV by combining relevant economic components
function compute_npv(P::LiPOMDP, s::State, a::Action)
    # Start with base profit (r5)
    npv = compute_r5(P, s, a)
    
    # Add domestic mining timing penalty (r1)
    npv += compute_r1(P, s, a)
    
    # Add demand satisfaction penalty (r4)
    npv += compute_r4(P, s, a)
    
    return npv
end

# Calculate volume reward (production component)
function compute_volume_reward(P::LiPOMDP, s::State, a::Action)
    volumes = compute_r2(P, s, a)
    return volumes.domestic + volumes.imported
end

# Combined NPV-emission tradeoff function
function compute_npv_emission_tradeoff(P::LiPOMDP, s::State, a::Action)
    # Calculate NPV component
    npv = compute_npv(P, s, a)
    
    # Calculate emission cost component
    emission_cost = compute_emission_cost(P, s, a)
    
    # Calculate volume component (r2)
    volume = compute_volume_reward(P, s, a)
    
    # Apply alpha weighting
    npv_term = P.alpha * npv
    emission_term = (1 - P.alpha) * emission_cost
    
    # Always include a small weight for volume to ensure production is valued
    volume_term = 0.1 * volume
    
    return npv_term + emission_term + volume_term
end

# ====================================================
# MAIN REWARD FUNCTION
# ====================================================

# Main reward function for the LiPOMDP
function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    if isterminal(P, s)
        return 0
    end

    if P.compute_tradeoff
        # Use NPV-emission tradeoff mode
        return compute_npv_emission_tradeoff(P, s, a)
    else
        # Use original reward calculation with individual components
        demand_unfulfilled_penalty = -150
        domestic_mining_penalty = -1000
        capex_per_mine = -25
        opex_per_mine = -10
        material_price = 100

        # Individual reward components
        r1 = compute_r1(P, s, a, domestic_mining_penalty=domestic_mining_penalty)
        r2 = sum(compute_r2(P, s, a))
        r3 = compute_emission_cost(P, s, a)
        r4 = compute_r4(P, s, a, demand_unfulfilled_penalty=demand_unfulfilled_penalty)
        r5 = compute_r5(P, s, a, capex_per_mine=capex_per_mine, opex_per_mine=opex_per_mine, price=material_price)
        
        return dot([r1, r2, r3, r4, r5], P.obj_weights)
    end
end


function POMDPs.transition(P::LiPOMDP, s::State, a::Action)

    if s.t >= P.time_horizon || isterminal(P, s)  
        sp = deepcopy(P.null_state)
        return Deterministic(sp)
    else
        sp = deepcopy(s) # Make a copy!!! need to be wary of this in Julia deepcopy might be slow
        sp.t = s.t + 1  # Increase time by 1 in all cases
        ΔV = 0.0
        ΔI = 0.0
        new_mine_output = 0.0

        #process new mine
        action_type = get_action_type(a)
        site_number = get_site_number(a)
        
        if action_type == "MINE"             
            if s.deposits[site_number] >= P.mine_output
                new_mine_output = P.mine_output
            else
                new_mine_output = s.deposits[site_number]
            end
            sp.deposits[site_number] = s.deposits[site_number] - new_mine_output            
            sp.have_mined[site_number] = true

            if get_site_number(a) in [1, 2]
                ΔV = new_mine_output
            else
                ΔI = new_mine_output
            end
        end

        # process the existing mines
        for i in 1:P.n_deposits #TODO: make this more flexible
            if s.have_mined[i]
                if s.deposits[i] >= P.mine_output
                    mine_output = P.mine_output
                else
                    mine_output = s.deposits[i]
                end
                sp.deposits[i] = s.deposits[i] - mine_output
                if i in [1, 2]                    
                    ΔV += mine_output
                else
                    ΔI += mine_output
                end
            end
        end

        sp.Vₜ = s.Vₜ + ΔV
        sp.Iₜ = s.Iₜ + ΔI

        return Deterministic(sp)
    end
   
end

function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)

    sp =rand(rng, transition(P, s, a))  
    o = rand(rng, observation(P, a, sp))  
    r = reward(P, s, a)
    out = (sp=sp, o=o, r=r)  

    return out
end


function POMDPs.observation(P::LiPOMDP, a::Action, sp::State)

    site_number = get_site_number(a)  #1, 2, 3, or 4, basically last character of    
    action_type = get_action_type(a)  # "EXPLORE" or "MINE"

    sentinel_dist = DiscreteNonParametric([-1.], [1.])
    temp::Vector{UnivariateDistribution} = fill(sentinel_dist, P.n_deposits)

    # handle degenerate case where we have no more Li at this site
    if sp.deposits[site_number] <= 0
        site_dist = sentinel_dist
        return product_distribution(temp)        
    end

    if action_type == "MINE"
        return product_distribution(temp) 
    else    
        site_dist = Normal(sp.deposits[site_number], P.σ_obs)
        quantile_vols = collect(0.:1.:60.) #TODO: put in the problem struct

        # Now get the chunk boundaries (Dashed lines in my drawings)
        chunk_boundaries = compute_chunk_boundaries(quantile_vols)

        # Now compute the probabilities of each chunk
        probs = compute_chunk_probs(chunk_boundaries, site_dist)

        temp[site_number] = DiscreteNonParametric(quantile_vols, probs)

        return product_distribution(temp)

    end
end


POMDPs.discount(P::LiPOMDP) = P.γ

POMDPs.isterminal(P::LiPOMDP, s::State) = s == P.null_state || s.t >= P.time_horizon

function POMDPs.initialize_belief(up::LiBeliefUpdater)
    deposit_dists = fill(Normal(), up.P.n_deposits)
    for i in 1:up.P.n_deposits
        #initalized belief is a normal distribution centered around the true value, standard deviation is defaulted to one because we only give it a mean 
       deposit_dists[i] = Normal(up.P.init_state.deposits[i]) 
    end 

    t = 1.0
    V_tot = 0.0
    I_tot = 0.0
    have_mined = [false for i in 1:up.P.n_deposits]
    
    return LiBelief(deposit_dists, t, V_tot, I_tot, have_mined)
end

function initialize_belief_import_only(up::LiBeliefUpdater)
    deposit_dists = fill(Normal(), up.P.n_deposits)
    initial_belief_mean = [16.0, 60.0, 80.0, 120.0] # Switched so it thinks site 3 has less than site 4
    
    for i in 1:up.P.n_deposits
        # Initialize belief as a normal distribution centered around the specified mean
        deposit_dists[i] = Normal(initial_belief_mean[i])
    end
    
    t = 1.0
    V_tot = 0.0
    I_tot = 0.0
    have_mined = [false for i in 1:up.P.n_deposits]
    
    return LiBelief(deposit_dists, t, V_tot, I_tot, have_mined)
end

POMDPs.initialize_belief(up::LiBeliefUpdater, dist) = POMDPs.initialize_belief(up)


function POMDPs.update(up::Updater, b::LiBelief, a::Action, o::Vector{Float64})
    
    action_type = get_action_type(a)
    site_number = get_site_number(a)
    P = up.P

    if action_type == "EXPLORE"
        bi = b.deposit_dists[site_number]  # This is a normal distribution
        μi = mean(bi)
        σi = std(bi)
        zi = o[site_number]
        μ_prime, σ_prime = kalman_step(P, μi, σi, zi)
        bi_prime = Normal(μ_prime, σ_prime)
        
        # Default, not including updated belief
        deposits = [(b.deposit_dists[i]) for i in 1:P.n_deposits]

        belief = LiBelief(deposits, b.t + 1, b.V_tot, b.I_tot, b.have_mined)

        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = bi_prime
        
        
        return belief

    # MINE actions: Shifts our mean of the distribution corresponding to the proper deposit down by 1 (since we
    # have just mined one unit deterministically). Does not affect certainty at all. 
    else # a must be a MINE action

        bi = b.deposit_dists[site_number]
        μi = mean(bi)
        σi = std(bi)
        
        if μi >= up.P.mine_output
            μi_prime = μi - up.P.mine_output
            n_units_mined = up.P.mine_output
        else 
            μi_prime = 0
            n_units_mined = μi  # the mine has been depleted
        end

        I_tot_prime = b.I_tot
        V_tot_prime = b.V_tot
        if site_number in [1, 2]
            I_tot_prime += n_units_mined
        else
            V_tot_prime += n_units_mined
        end
        
        n_deposits = P.n_deposits
        deposits = [(b.deposit_dists[i]) for i in 1:n_deposits]
        have_mined = [b.have_mined[i] for i in 1:n_deposits]

        belief = LiBelief(deposits, b.t + 1, b.V_tot, b.I_tot, b.have_mined)
        # Now, at the proper site number, update to contain the updated belief
        belief.deposit_dists[site_number] = Normal(μi_prime, σi)
        belief.have_mined[site_number] = true

        return belief
    end 
end
