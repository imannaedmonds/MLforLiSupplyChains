using Plots

# Data from results
conditions = [
    "P: det, E: sto",
    "P: sto, E: det",
    "P: sto, E: sto",
    "P: det, E: det"
]

mean_rewards_rand = [1690.8, 2139.05, 2259.54, 2218.85]
se_rewards_rand = [253.71, 234.63, 225.43, 74.6]

mean_rewards_pomcpow = [2161.26, 2184.6, 2634.68, 2439.15]
se_rewards_pomcpow = [182.43, 182.88, 54.86, 154.05]

# Define colors
light_pink = RGBA(1.0, 0.5, 0.5, 0.99)  # Light pink
sky_blue = RGBA(0/255, 120/255, 255/255, 1.0)  # Sky blue

# Define bar positions
x_positions = 1:length(conditions)
bar_width = 0.4  # Reduce width to separate bars

# Bar plot for Mean Rewards
p1 = plot(size=(900, 500), xlabel="Conditions", ylabel="Mean Reward",
          title="Mean Rewards Comparison", legend=:topright, grid=false)

# Plot bars for Random Planner (light pink)
bar!(x_positions .- bar_width/2, mean_rewards_rand, yerr=se_rewards_rand,
     bar_width=bar_width, label="Random Planner", color=light_pink, linecolor=:black, linewidth=1)

# Plot bars for POMCPOW Planner (sky blue)
bar!(x_positions .+ bar_width/2, mean_rewards_pomcpow, yerr=se_rewards_pomcpow,
     bar_width=bar_width, label="POMCPOW Planner", color=sky_blue, linecolor=:black, linewidth=1)

# Update x-ticks
xticks!(x_positions, conditions)
xrotation = 20  # Rotate labels for readability

# Line plot with error bars
p2 = plot(
    x_positions, 
    mean_rewards_rand, 
    ribbon=se_rewards_rand, 
    xlabel="Conditions",
    ylabel="Mean Reward",
    title="Random vs. POMCPOW Planner Performance",
    label="Random Planner",
    linewidth=3,
    color=light_pink,
    marker=:circle,
    xticks=(x_positions, conditions),
    xrotation=20
)

plot!(
    x_positions, 
    mean_rewards_pomcpow, 
    ribbon=se_rewards_pomcpow, 
    label="POMCPOW Planner",
    linewidth=3,
    color=sky_blue,
    marker=:diamond
)

# Show plots
display(p1)
display(p2)

# Save plots
savefig(p1, "mean_rewards_comparison.png")
savefig(p2, "planner_performance_comparison.png")
