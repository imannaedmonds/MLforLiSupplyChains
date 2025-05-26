using BSON
using DataFrames, StatsBase, Statistics
using Plots

const _DIR = @__DIR__
data = BSON.load(joinpath(_DIR, "..", "..", "action_logs_dict_new.bson"))
action_logs_dict = data[:action_logs_dict]


# 2) Build per‐time‐step action proportions
function action_freqs_per_timestep(action_trajectories::Vector{Vector{String}})
    max_steps = length(action_trajectories)
    all_actions = unique(vcat(action_trajectories...))
    df = DataFrame(time = 1:max_steps)
    for a in all_actions
        df[!, a] = zeros(Float64, max_steps)
    end
    for t in 1:max_steps
        cmap = countmap(action_trajectories[t])
        total = sum(values(cmap))
        for (a, c) in cmap
            df[t, a] = c/total
        end
    end
    return df
end


# the eight actions, in whatever order you like
const ACTIONS = [
  "EXPLORE1","EXPLORE2","EXPLORE3","EXPLORE4",
  "MINE1",   "MINE2",   "MINE3",   "MINE4"
]

const ACTION_COLORS = [
  :seagreen1,  # #9AFF9A
  :green3, # #00EE76
  :navyblue,     # #000080
  :deepskyblue1, # #00BFFF
  :orangered1,   # #FF4500
  :darkorange,   # #FF8C00
  :hotpink1,     # #FF6EB4
  :goldenrod1    # #FFC125
]

using CategoricalArrays  # for making a factor with fixed levels

function plot_stacked(df::DataFrame; title="")
  long = stack(df, Not(:time), variable_name=:action, value_name=:prop)

  # force the action factor to your fixed order
  long.action = categorical(long.action, levels=ACTIONS)

  bar(
    long.time,
    long.prop;
    group        = long.action,
    bar_position = :stack,
    palette      = ACTION_COLORS,
    xlabel       = "Time step",
    ylabel       = "Proportion",
    title        = title,
    legend       = :outerright,
    bar_width    = 0.8,
    lw           = 0,
    ylim         = (0,1)
  )
end



function flattened_corr(df1::DataFrame, df2::DataFrame)
    # get the union of action-column names (they’re Strings)
    acts = sort(collect(union(names(df1)[2:end], names(df2)[2:end])))

    v1 = Float64[]
    v2 = Float64[]

    for a in acts
        # if this action exists in df1, grab the column; otherwise fill zeros
        col1 = a in names(df1) ? df1[!, a] : zeros(nrow(df1))
        col2 = a in names(df2) ? df2[!, a] : zeros(nrow(df2))
        append!(v1, col1)
        append!(v2, col2)
    end

    return cor(v1, v2)
end

function main()
    policy = "POMCPOW"
    alphas = (0.0, 0.06, 0.08, 0.1, 0.2)
    logs = action_logs_dict[policy]
    dfs = Dict(α => action_freqs_per_timestep(logs[α]) for α in alphas)
    plots = [ plot_stacked(dfs[α], title="α = $(α)") for α in alphas ]

    # instead of display(), just save to disk

    p = plot(
        plots...;
        layout       = (5,1),
        size         = (800,1200),
        ylim         = (0,1),
        margin       = 5Plots.mm,       # 5 mm margin on all sides
        bottom_margin = 10Plots.mm,     # you can also fine‐tune each side
        left_margin   = 12Plots.mm,
        top_margin    = 5Plots.mm
      )
    
    savefig(joinpath(_DIR, "action_distribution_stacked_5.png"))

    println("corr(0.0 vs 0.06) = ", flattened_corr(dfs[0.0], dfs[0.06]))
    println("corr(0.0 vs 0.08) = ", flattened_corr(dfs[0.0], dfs[0.08]))
end

main()
