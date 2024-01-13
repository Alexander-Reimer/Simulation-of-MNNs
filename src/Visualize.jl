mutable struct Visualizer
    neuron_xs::Observables.Observable{Vector{Float64}}
    neuron_ys::Observables.Observable{Vector{Float64}}
    edge_pairs::Vector{Tuple}
    observers::Dict{Any,Observables.Observable}
    fig
    ax
end

function update_positions!(vis::Visualizer, network::Network)
    for n in 1:network.neuron_count
        vis.neuron_xs.val[n] = network.positions[1, n]
        vis.neuron_ys.val[n] = network.positions[2, n]
    end
    notify(vis.neuron_xs)
    notify(vis.neuron_ys)
    for (n1, n2, ax) in keys(vis.observers)
        if ax == :xs
            vis.observers[n1, n2, :xs].val[1] = vis.neuron_xs.val[n1]
            vis.observers[n1, n2, :xs].val[2] = vis.neuron_xs.val[n2]
        else
            vis.observers[n1, n2, :ys].val[1] = vis.neuron_ys.val[n1]
            vis.observers[n1, n2, :ys].val[2] = vis.neuron_ys.val[n2]
        end
        notify(vis.observers[n1, n2, ax])
    end
end

function Visualizer(network::Network;
    max_fps::Number=10,
    behaviour::Union{Nothing,Behaviour}=nothing)

    @info "test"
    neuron_xs = Observable(Vector{Float64}(undef, network.neuron_count), ignore_equal_values=true)
    neuron_ys = Observable(Vector{Float64}(undef, network.neuron_count), ignore_equal_values=true)
    throttle(1 / max_fps, neuron_xs)
    throttle(1 / max_fps, neuron_ys)
    edge_pairs = collect(keys(network.graph.edge_data))

    fig = Figure()
    ax = Axis(fig[1, 1])
    observers = Dict()
    vis = Visualizer(neuron_xs, neuron_ys, edge_pairs, observers, fig, ax)

    update_positions!(vis, network)

    for (n1, n2) in edge_pairs
        vis.observers[n1, n2, :xs] = Observable([neuron_xs.val[n1], neuron_xs.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :xs])
        vis.observers[n1, n2, :ys] = Observable([neuron_ys.val[n1], neuron_ys.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :ys])
        l = lines!(vis.observers[n1, n2, :xs], vis.observers[n1, n2, :ys], color=:grey, stroke=10)
    end

    scatter!(neuron_xs, neuron_ys, marker=:circle, markersize=25, color=:blue)

    if behaviour !== nothing
        for (n, coords) in behaviour.goals
            start_x, start_y = network.start_positions[n]
            scatter!(start_x + coords[1], start_y + coords[2], marker=:circle, markersize=25, color=:red)
            dist = sqrt((neuron_xs.val[n] - (start_x + coords[1]))^2 + (neuron_ys.val[n] - (start_y + coords[2]))^2)
            if dist > 0.1
                arrow_head_x, arrow_head_y = coords
                arrow_head_x -= arrow_head_x > 0 ? 0.01 : -0.01
                arrow_head_y -= arrow_head_y > 0 ? 0.01 : -0.01
                arrows!([start_x], [start_y], [arrow_head_x], [arrow_head_y], color=:green, linewidth=1, arrowsize=10, align = :origin)
            end
        end
    end
    display(fig)
    return vis
end

function Visualizer(network::Network, trainer::Trainer; max_fps::Number=10)
    Visualizer(network; max_fps=max_fps, behaviours=trainer.behaviours)
end