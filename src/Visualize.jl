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

distance(x1, y1, x2, y2) = sqrt((x1 - x2)^2 + (y1 - y2)^2)

function Visualizer(network::Network; max_fps::Number=10, behaviour=nothing)
    neuron_xs = Observable(Vector{Float64}(undef, network.neuron_count), ignore_equal_values=true)
    neuron_ys = Observable(Vector{Float64}(undef, network.neuron_count), ignore_equal_values=true)
    throttle(1 / max_fps, neuron_xs)
    throttle(1 / max_fps, neuron_ys)
    edge_pairs = collect(keys(network.graph.edge_data))

    fig = Figure()
    ax = Axis(fig[1, 1], aspect=DataAspect(), leftspinevisible=false, rightspinevisible=false, topspinevisible=false, bottomspinevisible=false)
    observers = Dict()
    vis = Visualizer(neuron_xs, neuron_ys, edge_pairs, observers, fig, ax)

    update_positions!(vis, network)

    for (n1, n2) in edge_pairs
        vis.observers[n1, n2, :xs] = Observable([neuron_xs.val[n1], neuron_xs.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :xs])
        vis.observers[n1, n2, :ys] = Observable([neuron_ys.val[n1], neuron_ys.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :ys])
        l = lines!(vis.observers[n1, n2, :xs], vis.observers[n1, n2, :ys], color=:grey)
    end

    scatter!(neuron_xs, neuron_ys, marker=:circle, color=:black)

    if behaviour !== nothing
        for (n, coords) in behaviour.goals
            start_pos = network.start_positions[n]
            goal_x, goal_y = goal_pos = start_pos + coords
            scatter!(goal_x, goal_y, marker=:circle, color=:transparent, strokewidth=1, strokecolor=:red)
            
            x, y = pos = [neuron_xs.val[n], neuron_ys.val[n]]
            v = goal_pos - pos
            vx, vy = v - normalize(v) * 0.15

            dist = distance(x, y, goal_pos[1], goal_pos[2])
            if dist > 0.1
                # shorten the arrow by 0.15 units so the tip is in middle of the circle
                arrows!([x], [y], [vx], [vy], color=:red)
            end
        end

        for (n, coords) in behaviour.modifiers
            coords *= 10
            start_coords = [neuron_xs.val[n], neuron_ys.val[n]]
            # shorten the arrow by 0.15 units so the tip is in middle of the circle
            start_coords -= normalize(coords) * 0.15
            x, y = start_coords
            change_x, change_y = coords
            dist = distance(x, y, x + change_x, y + change_y)
            if dist > 0.1
                arrows!([x], [y], [change_x], [change_y], color=:blue, align = :tailend)
            end
        end
    end
    hidedecorations!(ax)
    # colsize!(fig.layout, 1, 100)
    # rowsize!(fig.layout, 1, 250)
    resize_to_layout!(fig)
    display(fig)
    return vis
end