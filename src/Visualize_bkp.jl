
mutable struct Visualizer
    neuron_xs::Observables.Observable{Vector{Float64}}
    neuron_ys::Observables.Observable{Vector{Float64}}
    edge_pairs::Vector{Tuple}
    observers::Dict{Any,Observables.Observable}
    fig
    ax
end

function draw(network::Network)
    fig = Figure()
    ax = Axis(fig[1, 1])

    # edges first so they dont cover neurons
    # edges
    global edge_draw_obs = []
    i = 1
    for (n1, n2) in keys(network.graph.edge_data)
        xs = Observable(
            [get_neuron(network, n1).pos[1], get_neuron(network, n2).pos[1]],
        # ignore_equal_values=true
        )
        push!(edge_draw_obs, xs)
        ys = Observable(
            [get_neuron(network, n1).pos[2], get_neuron(network, n2).pos[2]],
        # ignore_equal_values=true
        )
        push!(edge_draw_obs, ys)
        lines!(edge_draw_obs[i], edge_draw_obs[i+1], color=:red, stroke=10)
        i += 2
    end

    # neurons
    xs = [get_neuron(network, i).pos[1] for i in 1:sum(network.row_counts)]
    ys = [get_neuron(network, i).pos[2] for i in 1:sum(network.row_counts)]
    scatter!(xs, ys, marker=:circle, markersize=25, color=:blue)
end


function draw!(r)
    fig = GLMakie.Figure()
    ax = Axis(fig[1, 1])
    global ox = Observable(zeros(Float64, Int(size(r)[1] / 2)))
    global oy = Observable(zeros(Float64, Int(size(r)[1] / 2)))
    GLMakie.scatter!(ox, oy, marker=:circle, markersize=25, color=:blue)
    display(fig)
    sleep(1)
    @info "Started!"

    x = zeros(Float64, Int(size(r)[1] / 2))
    y = zeros(Float64, Int(size(r)[1] / 2))
    for i = 1:(size(r)[2])
        for j = 1:size(r)[1]
            if j % 2 == 0
                y[Int(j / 2)] = r[j, i]
            else
                x[Int(j / 2 + 0.5)] = r[j, i]
            end

        end
        ox[] = x
        oy[] = y

        sleep(0.08)
    end

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

function Visualizer(network::Network; max_fps::Number=10)
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
    scatter!(neuron_xs, neuron_ys, marker=:circle, markersize=25, color=:blue)

    for (n1, n2) in edge_pairs
        vis.observers[n1, n2, :xs] = Observable([neuron_xs.val[n1], neuron_xs.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :xs])
        vis.observers[n1, n2, :ys] = Observable([neuron_ys.val[n1], neuron_ys.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :ys])
        l = lines!(vis.observers[n1, n2, :xs], vis.observers[n1, n2, :ys], color=:red, stroke=10)
    end

    display(fig)
    return vis
end

function show(net::Network, modifier::Function=NoModifier())
    reset!(net)
    r = simulate!(net, (0.0, 100.0), modifier=modifier)
    draw!(r)
end

