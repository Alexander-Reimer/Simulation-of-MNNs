mutable struct Visualizer
    neuron_xs::Observables.Observable{Vector{Float64}}
    neuron_ys::Observables.Observable{Vector{Float64}}
    edge_pairs::Vector{Tuple}
    observers::Dict{Any,Observables.Observable}
    fig::Any
    ax::Any
end

function update_positions!(vis::Visualizer, network::Network)
    for n in 1:(network.neuron_count)
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
    neuron_xs = Observable(
        Vector{Float64}(undef, network.neuron_count); ignore_equal_values=true
    )
    neuron_ys = Observable(
        Vector{Float64}(undef, network.neuron_count); ignore_equal_values=true
    )
    throttle(1 / max_fps, neuron_xs)
    throttle(1 / max_fps, neuron_ys)
    edge_pairs = collect(keys(network.graph.edge_data))

    fig = Figure()
    ax = Axis(
        fig[1, 1];
        aspect=DataAspect(),
        # limits = (-1, nothing, -1, nothing),
        xautolimitmargin=(0.5, 0.5),
        yautolimitmargin=(0.5, 0.5),
        leftspinevisible=false,
        rightspinevisible=false,
        topspinevisible=false,
        bottomspinevisible=false,
    )

    observers = Dict()
    vis = Visualizer(neuron_xs, neuron_ys, edge_pairs, observers, fig, ax)

    update_positions!(vis, network)

    for (n1, n2) in edge_pairs
        vis.observers[n1, n2, :xs] = Observable([neuron_xs.val[n1], neuron_xs.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :xs])
        vis.observers[n1, n2, :ys] = Observable([neuron_ys.val[n1], neuron_ys.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :ys])
        l = lines!(vis.observers[n1, n2, :xs], vis.observers[n1, n2, :ys]; color=:grey)
    end

    scatter!(neuron_xs, neuron_ys; marker=:circle, color=:black)

    if behaviour !== nothing
        for (n, coords) in behaviour.goals
            start_pos = network.start_positions[n]
            goal_x, goal_y = goal_pos = start_pos + coords
            scatter!(
                goal_x,
                goal_y;
                marker=:circle,
                color=:transparent,
                strokewidth=1,
                strokecolor=:red,
            )

            pos = @lift([$neuron_xs[n], $neuron_ys[n]])
            x = @lift([$pos[1]])
            y = @lift([$pos[2]])
            v = lift(pos) do pos
                w = goal_pos - pos
                # shorten the arrow by 0.15 units so the tip is in middle of the circle
                return w - normalize(w) * 0.15
            end
            vx = @lift([$v[1]])
            vy = @lift([$v[2]])

            arrows!(x, y, vx, vy; color=:red)
        end

        for (n, coords) in behaviour.modifiers
            coords *= 20
            # shorten the arrow by 0.15 units so the tip is in middle of the circle
            start_coords = @lift([$neuron_xs[n], $neuron_ys[n]] - normalize(coords) * 0.15)
            x = @lift([$start_coords[1]])
            y = @lift([$start_coords[2]])
            change_x, change_y = coords

            arrows!(x, y, [change_x], [change_y]; color=:blue, align=:tailend)
        end
    end
    hidedecorations!(ax)
    resize_to_layout!(fig)
    display(fig)
    return vis
end

function draw!(r)
    fig = GLMakie.Figure()
    ax = Axis(fig[1, 1])
    global ox = Observable(zeros(Float64, Int(size(r)[1] / 2)))
    global oy = Observable(zeros(Float64, Int(size(r)[1] / 2)))
    GLMakie.scatter!(ox, oy; marker=:circle, markersize=25, color=:blue)
    display(fig)
    sleep(1)
    @info "Started!"

    x = zeros(Float64, Int(size(r)[1] / 2))
    y = zeros(Float64, Int(size(r)[1] / 2))
    for i in 1:(size(r)[2])
        for j in 1:size(r)[1]
            if j % 2 == 0
                y[Int(j / 2)] = r[j, i]
            else
                x[Int(j / 2 + 0.5)] = r[j, i]
            end
        end
        ox[] = x
        oy[] = y

        sleep(0.01)
    end
end
