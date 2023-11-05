module MNN

using GLMakie
using Graphs
using LinearAlgebra
using MetaGraphsNext
using Observables

struct Neuron
    # https://jkrumbiegel.com/pages/2020-10-31-tuples-and-vectors/
    # TODO: cause of error?
    # pos::Tuple{Float64, Float64}
    pos::Array{Float64,1}
    velocity::Vector{Float64}
    movable::Bool
end

mutable struct Spring
    spring_constant::Float64
    const length::Float64
end
spring_init() = rand() - 0.5

mutable struct Visualizer
    neuron_xs::Observables.Observable{Vector{Float64}}
    neuron_ys::Observables.Observable{Vector{Float64}}
    edge_pairs::Vector{Tuple}
    observers::Dict{Any,Observables.Observable}
    fig
    ax
end

mutable struct Network
    graph::MetaGraphsNext.MetaGraph
    rows::Int # how many rows fixed columns have
    row_counts::Array{Int,1}
    col_fixed::Array{Bool,1}
    columns::Int
    neuron_count::Int
    xdist::Float64
    ydist::Float64
end

function Network(graph::MetaGraphsNext.MetaGraph, rows::Int, columns::Int, xdist::Float64=1.0)
    row_counts = Array{Int}(undef, columns)
    col_fixed = Array{Bool}(undef, columns)
    neuron_count = 0
    Network(graph, rows, row_counts, col_fixed, columns, neuron_count, xdist, calc_ydist(xdist))
end

function print_graph(network::Network)
    for col in 1:network.columns
        println("Column: $col")
        for row in 1:network.row_counts[col]
            i = get_neuron_index(network, col, row)
            n = get_neuron(network, i)
            println("\t Pos: $(n.pos), movable: $(n.movable)")
        end
    end
    # for n in network.graph.vertex_properties |> values
    #     println("Pos: $(n[2].pos), movable: $(n[2].movable)")
    # end
end


function new_graph()
    return MetaGraphsNext.MetaGraph(
        Graph();  # underlying graph structure
        label_type=Int,
        vertex_data_type=Neuron,
        edge_data_type=Spring,
        graph_data="Simulation"  # tag for the whole graph
    )
end

function get_neuron_index(network::Network, column::Int, row::Int)
    return sum(network.row_counts[1:column-1]) + row
end

# label: whether index is "true" (-> internal) index or "label" given by user at
# definition
function get_neuron(network::Network, index::Int; label::Bool=true)
    true_index = label ? network.graph.vertex_labels[index] : index
    # Tuple of [1] index and [2] neuron
    x = network.graph.vertex_properties[true_index]
    x[1] != index && error("Error correctly converting index")
    return x[2]
end

function get_neuron(network::Network, column::Int, row::Int)
    return get_neuron(network, get_neuron_index(network, column, row))
end

function set_neuron_positions!(network::Network, column::Int)
    xdist = network.xdist
    ydist = network.ydist
    fixed = network.col_fixed[column]
    rows = network.row_counts[column]
    for row in 1:rows
        xpos = (column - 1) * xdist
        ypos = (row - 1) * ydist
        if !fixed
            ypos += ydist / 2
        end

        i = get_neuron_index(network, column, row)
        neuron = get_neuron(network, i)
        neuron.pos[1] = xpos
        neuron.pos[2] = ypos
    end
end

function set_neuron_velocities!(network::Network, column::Int)
    for row in network.row_counts[column]
        get_neuron(network, column, row).velocity[1] = 0
        get_neuron(network, column, row).velocity[2] = 0
    end
end

function set_neuron_velocities!(network::Network)
    for n in 1:network.neuron_count
        get_neuron(network, n).velocity[1] = 0
        get_neuron(network, n).velocity[2] = 0
    end
end

function set_neuron_positions!(network::Network)
    for col in 1:network.columns
        set_neuron_positions!(network, col)
    end
end

function initialize_neurons!(network::Network)
    function add_column!(rows::Int, column::Int, fixed::Bool)

        function add_neuron!(movable::Bool, xpos::Number, ypos::Number)
            network.neuron_count += 1
            network.graph[network.neuron_count] = Neuron([xpos, ypos], [0, 0], movable)
        end

        network.row_counts[column] = rows
        network.col_fixed[column] = fixed

        for row in 1:rows
            movable_neuron = !(fixed && (row == 1 || row == rows))
            add_neuron!(movable_neuron, 0, 0)
        end
    end

    fixed = false
    for column in 1:network.columns
        rows = fixed ? network.rows : network.rows - 1
        add_column!(rows, column, fixed)
        fixed = !fixed
        set_neuron_positions!(network, column)
    end
end

function initialize_springs!(network::Network)
    function add_spring!(n1::Int, n2::Int)
        pos1 = get_neuron(network, n1).pos
        pos2 = get_neuron(network, n2).pos
        dx = pos1[1] - pos2[1]
        dy = pos1[2] - pos2[2]
        len = sqrt(dx^2 + dy^2)
        return add_edge!(network.graph, n1, n2, Spring(spring_init(), len))
    end
    function good_index(col, row)
        return 1 <= col <= network.columns && 1 <= row <= network.row_counts[col]
    end
    function add_spring!(col1, row1, col2, row2)
        if good_index(col1, row1) && good_index(col2, row2)
            add_spring!(
                get_neuron_index(network, col1, row1),
                get_neuron_index(network, col2, row2)
            )
        end
    end

    for col in 1:network.columns
        fixed = network.col_fixed[col]
        rows = network.row_counts[col]
        for row in 1:rows
            add_spring!(col, row, col + 1, row)
            if !(fixed && (row == 1 || row == rows))
                add_spring!(col, row, col + 2, row)
            end
            if fixed
                add_spring!(col, row, col + 1, row - 1)
            else
                add_spring!(col, row, col + 1, row + 1)
            end
        end
    end
end

function initialize_graph!(network::Network)
    initialize_neurons!(network)
    initialize_springs!(network)
    print_graph(network)

    # res = true
    # # TODO
    # rows = network.rows
    # cols = network.columns
    # for col in 2:cols
    #     for row1 in 1:rows
    #         for row2 in 1:rows
    #             i1 = (col-2)*rows + row1
    #             i2 = (col-1)*rows + row2
    #             res = res && add_edge!(network.graph, )
    #         end

    #         i_right_row = (row-1)*cols + col
    #         res = res && add_edge!(network.graph, i_right_row-cols, i_)
    #     end
    # end
    # for i = (2*rows + 2) : (2*rows + 1)*cols + rows
    #     res = res && add_edge!(network.graph, i, i-(rows*2 + 1), Spring(spring_init(), 1))
    # end
    # for i = rows + 1:(2*rows +1)*(cols)+rows
    #     if norm(network.graph[i].pos .- network.graph[i-rows].pos) < 1.1
    #         res = res && add_edge!(network.graph, i, i-rows, Spring(r_spring_constant(),1))
    #     end
    #     if (i-rows-1 > 0 && norm(network.graph[i].pos .- network.graph[i-rows-1].pos) < 1.1)
    #         res = res && add_edge!(network.graph, i, i-rows-1, Spring(r_spring_constant(),1))
    #     end
    # end
    # return res
end

"""
    calc_ydist(xdist)

Calculate necessary distance on y axis to create an equilateral triangle with
given distance between columns on x axis.
"""
calc_ydist(xdist) = 2 * sqrt(3) * xdist

function Network(columns, rows)
    graph = new_graph()
    network = Network(graph, rows, columns)
    initialize_graph!(network)
    # @info "Successful initialization?" res
    print_graph(network)
    return network
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

function get_spring(network, n1, n2; label::Bool=true)
    tn1 = label ? network.graph.vertex_labels[n1] : n1
    tn2 = label ? network.graph.vertex_labels[n2] : n2
    if !haskey(network.graph.edge_data, (tn1, tn2))
        return network.graph.edge_data[tn2, tn1]
    end
    return network.graph.edge_data[tn1, tn2]
end

function displacement2force(diff, spring_constant)
    if abs(diff) > 0.1
        return diff * 100
    else
        return diff * spring_constant
    end
end

function calculate_force(network::Network, n::Int)
    n = network.graph.vertex_labels[n]
    f = [0, 0]
    # neighbouring neurons
    nneurons = neighbors(network.graph, n)
    for nn in nneurons
        neuron = get_neuron(network, n, label=false)
        nneuron = get_neuron(network, nn, label=false)
        spring = get_spring(network, n, nn, label=false)
        diff = neuron.pos - nneuron.pos
        f += displacement2force(spring.length - norm(diff),
            spring.spring_constant) * diff / norm(diff)
        if isnan.(f) != [0, 0]
            throw("calculated force as NaN at neighboring neuron $nn with pos $(neuron.pos)")
        end
    end
    return f
end

function update_position!(network::Network, n::Int, delta::Number)
    neuron = get_neuron(network, n)
    if neuron.movable == false
        return
    end
    f = calculate_force(network, n) * delta
    # @info "f: $(round.(f, digits=2))"
    neuron.velocity .+= f
    neuron.velocity .*= 0.95
    # @info "v: $(round.(neuron.velocity, digits=3))"
    neuron.pos .+= neuron.velocity * delta
end

function simulate!(network::Network, delta::Number, epochs::Int;
    vis::Union{Visualizer,Nothing}=nothing, showfps::Bool=false)
    showfps ? start_time = time() : nothing
    for i = 1:epochs
        for row in 1:network.row_counts[1]
            get_neuron(network, 0, row).velocity[1] += 0.5 / (1 / delta)
        end
        for n in 1:network.neuron_count
            update_position!(network, n, delta)
        end
        if vis !== nothing
            update_positions!(vis, network)
            sleep(0.005)
        end
        # TODO: adjust delta automatically dependant on fps?
    end
    if showfps
        fps = epochs / (time() - start_time)
        @info "FPS: $fps"
    end
end

function loss(network::Network)
    col = network.columns
    rows = network.row_counts[col]
    x_positions = Array{Float64}(undef, rows)
    goal_positions = Array{Float64}(undef, rows)
    for row in 1:rows
        neuron = get_neuron(network, col, row)
        x_positions[row] = neuron.pos[1]
        goal_positions[row] = (col - 1) * network.xdist
    end
    goal_positions[rows÷2] += 1
    return sum((goal_positions .- x_positions) .^ 2) / rows
end

function reset!(network::Network)
    set_neuron_positions!(network)
    set_neuron_velocities!(network)
end

function update_positions!(vis::Visualizer, network::Network)
    for n in 1:network.neuron_count
        neuron = get_neuron(network, n)
        vis.neuron_xs.val[n] = neuron.pos[1]
        vis.neuron_ys.val[n] = neuron.pos[2]
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
    scatter!(neuron_xs, neuron_ys, marker=:circle, markersize=25, color=:blue)

    for (n1, n2) in edge_pairs
        vis.observers[n1, n2, :xs] = Observable([neuron_xs.val[n1], neuron_xs.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :xs])
        vis.observers[n1, n2, :ys] = Observable([neuron_ys.val[n1], neuron_ys.val[n2]])
        throttle(1 / max_fps, vis.observers[n1, n2, :ys])
        l = lines!(vis.observers[n1, n2, :xs], vis.observers[n1, n2, :ys], color=:red, stroke=10)
        # break
    end

    display(fig)
    return vis
end

function set_spring_data!(network::Network, spring_data::Dict)
    for x in spring_data
        key = x[1]
        val = x[2]
        network.graph.edge_data[key] = val
    end
end

function get_spring_constants(network::Network)
    # spring_constants = Array{Float64}(undef, length(network.graph.edge_data))
    return network.graph.edge_data
end

function mutate!(spring_data::Dict, strength=0.1)
    for x in spring_data
        spring = x[2]
        spring.spring_constant += strength * (rand() - 0.5)
    end
    return spring_data
end

function create_mutation(spring_data::Dict, strength=0.1)
    return mutate!(deepcopy(spring_data), strength)
end

function create_mutations!(spring_data::Dict{Tuple{Int64,Int64},MNN.Spring}, mutations::Vector{Dict{Tuple{Int64,Int64},MNN.Spring}}; strength=0.1)
    for i in eachindex(mutations)
        mutations[i] = create_mutation(spring_data, strength)
    end
end

function calc_loss(network::Network, spring_data::Dict{Tuple{Int64,Int64},MNN.Spring}; vis=nothing, delta=0.01, epochs=500)
    reset!(network)
    set_spring_data!(network, spring_data)
    simulate!(network, delta, epochs, vis=vis)
    return loss(network)
end

function calc_losses!(network, candidates, losses; vis=nothing, delta=0.01, epochs=250)
    for i in eachindex(losses)
        losses[i] = calc_loss(network, candidates[i], vis=vis, delta=delta, epochs=epochs)
    end
end

function calc_losses_parallel!(network, candidates, losses; vis=nothing, delta=0.01, epochs=250)
    candidate_is = [i for i in eachindex(candidates)]
    chunks = Iterators.partition(candidate_is, max(length(candidate_is) ÷ Threads.nthreads(), 1))
    tasks = map(chunks) do i
        if length(i) == 1
            i = i[1]
            Threads.@spawn calc_loss(deepcopy($network), $(candidates[i]))
        else
            for j = i[1]:i[2]
                Threads.@spawn calc_loss(deepcopy($network), $(candidates[j]))
            end
        end
    end
    results = fetch.(tasks)
    for i in eachindex(results)
        losses[i] = results[i]
    end
end

function train!(network::Network, epochs::Int; vis=nothing, mutations=20, parallel=true, mutation_strength=0.3)
    candidates = Array{Dict{Tuple{Int64,Int64},MNN.Spring}}(undef, mutations)
    [candidates[i] = Dict() for i in eachindex(candidates)]
    candidate_losses = Vector{Float64}(undef, mutations)
    best_candidate = copy(get_spring_constants(network))
    loss_function! = parallel ? calc_losses_parallel! : calc_losses!
    for i = 1:epochs
        create_mutations!(best_candidate, candidates, strength=mutation_strength)
        loss_function!(network, candidates, candidate_losses, vis=vis)
        best_candidate = deepcopy(candidates[argmin(candidate_losses)])
        @info "Epoch: $i, best loss: $(minimum(candidate_losses))"
    end
    # this call to calc_loss also saves best candidate in network
    @info "Final loss: $(calc_loss(network, best_candidate))"
end

end