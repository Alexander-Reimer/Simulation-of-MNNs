module MNN

export Network, simulate!, Trainer, train!, Visualizer, reset!, PPS, Evolution

using AngleBetweenVectors
using CairoMakie # visualization (pdf-capable backend)
using DifferentialEquations
using GLMakie # visualization (interactive backend)
using Graphs
using LinearAlgebra # norm, normalize
using MetaGraphsNext
using Observables # updating visualization
using Random # UUIDs & setting rand seed
using StaticArrays
# CairoMakie.activate!()

struct Neuron
    movable::Bool
end

mutable struct Spring
    spring_constant::Float64
    const length::Float64
end
spring_init() = rand() - 0.5

mutable struct Network
    graph::MetaGraphsNext.MetaGraph
    rows::Int # how many rows fixed columns have
    row_counts::Array{Int,1}
    col_fixed::Array{Bool,1}
    columns::Int
    neuron_count::Int
    xdist::Float64
    ydist::Float64
    start_positions::Dict{Int,Vector{Number}}
    positions::Array{Float64,2}
    velocities::Array{Float64,2}
end

include("Visualize.jl")

mutable struct Behaviour
    goals::Dict{Int,Vector{Number}}
    relative::Bool
    modifiers::Dict{Int,Vector{Number}}
end

abstract type Optimization end
abstract type Simulation end

mutable struct Trainer
    behaviours::Vector{Behaviour}
    simulation::Simulation
    optimization::Optimization
end

function Network(graph::MetaGraphsNext.MetaGraph, rows, columns, xdist=1.0)
    row_counts = Vector{Int}(undef, columns)
    col_fixed = Vector{Bool}(undef, columns)
    neuron_count = 0
    start_positions = Dict{Int,Vector{Number}}()
    positions = zeros(Float64, 0, 0)
    velocities = zeros(Float64, 0, 0)
    return Network(
        graph,
        rows,
        row_counts,
        col_fixed,
        columns,
        neuron_count,
        xdist,
        calc_ydist(xdist),
        start_positions,
        positions,
        velocities,
    )
end

function print_graph(network::Network)
    for col in 1:(network.columns)
        println("Column: $col")
        for row in 1:network.row_counts[col]
            i = get_neuron_index(network, col, row)
            n = get_neuron(network, i)
            println("\t Pos: $(network.positions[:,i]), movable: $(n.movable)")
        end
    end
end

function new_graph()
    return MetaGraphsNext.MetaGraph(
        Graph();  # underlying graph structure
        label_type=Int,
        vertex_data_type=Neuron,
        edge_data_type=Spring,
        graph_data="Simulation",  # tag for the whole graph
    )
end

function get_neuron_index(network::Network, column::Int, row::Int)
    return sum(network.row_counts[1:(column - 1)]) + row
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
        network.positions[1, i] = xpos
        network.positions[2, i] = ypos
        network.start_positions[i] = [xpos, ypos]
    end
end

function set_neuron_velocities!(network::Network, column::Int)
    for row in network.row_counts[column]
        n = get_neuron_index(network, column, row)
        network.velocities[:, n] = [0.0, 0.0] #sollte das nicht network.velocities sein????????????????????????????????????????????????????????????????????????????????????????????????????????
    end
end

function set_neuron_velocities!(network::Network)
    for n in 1:(network.neuron_count)
        network.velocities[:, n] = [0.0, 0.0]
    end
end

function set_neuron_positions!(network::Network)
    for col in 1:(network.columns)
        set_neuron_positions!(network, col)
    end
end

function initialize_neurons!(network::Network)
    function add_column!(rows::Int, column::Int, fixed::Bool)
        function add_neuron!(movable::Bool)
            network.neuron_count += 1
            return network.graph[network.neuron_count] = Neuron(movable)
        end

        network.row_counts[column] = rows
        network.col_fixed[column] = fixed

        for row in 1:rows
            movable_neuron = !(fixed && (row == 1 || row == rows))
            add_neuron!(movable_neuron)
        end
    end

    fixed = false
    for column in 1:(network.columns)
        rows = fixed ? network.rows : network.rows - 1
        add_column!(rows, column, fixed)
        fixed = !fixed
    end
    network.positions = Array{Number,2}(undef, 2, network.neuron_count)
    network.velocities = Array{Number,2}(undef, 2, network.neuron_count)
    set_neuron_positions!(network)
    return set_neuron_velocities!(network)
end

function initialize_springs!(network::Network)
    function add_spring!(n1::Int, n2::Int)
        pos1 = network.positions[:, n1]
        pos2 = network.positions[:, n2]
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
                get_neuron_index(network, col1, row1), get_neuron_index(network, col2, row2)
            )
        end
    end

    for col in 1:(network.columns)
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
    return initialize_springs!(network)
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
    return network
end

function get_spring(network, n1, n2; label::Bool=true)
    tn1 = label ? network.graph.vertex_labels[n1] : n1
    tn2 = label ? network.graph.vertex_labels[n2] : n2
    if !haskey(network.graph.edge_data, (tn1, tn2))
        return network.graph.edge_data[tn2, tn1]
    end
    return network.graph.edge_data[tn1, tn2]
end

function addvelocity!(network::Network, acc::Matrix, modifiers)
    #TODO: what about unmovables?
    for row in 1:network.row_counts[1]
        neuron_i = get_neuron_index(network, 1, row)
        if haskey(modifiers, neuron_i)
            #TODO: check indexing
            acc[:, row] += modifiers[neuron_i]
        end
    end
end

netpush!(network, acc) = addvelocity!(network, acc, [0.1, 0])
netpull!(network, acc) = addvelocity!(network, acc, [-0.1, 0])

function loss(network::Network, behaviour::Behaviour)
    s = 0
    length(behaviour.goals) == 0 && @warn "No goals in behaviour!"
    for (neuron_i, goal_pos) in behaviour.goals
        # neuron = get_neuron(network, neuron_i)
        if behaviour.relative
            start_pos = network.start_positions[neuron_i]
            goal_pos += start_pos
        end
        s += (goal_pos[1] - network.positions[1, neuron_i])^2
        #s += (goal_pos[1] - neuron.pos[1])^2
        s += (goal_pos[2] - network.positions[2, neuron_i])^2
        #s += (goal_pos[2] - neuron.pos[2])^2
    end
    return s / length(behaviour.goals)
end

function reset!(network; vis=nothing)
    set_neuron_positions!(network)
    set_neuron_velocities!(network)
    if vis !== nothing
        update_positions!(vis, network)
    end
end

function set_spring_data!(network::Network, spring_data::Dict)
    for (key, val) in spring_data
        network.graph.edge_data[key] = val
    end
end

function set_spring_data!(network::Network, spring_data::Vector{Float64})
    springs = Tuple.(edges(network.graph))
    for i in 1:length(springs)
        spring = springs[i]
        network.graph.edge_data[spring].spring_constant = spring_data[i]
    end
end

function get_spring_constants(network::Network)
    return network.graph.edge_data
end

function get_spring_constants_vec(network::Network)
    spring_constants = zeros(Float64, ne(network.graph))
    springs = Tuple.(edges(network.graph))

    for i in 1:length(springs)
        spring_constants[i] = network.graph[springs[i][1], springs[i][2]].spring_constant
    end

    return spring_constants
end

function calc_loss(network::Network, sim::Simulation, behaviours::Vector{Behaviour})
    l = 0
    for b in behaviours
        reset!(network)
        simulate!(network, sim, b)
        l += loss(network, b)
    end
    if isnan(l)
        @info "l: $l, length: $(length(behaviours))"
    end
    return l / length(behaviours)
end

function calc_losses!(
    network,
    candidates,
    losses,
    behaviours::Vector{Behaviour};
    vis=nothing,
    delta=0.01,
    epochs=250,
)
    for i in eachindex(losses)
        set_spring_data!(network, candidates[i])
        losses[i] = calc_loss(network, behaviours; vis=vis, delta=delta, epochs=epochs)
    end
end

function calc_losses_parallel!(
    network,
    candidates,
    losses,
    behaviours::Vector{Behaviour};
    vis=nothing,
    delta=0.01,
    epochs=250,
)
    candidate_is = [i for i in eachindex(candidates)]
    chunks = Iterators.partition(
        candidate_is, max(length(candidate_is) ÷ Threads.nthreads(), 1)
    )
    tasks = map(chunks) do i
        if length(i) == 1
            i = i[1]
            Threads.@spawn calc_loss(
                deepcopy($network), $(candidates[i]), behaviours, epochs=$epochs
            )
        else
            for j in i[1]:i[2]
                Threads.@spawn calc_loss(
                    deepcopy($network), $(candidates[j]), behaviours, epochs=$epochs
                )
            end
        end
    end
    results = fetch.(tasks)
    for i in eachindex(results)
        losses[i] = results[i]
    end
end

function train!(network::Network, epochs::Int, trainer::Trainer)
    return train!(
        network, epochs, trainer.behaviours, trainer.simulation, trainer.optimization
    )
end

function behaviour_unmoving(network::Network)
    col = network.columns
    rows = network.row_counts[col]
    goals = Dict{Int,Vector{Number}}()
    for row in 1:rows
        neuron_i = get_neuron_index(network, col, row)
        goals[neuron_i] = [0.0, 0.0]
    end
    return Behaviour(goals, true, Dict())
end

function random_distanced_vector(others, m, min_angle)
    j = 1
    result = [(rand() - 0.5) * m, (rand() - 0.5) * m]
    while j <= size(others, 2)
        α = angle(result, others[:, j])
        if α < min_angle
            result .= [(rand() - 0.5) * m, (rand() - 0.5) * m]
            j = 1
            # @info "Too close!"
            continue
        else
            j += 1
        end
    end
    return result
end

function create_behaviours(network::Network, num::Int; min_angle=π / 3, m_goal=1, m_mod=0.1)
    behaviours = Vector{Behaviour}(undef, num)
    goals = Array{Float64,3}(undef, 2, num, network.row_counts[end])
    modifiers = Array{Float64,3}(undef, 2, num, network.row_counts[1])
    for i in eachindex(behaviours)
        b_goals = Dict()
        b_modifiers = Dict()
        for row in 1:network.row_counts[end]
            neuron_i = get_neuron_index(network, network.columns, row)
            neuron = get_neuron(network, neuron_i)
            !neuron.movable && continue
            others = goals[:, 1:(i - 1), row]
            goals[:, i, row] .= random_distanced_vector(others, m_goal, min_angle)
            b_goals[neuron_i] = goals[:, i, row]
        end
        for row in shuffle(1:network.row_counts[1])
            neuron_i = get_neuron_index(network, 1, row)
            neuron = get_neuron(network, neuron_i)
            !neuron.movable && continue
            others = modifiers[:, 1:(i - 1), row]
            modifiers[:, i, row] .= random_distanced_vector(others, m_mod, min_angle)
            b_modifiers[neuron_i] = modifiers[:, i, row]
        end
        # behaviours[i] = behaviour_unmoving(network)
        behaviours[i] = Behaviour(b_goals, true, b_modifiers)
    end
    return behaviours
end

function Trainer(net::Network, opt::Optimization, sim::Simulation, num::Int)
    return Trainer(create_behaviours(net, num), sim, opt)
end

include("SimulationDiff.jl")
include("SimulationEuler.jl")
include("PPSOptimizer.jl")
include("Evolution.jl")
include("Backpropagation.jl")

function simulate!(
    network::Network,
    sim::Simulation,
    behaviour::Behaviour;
    vis::Union{Visualizer,Nothing}=nothing,
)
    sim.modifier = (network, acc) -> addvelocity!(network, acc, behaviour.modifiers)
    return simulate!(network, sim; vis=vis)
end

end
