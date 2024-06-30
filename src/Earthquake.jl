mutable struct House <: Network
    graph::MetaGraphsNext.MetaGraph
    rows::Int
    columns::Int
    neuron_count::Int
    start_positions::Dict{Int,Vector{Number}}
    positions::Array{Float64,2}
    velocities::Array{Float64,2}
end

function House(rows, columns)
    graph = new_graph()
    network = House(graph, rows, columns)
    initialize_graph!(network)
    return network
end

function House(graph::MetaGraphsNext.MetaGraph, rows, columns)
    row_counts = Vector{Int}(undef, columns)
    col_fixed = Vector{Bool}(undef, columns)
    neuron_count = 0
    start_positions = Dict{Int,Vector{Number}}()
    positions = zeros(Float64, 0, 0)
    velocities = zeros(Float64, 0, 0)
    return House(graph, rows, columns, neuron_count, start_positions, positions, velocities)
end

function initialize_graph!(network::House)
    initialize_neurons!(network)
    return initialize_springs!(network)
end

function set_neuron_velocities!(network::House)
    for n in 1:(network.neuron_count)
        network.velocities[:, n] = [0.0, 0.0]
    end
end

function set_neuron_positions!(network::House)
    for row in 1:(network.rows)
        for col in 1:(network.columns)
            network.positions[:, neuron_index(network, col, row)] = [col, row]
            network.start_positions[neuron_index(network, col, row)] = [col, row]
        end
    end
end

function initialize_neurons!(network::House)
    function add_neuron!(movable::Bool)
        network.neuron_count += 1
        return network.graph[network.neuron_count] = Neuron(movable)
    end

    for i2 in 1:(network.columns)
        add_neuron!(false)
    end
    for i in 1:(network.rows - 1)
        for i2 in 1:(network.columns)
            add_neuron!(true)
        end
    end

    network.positions = Array{Number,2}(undef, 2, network.neuron_count)
    network.velocities = Array{Number,2}(undef, 2, network.neuron_count)
    set_neuron_positions!(network)
    return set_neuron_velocities!(network)
end

function initialize_springs!(network::House)
    function add_spring!(n1::Int, n2::Int)
        pos1 = network.positions[:, n1]
        pos2 = network.positions[:, n2]
        dx = pos1[1] - pos2[1]
        dy = pos1[2] - pos2[2]
        len = sqrt(dx^2 + dy^2)
        return add_edge!(network.graph, n1, n2, Spring(abs(spring_init() * 10), len))
    end

    for row in 1:(network.rows)
        for col in 1:(network.columns)
            if row + 1 <= network.rows
                add_spring!(
                    neuron_index(network, col, row), neuron_index(network, col, row + 1)
                )
            end
            if col + 1 <= network.columns
                add_spring!(
                    neuron_index(network, col, row), neuron_index(network, col + 1, row)
                )
            end
            if col - 1 >= 1 && row + 1 <= network.rows
                add_spring!(
                    neuron_index(network, col, row), neuron_index(network, col - 1, row + 1)
                )
            end
            if col + 1 <= network.columns && row + 1 <= network.rows
                add_spring!(
                    neuron_index(network, col, row), neuron_index(network, col + 1, row + 1)
                )
            end
        end
    end
end

function neuron_index(network::House, col, row)
    return (row - 1) * network.columns + col
end

mutable struct Shaking <: Behaviour
    frequency::Float64
end

frequency(shaking::Shaking) = shaking.frequency

function simulate!(
    network::House,
    sim::Simulation,
    behaviour::Shaking;
    vis::Union{Visualizer,Nothing}=nothing,
)
    sim.modifier = () -> frequency(behaviour)
    return simulate!(network, sim; vis=vis)
end


function calc_loss(network::House, sim::Simulation, behaviours::Vector{Shaking})
    loss = 0
    for b in behaviours
        reset!(network)
        sol = simulate!(network, sim, b).sol
        network.positions = sol.u[length(sol.u)][:, :, 1]
        #vis = Visualizer(network)
        loss += sum([
            norm(network.start_positions[i] - network.positions[:, i]) for
            i in 1:size(network.positions)[2]
        ])
    end
    return loss / length(behaviours)
end
