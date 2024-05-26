mutable struct Shake <: Behaviour
    frequency::Float64
end

mutable struct HouseNetwork <: Network 
    graph::MetaGraphsNext.MetaGraph
    rows::Int
    columns::Int
    neuron_count::Int
    start_positions::Dict{Int,Vector{Number}}
    positions::Array{Float64,2}
    velocities::Array{Float64,2}
end

function get_frequency(behaviour)
    return behaviour.frequency
end

function calc_loss(network::HouseNetwork, sim::Simulation, behaviours::Vector{Shake})
    loss = 0
    for b in behaviours
        reset!(network)
        sol = simulate!(network,sim,b)
        network.positions = sol.u[length(sol.u)][:,:,1]
        #vis = Visualizer(network)
        loss += sum([norm(network.start_positions[i]-network.positions[:,i]) for i in 1:size(network.positions)[2]])
    end
    return loss / length(behaviours)
end

function simulate!(
    network::Network,
    sim::Simulation,
    behaviour::Shake;
    vis::Union{Visualizer,Nothing}=nothing,
)
    sim.modifier = () -> get_frequency(behaviour)
    return simulate!(network, sim; vis=vis)
end

function HouseNetwork(rows, columns)
    graph = new_graph()
    network = HouseNetwork(graph, rows, columns)
    initialize_graph!(network)
    return network
end



function HouseNetwork(graph::MetaGraphsNext.MetaGraph, rows, columns)
    row_counts = Vector{Int}(undef, columns)
    col_fixed = Vector{Bool}(undef, columns)
    neuron_count = 0
    start_positions = Dict{Int,Vector{Number}}()
    positions = zeros(Float64, 0, 0)
    velocities = zeros(Float64, 0, 0)
    return HouseNetwork(
        graph,
        rows,
        columns,
        neuron_count,
        start_positions,
        positions,
        velocities,
    )
end

function initialize_graph!(network::HouseNetwork)
    initialize_neurons!(network)
    return initialize_springs!(network)
end

function set_neuron_velocities!(network::HouseNetwork)
    for n in 1:(network.neuron_count)
        network.velocities[:, n] = [0.0, 0.0]
    end
end

function set_neuron_positions!(network::HouseNetwork)
    for row in 1:(network.rows)
        for col in 1:(network.columns)
            network.positions[:, neuron_index(network,col,row)] = [col,row]
            network.start_positions[neuron_index(network,col,row)] = [col,row]
        end
    end
end

function initialize_neurons!(network::HouseNetwork)
    function add_neuron!(movable::Bool)
        network.neuron_count += 1
        return network.graph[network.neuron_count] = Neuron(movable)
    end

    for i2 = 1:network.columns
        add_neuron!(false)
    end
    for i = 1:(network.rows-1)
        for i2 = 1:network.columns
            add_neuron!(true)
        end
    end

    network.positions = Array{Number,2}(undef, 2, network.neuron_count)
    network.velocities = Array{Number,2}(undef, 2, network.neuron_count)
    set_neuron_positions!(network)
    return set_neuron_velocities!(network)
end

function initialize_springs!(network::HouseNetwork)
    function add_spring!(n1::Int, n2::Int)
        pos1 = network.positions[:, n1]
        pos2 = network.positions[:, n2]
        dx = pos1[1] - pos2[1]
        dy = pos1[2] - pos2[2]
        len = sqrt(dx^2 + dy^2)
        return add_edge!(network.graph, n1, n2, Spring(abs(spring_init()*10), len))
    end
    
    for row = 1:(network.rows)
        for col = 1:(network.columns)
            if row+1 <= network.rows
                add_spring!(neuron_index(network,col,row), neuron_index(network,col,row+1))
            end
            if col+1 <= network.columns
                add_spring!(neuron_index(network,col,row), neuron_index(network,col+1,row))
            end
            if col-1>=1 && row+1 <= network.rows
                add_spring!(neuron_index(network,col,row), neuron_index(network,col-1,row+1))
            end
            if col+1<=network.columns && row+1 <= network.rows
                add_spring!(neuron_index(network,col,row), neuron_index(network,col+1,row+1))
            end
        end
    end
    
end

function neuron_index(network, col, row)
    return (row-1)*network.columns+col
end 
