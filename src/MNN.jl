module MNN

# using GLMakie
using Graphs
using LinearAlgebra
using MetaGraphsNext

struct Neuron
    # https://jkrumbiegel.com/pages/2020-10-31-tuples-and-vectors/
    pos::Tuple{Float64, Float64}
    velocity::Vector{Float64}
    movable::Bool
end

struct Spring
    spring_constant::Float64
    length::Float64
end
spring_init() = rand() - 0.5

mutable struct Network
    graph::MetaGraphsNext.MetaGraph
    rows::Int
    columns::Int
    neuron_count::Int
end

function print_graph(network::Network)
    for n in network.graph.vertex_properties |> values
        println("Pos: $(n[2].pos), movable: $(n[2].movable)")
    end
end

function add_column!(network::Network, rows::Int, column::Int, movable::Bool, yoffset=sqrt(0.75)*2, xoffset=1)
    function add_neuron!(movable::Bool, xpos::Number, ypos::Number, row::Number)
        network.graph[network.neuron_count] = Neuron((xpos, ypos), [0, 0], movable)
        network.neuron_count += 1
    end
    for row in 1:rows
        movable_neuron = movable || !(row == 1 || row == rows)
        add_neuron!(movable_neuron, (column-1) * xoffset, (row-1) * yoffset, row)
    end
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

function initialize_graph!(network::Network)
    yoffset = sqrt(0.75)
    for column in 1:network.columns
        add_column!(network, network.rows, column, false)
    end
    print_graph(network)
    
    res = true
    # TODO
    rows = network.rows
    cols = network.columns
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
    return res
end

function Network(columns, rows)
    graph = new_graph()
    network = Network(graph, rows, columns, 1)
    res = initialize_graph!(network)
    @info "Successful initialization?" res
    print_graph(network)
    return network
end

end