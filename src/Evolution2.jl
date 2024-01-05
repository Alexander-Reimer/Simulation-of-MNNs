#module MNN

export Network, simulate!, Trainer, train!, Visualizer, netpush!, netpull!, draw!, reset!

using DifferentialEquations
using GLMakie
using Graphs
using LinearAlgebra
using MetaGraphsNext
using Observables
using Evolutionary

struct Neuron
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
    start_positions::Dict{Int,Vector{Number}}
    positions::Array{Float64,2}
    velocities::Array{Float64,2}
end

mutable struct Behaviour
    goals::Dict{Int,Vector{Number}}
    relative::Bool
    modifier::Function
end

NoModifier() = (network, delta) -> nothing
NoBehaviour() = Behaviour(Dict(), false, NoModifier())

mutable struct Trainer
    behaviours::Vector{Behaviour}
end

function Network(graph::MetaGraphsNext.MetaGraph, rows, columns, xdist=1.0)
    row_counts = Vector{Int}(undef, columns)
    col_fixed = Vector{Bool}(undef, columns)
    neuron_count = 0
    start_positions = Dict{Int,Vector{Number}}()
    positions = zeros(Float64,0,0)
    velocities = zeros(Float64,0,0)
    Network(graph, rows, row_counts, col_fixed, columns, neuron_count, xdist,
        calc_ydist(xdist), start_positions, positions, velocities)
end

function print_graph(network::Network)
    for col in 1:network.columns
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
        network.positions[1,i] = xpos
        network.positions[2,i] = ypos
        network.start_positions[i] = [xpos, ypos]
    end
end

function set_neuron_velocities!(network::Network, column::Int)
    for row in network.row_counts[column]
        n = get_neuron_index(network, column, row)
        network.velocities[:,n] = [0.0, 0.0]
    end
end

function set_neuron_velocities!(network::Network)
    for n in 1:network.neuron_count
        network.velocities[:, n] = [0.0, 0.0]
    end
end

function set_neuron_positions!(network::Network)
    for col in 1:network.columns
        set_neuron_positions!(network, col)
    end
end

function initialize_neurons!(network::Network)
    function add_column!(rows::Int, column::Int, fixed::Bool)

        function add_neuron!(movable::Bool)
            network.neuron_count += 1
            network.graph[network.neuron_count] = Neuron(movable)
        end

        network.row_counts[column] = rows
        network.col_fixed[column] = fixed

        for row in 1:rows
            movable_neuron = !(fixed && (row == 1 || row == rows))
            add_neuron!(movable_neuron)
        end
    end

    fixed = false
    for column in 1:network.columns
        rows = fixed ? network.rows : network.rows - 1
        add_column!(rows, column, fixed)
        fixed = !fixed
    end
    network.positions = Array{Number,2}(undef, 2, network.neuron_count)
    network.velocities = Array{Number,2}(undef, 2, network.neuron_count)
    set_neuron_positions!(network)
    set_neuron_velocities!(network)
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
    print_graph(network)
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

function addvelocity!(network::Network, acc::Matrix, v::Vector)
    for row in 1:network.row_counts[1]
        acc[:,row] += v
    end
end

netpush!(network, acc) = addvelocity!(network, acc, [0.005,0])
netpull!(network, acc) = addvelocity!(network, acc, [-0.005,0])


springforce(x, k) = -x * (k + x * x)

function simulation_step(accelerations, velocities, positions, p, t)
    (network, gam, modifier) = p
    sim = network.graph
    for v in vertices(sim)
        f = [0.0, 0.0]
        if sim[v].movable  # is movable
            neuros = neighbors(sim, v)
            springs = collect(sim[neighbor, v] for neighbor in neuros)
            for j = 1 : length(springs)
                
                dif = positions[:, v] - positions[:, neuros[j]]
                dist = norm(dif)
                f += springforce((dist - springs[j].length), springs[j].spring_constant)  * dif / dist
            end
            f -= gam*velocities[:,v] # damping
            #f[2] -= 0.1 #gravity
            accelerations[:,v] = f
            # accelerations[1,v] = f[1]
            # accelerations[2,v] = f[2]
        end
    end
    modifier(network, accelerations)
end

function simulate!(network, tspan; modifier::Function=NoModifier())
    p = (network,0.1,modifier)

    prob = SecondOrderODEProblem{true}(simulation_step, network.velocities, network.positions, tspan, p)
    #prob = SteadyStateProblem(SecondOrderODEProblem{true}(simulation_step, network.velocities, network.positions, tspan, p))

    sol = solve(prob, reltol=1e-7; saveat = 0.1)
    #sol = solve(prob, DynamicSS(Tsit5()))
    mat = hcat(sol.u[:]...)
    pos = mat[Int(size(mat)[1]/2+1):size(mat)[1],size(mat)[2]]

    positions = zeros(Float64, 2, Int(length(pos)/2))
    for i = 1:length(pos)
        if i%2 == 0
            positions[2,Int(i/2)] = pos[i]
        else
            positions[1,Int((i+1)/2)] = pos[i]
        end
    end
    network.positions = positions
    return mat[Int(size(mat)[1]/2+1):size(mat)[1],:]
end

function draw!(r)
    fig = GLMakie.Figure()
    ax = Axis(fig[1,1]) 
    global ox = Observable(zeros(Float64, Int(size(r)[1] / 2 )))
    global oy = Observable(zeros(Float64, Int(size(r)[1] / 2 )))
    GLMakie.scatter!(ox,oy, marker = :circle, markersize = 25, color = :blue)
    display(fig)

    x = zeros(Float64, Int(size(r)[1] / 2 ))
    y = zeros(Float64, Int(size(r)[1] / 2 ))
    for i = 1:(size(r)[2])
        for j = 1:size(r)[1]
            if j%2 == 0
                y[Int(j/2)] = r[j,i]
            else
                x[Int(j/2+0.5)] = r[j,i]
            end
        end
        ox[] = x
        oy[] = y
        sleep(0.008)
    end   
end

function simulate_b!(network::Network; behaviour::Behaviour=NoBehaviour())
    simulate!(network, (0.0,100.0), modifier=behaviour.modifier)
end

function loss(network::Network, behaviour::Behaviour)
    s = 0
    for (neuron_i, goal_pos) in behaviour.goals
        neuron = get_neuron(network, neuron_i)
        if behaviour.relative
            start_pos = network.start_positions[neuron_i]
            goal_pos += start_pos
        end
        s += (goal_pos[1] - network.positions[1, neuron_i])^2
        s += (goal_pos[2] - network.positions[2, neuron_i])^2
    end
    return s / length(behaviour.goals)
end

function reset!(network)
    set_neuron_positions!(network)
    set_neuron_velocities!(network)
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
    end

    display(fig)
    return vis
end

function set_spring_data!(network::Network, spring_data::Vector{Float64})
    springs = Tuple.(edges(network.graph))
    for i =  1:length(springs)
        spring = springs[i]
        network.graph.edge_data[spring].spring_constant = spring_data[i]
    end
end

function get_spring_constants_vec(network::Network)
    spring_constants = zeros(Float64, ne(network.graph))
    springs = Tuple.(edges(network.graph))

    for i = 1:length(springs)
        spring_constants[i] = network.graph[springs[i][1], springs[i][2]].spring_constant
    end

    return spring_constants
end

function calc_loss(spring_vec::Vector, network::Network, trainer::Trainer)
    l = 0
    set_spring_data!(network, spring_vec)
    for b in trainer.behaviours
        reset!(network)
        simulate_b!(network, behaviour=b)
        #println(network.positions)
        l += loss(network, b)
    end

    return l
end

function mutation(spring_data, strength)
    return spring_data + (rand(length(spring_data)).-0.5) * strength
end

function get_parent_index(index)
    i = 1
    while rand() < 0.55 && i<length(index)
        i+=1
    end
    #println(i)
    return(index[i])
end

function cross_over(candidates, index, out_size)
    out = [zeros(length(candidates[1])) for i = 1:out_size]
    for i = 1:Int(ceil(out_size/2))
        out1, out2 = single_point_crossover(candidates[get_parent_index(index)], candidates[get_parent_index(index)])
        out[i*2-1] = out1
        out[i*2] = out2
    end
    return out
end

function single_point_crossover(can1, can2)
    l = length(can1)
    out1 = zeros(l)
    out2 = zeros(l)
    p = rand(2:l-1)
    out1[1:p] = can1[1:p]
    out2[1:p] = can2[1:p]
    out1[p+1:l] = can2[p+1:l]
    out2[p+1:l] = can1[p+1:l]
    return(out1, out2)
end

function train1(pop_size, network, trainer)
    strength = 0.00005
    spring_data = get_spring_constants_vec(network)
    loss = calc_loss(spring_data, network, trainer)
    candidates = [mutation(spring_data) for i = 1:pop_size]

    for i = 1:10
        for i2 = 1:length(candidates)
            candidates[i2] = mutation(candidates[i2], strength)
        end
        losses = [calc_loss(candidate, network, trainer) for candidate in candidates]
        index = sortperm(losses)
        if losses[index[1]] < loss
            spring_data = candidates[index[1]]
            loss = losses[index[1]]
        else
            strength *= 0.7
        end
        println(loss)

        next_gen = [zeros(length(spring_data)) for i = 1:pop_size]
        next_gen[1:Int(floor(pop_size/5))] = candidates[index[1:Int(floor(pop_size/5))]]  # copying best 20%
        next_gen[Int(floor(pop_size/5))+1:length(next_gen)] = cross_over(candidates, index, Int(ceil(pop_size*0.8)))  # other 80% are crossover
        candidates = copy(next_gen)
    end
    set_spring_data!(n, spring_data)
end

function train2(pop_size, network, trainer)
    spring_data = get_spring_constants_vec(network)
    loss = calc_loss(spring_data, network, trainer)
    

    for i = 1:1
        candidates = [mutation(spring_data, 0.0001) for i = 1:pop_size]
        losses = [calc_loss(candidate, network, trainer) for candidate in candidates]
        index = sortperm(losses)
        if losses[index[1]] < loss
            spring_data = candidates[index[1]]
            loss = losses[index[1]]
        end
        println(loss)
    end
    set_spring_data!(n, spring_data)
end

function behaviour_unmoving(network::Network)
    col = network.columns
    rows = network.row_counts[col]
    goals = Dict{Int,Vector{Number}}()
    for row in 1:rows
        neuron_i = get_neuron_index(network, col, row)
        goals[neuron_i] = [0.0, 0.0]
    end
    return Behaviour(goals, true, NoModifier())
end

function Trainer(network::Network)
    col = network.columns
    rows = network.row_counts[col]

    mid_neuron_i = get_neuron_index(network, col, rows ÷ 2)

    behaviour_pull = behaviour_unmoving(network)
    behaviour_pull.goals[mid_neuron_i] = [-1.0, 0.0]
    behaviour_pull.modifier = netpush!

    behaviour_push = behaviour_unmoving(network)
    behaviour_push.goals[mid_neuron_i] = [1.0, 0.0]
    behaviour_push.modifier = netpull!

    return Trainer([behaviour_pull, behaviour_push])
end

function modifier1!(network, acc)
    acc[:,1] += [-0.12, -0.05] * 0.01
    acc[:,2] += [0.07, 0.14] * 0.01
    acc[:,3] += [-0.41, 0.34] * 0.01
    acc[:,4] += [-0.09, -0.44] * 0.01
end

function modifier2!(network, acc)
    acc[:,1] += [-0.32, 0.36] * 0.01
    acc[:,2] += [0.45, 0.01] * 0.01
    acc[:,3] += [-0.48, -0.34] * 0.01
    acc[:,4] += [-0.30, -0.45] * 0.01
end

function RandTrainer(network) # Random behaviours for benchmarking of a Network with size (11, 5)
    col = network.columns
    rows = network.row_counts[col]

    behaviour1 = behaviour_unmoving(network)
    behaviour1.goals[46] = [0.12, 0.43]
    behaviour1.goals[47] = [-0.41, 0.38]
    behaviour1.goals[48] = [0.19, -0.28]
    behaviour1.goals[49] = [-0.47, -0.36]
    behaviour1.modifier = modifier1!

    behaviour2 = behaviour_unmoving(network)
    behaviour2.goals[46] = [-0.38, 0.17]
    behaviour2.goals[47] = [-0.21, -0.10]
    behaviour2.goals[48] = [0.29, -0.34]
    behaviour2.goals[49] = [-0.11, 0.27]
    behaviour2.modifier = modifier2!

    return Trainer([behaviour1, behaviour2])
end

#end