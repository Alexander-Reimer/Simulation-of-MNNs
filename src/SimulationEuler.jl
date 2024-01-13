mutable struct Euler <: Simulation
    time::Number
    delta::Number
    showfps::Bool
    modifier::Function
end
Euler() = Euler(0.01, false, NoModifier(), 1.0)

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
    network.velocities[:, n] .+= f
    neuron.velocity .*= 0.95
    neuron.pos .+= neuron.velocity * delta
end

function addvelocity!(network::Network, delta::Number, v::Number)
    for row in 1:network.row_counts[1]
        get_neuron(network, 0, row).velocity[1] += v / (1 / delta) #TODO: *
    end
end

function simulate!(network::Network, sim::Euler; vis::Union{Visualizer,Nothing}=nothing)
    sim.showfps ? start_time = time() : nothing
    steps = sim.time / sim.delta
    for _ in 1:steps
        sim.modifier(network, sim.delta)
        for n in 1:network.neuron_count
            update_position!(network, n, sim.delta)
        end
        if vis !== nothing
            update_positions!(vis, network)
            sleep(sim.delta)
        end
        # TODO: adjust delta automatically dependant on fps?
    end
    if sim.showfps
        fps = sim.steps / (time() - start_time)
        @info "FPS: $fps"
    end
end
