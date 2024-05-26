"""
    mutable struct Diff <: Simulation
"""
mutable struct FirstOrderDiff <: Simulation
    time::Number
    modifier::Function
end
FirstOrderDiff(time::Number) = FirstOrderDiff(time, (network, accelerations, t) -> nothing)

springforce(x, k) = -x * (k + x * x)

function simulation_step(du, d, p, t)
    (network, gam, modifier) = p
    positions = d[:,:,1]
    velocities = d[:,:,2]
    du[:,:,1] = velocities
    om = modifier()
    [du[:,i,1] += [sin(t*om)*om*0.2,0] for i in 1:network.columns]

    graph = network.graph
    force = [0.0,0.0]

    for neuron in vertices(graph)
        if graph[neuron].movable  # is movable
            force = [0.0, 0.0]

            for neighbor in neighbors(graph, neuron)
                spring = graph[neighbor, neuron]
                diff = positions[:, neuron] - positions[:, neighbor]
                dist = norm(diff)
                # diff .= springforce(dist - spring.length, spring.spring_constant) .* diff / dist
                diff *= springforce(dist - spring.length, spring.spring_constant)
                if dist == 0 
                    println("Nan")
                end
                force += diff / dist
            end
            force -= gam * velocities[:,neuron] # damping
            force[2] -= 0.05 #gravity
            
            du[:,neuron,2] = force
            
        end
    end

    #modifier(network, accelerations, t)
    
end

function simulate!(network::Network, sim::FirstOrderDiff; vis::Union{Visualizer,Nothing}=nothing)
    p = (network, 0.2, sim.modifier)
    tspan = (0.0, sim.time)

    u = cat(network.positions, zeros(size(network.positions)),dims=(3)) 
    prob = ODEProblem(
        simulation_step, u, tspan, p
    )
    sol = solve(prob, saveat=1.0)
    return sol
end