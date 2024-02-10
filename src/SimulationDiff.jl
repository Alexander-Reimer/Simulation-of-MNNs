mutable struct Diff <: Simulation
    time::Number
    modifier::Function
end
Diff(time::Number) = Diff(time, (network, accelerations) -> nothing)

springforce(x, k) = -x * (k + x * x)

function simulation_step(accelerations, velocities, positions, p, t)
    (network, gam, modifier) = p
    # velocities = network.velocities
    # positions = network.positions
    graph = network.graph
    force = MVector{2,Float64}(undef)
    diff = MVector{2,Float64}(undef)
    positionsn = MVector{2,Float64}(undef)
    # diff = Vector{Float64}(undef,2)
    @fastmath @inbounds @simd ivdep for neuron in vertices(graph)
        if graph[neuron].movable  # is movable
            force .= [0.0, 0.0]
            positionsn .= positions[:, neuron]

            @simd ivdep for neighbor in neighbors(graph, neuron)
                spring = graph[neighbor, neuron]
                diff = positionsn - positions[:, neighbor]
                dist = norm(diff)
                # diff .= springforce(dist - spring.length, spring.spring_constant) .* diff / dist
                diff *= springforce(dist - spring.length, spring.spring_constant)
                diff /= dist
                force += diff
            end
            force -= gam * view(velocities, :, neuron) # damping
            #f[2] -= 0.1 #gravity
            accelerations[:, neuron] = force
            # accelerations[1,v] = f[1]
            # accelerations[2,v] = f[2]
        end
    end

    return modifier(network, accelerations)
end

function simulate!(network::Network, sim::Diff; vis::Union{Visualizer,Nothing}=nothing)
    p = (network, 0.1, sim.modifier)
    tspan = (0.0, sim.time)
    # prob = SteadyStateProblem(SecondOrderODEProblem{true}(
    #     simulation_step, network.velocities, network.positions, tspan, p
    # ))
    prob = SecondOrderODEProblem{true}(simulation_step, network.velocities, network.positions, tspan, p)

    if vis === nothing
        integrator = init(prob, AutoTsit5(Rosenbrock23()), saveat=tspan[end])
    else
        integrator = init(prob, AutoTsit5(Rosenbrock23()))
    end
    for integ in integrator
        global myinteg = integ
        if vis !== nothing
            network.positions = integ.sol.u[end].x[2]
            update_positions!(vis, network)
            sleep(0.01)
        end
    end
    network.positions = integrator.sol.u[end].x[2]
end
