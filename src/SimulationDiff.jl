"""
    mutable struct Diff <: Simulation
"""
mutable struct Diff <: Simulation
    time::Number
    modifier::Function
end
Diff(time::Number) = Diff(time, (network, accelerations, t) -> nothing)

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
    for neuron in vertices(graph)
        if graph[neuron].movable  # is movable
            force .= [0.0, 0.0]
            positionsn .= positions[:, neuron]

            for neighbor in neighbors(graph, neuron)
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

    return modifier(network, accelerations, t)
end

function simulate!(network::Network, sim::Diff; vis::Union{Visualizer,Nothing}=nothing)
    p = (network, 0.2, sim.modifier)
    tspan = (0.0, sim.time)
    prob = SecondOrderODEProblem{true}(
        simulation_step, network.velocities, network.positions, tspan, p
    )

    if vis === nothing
        integrator = init(prob, AutoTsit5(Rosenbrock23()))
    else
        integrator = init(prob, AutoTsit5(Rosenbrock23()))
    end
    for integ in integrator
        if vis !== nothing
            network.positions = integ.sol.u[end].x[2]
            update_positions!(vis, network)
            # @info "t: $(integ.t)"
            sleep(0.01)
        end
        # TODO: maximum?
        mean_pos_change = sum(abs, integ.u.x[2] .- integ.uprev.x[2]) / length(integ.u.x[2])
        mean_velocity = sum(abs, integ.u.x[1]) / length(integ.u.x[1])
        if integ.t > 5 && mean_pos_change < 1.5e-5 && mean_velocity < 1.5e-5
            # @info "Early break at: $(integ.t)"
            break
        end
    end
    if (integrator.t == sim.time)
        # @warn "Simulation did not reach steady state!"
    end
    network.positions = integrator.sol.u[end].x[2]
    return integrator
end
