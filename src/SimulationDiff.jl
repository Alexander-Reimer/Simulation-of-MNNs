export SecondOrderDiff, FirstOrderDiff
"""
    mutable struct FirstOrderDiff <: Simulation
"""
mutable struct FirstOrderDiff <: Simulation
    time::Number
    modifier::Function
end
FirstOrderDiff(time::Number) = FirstOrderDiff(time, (network, accelerations, t) -> nothing)

"""
    mutable struct SecondOrderDiff <: Simulation
"""
mutable struct SecondOrderDiff <: Simulation
    time::Number
    modifier::Function
end
function SecondOrderDiff(time::Number)
    return SecondOrderDiff(time, (network, accelerations, t) -> nothing)
end

springforce(x, k) = -x * (k + x * x)

function simulation_step_first_order(du, d, p, t)
    (network, gam, modifier) = p
    positions = d[:, :, 1]
    velocities = d[:, :, 2]
    du[:, :, 1] = velocities
    om = modifier()
    # add acceleration to velocities of first neuron in every column
    [du[:, i, 1] += [sin(t * om) * om * 0.2, 0] for i in 1:(network.columns)]

    graph = network.graph
    force = [0.0, 0.0]

    for neuron in vertices(graph)
        if graph[neuron].movable # is movable
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
            force -= gam * velocities[:, neuron] # damping
            force[2] -= 0.05 #gravity
            # prevent neurons from going below ground
            if positions[2, neuron] < 1.0
                force[2] = 0.0
                du[2, neuron, 1] = 0.0
            end
            du[:, neuron, 2] = force
        end
    end
end

function simulation_step_second_order(accelerations, velocities, positions, p, t)
    (network, gam, modifier) = p
    # velocities = network.velocities
    # positions = network.positions
    graph = network.graph
    force = MVector{2,Float64}(undef)
    diff = MVector{2,Float64}(undef)
    # diff = Vector{Float64}(undef,2)
    for neuron in vertices(graph)
        if graph[neuron].movable  # is movable
            force .= [0.0, 0.0]

            for neighbor in neighbors(graph, neuron)
                spring = graph[neighbor, neuron]
                diff = positions[:, neuron] - positions[:, neighbor]
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

function simulate!(
    network::House, sim::FirstOrderDiff; vis::Union{Visualizer,Nothing}=nothing
)
    p = (network, 0.2, sim.modifier)
    tspan = (0.0, sim.time)

    u = cat(network.positions, zeros(size(network.positions)); dims=(3))
    prob = ODEProblem(simulation_step_first_order, u, tspan, p)
    if vis === nothing
        integrator = init(prob, AutoTsit5(Rosenbrock23()))
    else
        integrator = init(prob, AutoTsit5(Rosenbrock23()))
    end
    # sol = solve(prob)
    for integ in integrator
        if vis !== nothing
            network.positions = integ.sol.u[end][:, :, 1]
            update_positions!(vis, network)
            # @info "t: $(integ.t)"
            sleep(integ.dt / 100)
        end
    end
    # network.positions = integrator.sol.u[end].x[2]
    return integrator
end

function simulate!(
    network::Network, sim::SecondOrderDiff; vis::Union{Visualizer,Nothing}=nothing
)
    p = (network, 0.2, sim.modifier)
    tspan = (0.0, sim.time)
    prob = SecondOrderODEProblem{true}(
        simulation_step_second_order, network.velocities, network.positions, tspan, p
    )

    if vis === nothing
        sol = solve(prob, AutoTsit5(Rosenbrock23());)
        network.velocities = sol.u[end].x[1]
        network.positions = sol.u[end].x[2]
        return sol
    else
        integrator = init(prob, AutoTsit5(Rosenbrock23()))
        for integ in integrator
            if vis !== nothing
                network.positions = integ.sol.u[end].x[2]
                update_positions!(vis, network)
                sleep(integ.dt / 100)
            end
            # TODO: maximum?
            mean_pos_change =
                sum(abs, integ.u.x[2] .- integ.uprev.x[2]) / length(integ.u.x[2])
            mean_velocity = sum(abs, integ.u.x[1]) / length(integ.u.x[1])
            if integ.t > 5 && mean_pos_change < 1e-3 && mean_velocity < 1e-3
                @info "Early break at: $(integ.t)s"
                break
            end
        end
        network.velocities = integrator.sol.u[end].x[1]
        network.positions = integrator.sol.u[end].x[2]
        return integrator
    end
end
