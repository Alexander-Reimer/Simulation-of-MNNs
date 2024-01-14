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
    vis !== nothing && @warn "Visualizer is not implemented for Diff simulation"

    p = (network, 0.1, sim.modifier)
    tspan = (0.0, sim.time)
    prob = SecondOrderODEProblem{true}(
        simulation_step, network.velocities, network.positions, tspan, p
    )
    #prob = SteadyStateProblem(SecondOrderODEProblem{true}(simulation_step, network.velocities, network.positions, tspan, p))

    sol = solve(prob, AutoTsit5(Rosenbrock23()); saveat=tspan[2])
    # , reltol=1e-7; saveat = tspan[2])
    #sol = solve(prob, DynamicSS(Tsit5()))
    mat = hcat(sol.u[:]...)
    pos = mat[Int(size(mat)[1] / 2 + 1):size(mat)[1], size(mat)[2]]

    positions = zeros(Float64, 2, Int(length(pos) / 2))
    for i in eachindex(pos)
        if i % 2 == 0
            positions[2, Int(i / 2)] = pos[i]
        else
            positions[1, Int((i + 1) / 2)] = pos[i]
        end
    end
    network.positions = positions
    return mat[Int(size(mat)[1] / 2 + 1):size(mat)[1], :]
end
