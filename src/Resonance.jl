mutable struct Resonance <: Behaviour
    goals::Dict{Int,Number}
    modifiers::Dict{Int,Vector{Number}}
end

function resonate!(network::Network, acc, modifiers, t)
    for (n, m) in modifiers
        f, a = m[1], m[2] # frequency and amplitude
        acc[1, n] += sin(t * f / (2pi)) * a # only affects x component
    end
end

#TODO: implement for Euler
function calc_loss(network::Network, sim::Diff, behaviour::Resonance)
    reset!(network)
    integ = simulate!(network, sim, behaviour)
    # portion of the simulation to consider (0.2 --> last 20% of the simulation)
    portion = round(Int, integ.t / 5)
    l = 0.0
    for (n, target_a) in behaviour.goals # neuron and target amplitude
        max_x = maximum(integ.sol.u[(end - portion):end]) do curr_solution
            positions = curr_solution.x[2]
            return positions[1, n] # only x component
        end
        min_x = minimum(integ.sol.u[(end - portion):end]) do curr_solution
            positions = curr_solution.x[2]
            return positions[1, n] # only x component
        end
        actual_a = (max_x - min_x) / 2 # actual amplitude
        l += (target_a - actual_a)^2
    end
    return l / length(behaviour.goals)
end

function simulate!(
    network::Network,
    sim::Simulation,
    behaviour::Resonance;
    vis::Union{Visualizer,Nothing}=nothing,
)
    sim.modifier = (network, acc, t) -> resonate!(network, acc, behaviour.modifiers, t)
    return simulate!(network, sim; vis=vis)
end
