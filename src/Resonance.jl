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
function calc_loss(network::Network, sim::Diff, behaviours::Vector{Resonance})
    l = 0
    for b in behaviours
        reset!(network)
        integ = simulate!(network, sim, b)
        foo = round(Int, integ.t / 5)
        b_l = 0.0
        for (n, target_a) in b.goals # neuron and target amplitude
            max_x = maximum(integ.sol.u[end-foo:end]) do curr_solution
                positions = curr_solution.x[2]
                return positions[1, n] # only x component
            end
            min_x = minimum(integ.sol.u[(end - foo + 1):end]) do curr_solution
                positions = curr_solution.x[2]
                return positions[1, n] # only x component
            end
            actual_a = (max_x - min_x) / 2 # actual amplitude
            b_l += (target_a - actual_a)^2
        end
        l += b_l / length(b.goals)
    end
    if isnan(l)
        @info "l: $l, length: $(length(behaviours))"
    end
    return l / length(behaviours)
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
