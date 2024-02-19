mutable struct Resonance <: Behaviour
    goals::Dict{Int,Vector{Number}}
    modifiers::Dict{Int,Vector{Number}}
end

function resonate!(network::Network, acc, modifiers, t)
    for (n, m) in modifiers
        f, a = m[1], m[2] # frequency and amplitude
        acc[1, n] += sin(t * f / (2pi)) * a # only affects x component
    end
end

function calc_loss(network::Network, sim::Simulation, behaviours::Vector{Resonance})
    l = 0
    for b in behaviours
        reset!(network)
        simulate!(network, sim, b)
        l += loss(network, b)
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
    @info "here"
    sim.modifier = (network, acc, t) -> resonate!(network, acc, behaviour.modifiers, t)
    return simulate!(network, sim; vis=vis)
end
