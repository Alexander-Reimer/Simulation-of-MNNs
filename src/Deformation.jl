mutable struct Deformation <: Behaviour
    goals::Dict{Int,Vector{Number}}
    relative::Bool
    modifiers::Dict{Int,Vector{Number}}
end

function addvelocity!(network::Network, acc::Matrix, modifiers)
    #TODO: check if neurons movable
    for (n, m) in modifiers
        acc[:, n] += m
    end
end

function calc_loss(network::Network, sim::Simulation, behaviours::Vector{Deformation})
    len = length(behaviours)
    len == 0 && throw(ArgumentError("`behaviours` can't be an empty vector"))
    l = 0
    for b in behaviours
        reset!(network)
        simulate!(network, sim, b)
        l += loss(network, b)
    end
    if isnan(l)
        @info "l: $l, length: $(len)"
    end
    return l / len
end

function simulate!(
    network::Network,
    sim::Simulation,
    behaviour::Deformation;
    vis::Union{Visualizer,Nothing}=nothing,
)
    sim.modifier = (network, acc, t) -> addvelocity!(network, acc, behaviour.modifiers)
    simulate!(network, sim; vis=vis)
    return nothing
end
