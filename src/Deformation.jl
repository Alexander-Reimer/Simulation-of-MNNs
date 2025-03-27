mutable struct Deformation <: Behaviour
    goals::Dict{Int,Vector{Number}}
    relative::Bool
    modifiers::Dict{Int,Vector{Number}}
end

function addvelocity!(network::Network, acc, modifiers)
    #TODO: check if neurons movable
    for (n, m) in modifiers
        acc[:, n] += m
    end
end

function calc_loss(network::Network, sim::Simulation, behaviour::Deformation)
    reset!(network)
    simulate!(network, sim, behaviour)
    return loss(network, behaviour)
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
