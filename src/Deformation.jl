mutable struct Deformation <: Behaviour
    goals::Dict{Int,Vector{Number}}
    relative::Bool
    modifiers::Dict{Int,Vector{Number}}
end

function create_deformation_unmoving(network::Network)
    col = network.columns
    rows = network.row_counts[col]
    goals = Dict{Int,Vector{Number}}()
    for row in 1:rows
        neuron_i = get_neuron_index(network, col, row)
        goals[neuron_i] = [0.0, 0.0]
    end
    return Behaviour(goals, true, Dict())
end

function random_distanced_vector(others, m, min_angle)
    j = 1
    result = [(rand() - 0.5) * m, (rand() - 0.5) * m]
    while j <= size(others, 2)
        α = angle(result, others[:, j])
        if α < min_angle
            result .= [(rand() - 0.5) * m, (rand() - 0.5) * m]
            j = 1
            # @info "Too close!"
            continue
        else
            j += 1
        end
    end
    return result
end

function create_deformation_behaviours(
    network::Network, num::Int; min_angle=π / 3, m_goal=1, m_mod=0.1
)
    behaviours = Vector{Deformation}(undef, num)
    goals = Array{Float64,3}(undef, 2, num, network.row_counts[end])
    modifiers = Array{Float64,3}(undef, 2, num, network.row_counts[1])

    for i in eachindex(behaviours)
        b_goals = Dict()
        b_modifiers = Dict()
        for row in 1:network.row_counts[end]
            neuron_i = get_neuron_index(network, network.columns, row)
            neuron = get_neuron(network, neuron_i)
            !neuron.movable && continue
            others = goals[:, 1:(i - 1), row]
            goals[:, i, row] .= random_distanced_vector(others, m_goal, min_angle)
            b_goals[neuron_i] = goals[:, i, row]
        end
        for row in shuffle(1:network.row_counts[1])
            neuron_i = get_neuron_index(network, 1, row)
            neuron = get_neuron(network, neuron_i)
            !neuron.movable && continue
            others = modifiers[:, 1:(i - 1), row]
            modifiers[:, i, row] .= random_distanced_vector(others, m_mod, min_angle)
            b_modifiers[neuron_i] = modifiers[:, i, row]
        end
        behaviours[i] = Deformation(b_goals, true, b_modifiers)
    end
    return behaviours
end

function addvelocity!(network::Network, acc::Matrix, modifiers)
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
