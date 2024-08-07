using Evolutionary

mutable struct Evolution <: Optimization
    parallel::Bool
    mutation_strength::Float64
    popsize::Int
    candidates::Vector{Vector{Float64}}
    simepochs::Int
end

function Evolution(net::Network)
    spring_data = get_spring_constants_vec(net)
    popsize = 10
    simepochs = 500
    mutation_strength = 0.00005
    candidates = [mutation(spring_data, mutation_strength) for _ in 1:popsize]
    return Evolution(true, mutation_strength, popsize, candidates, simepochs)
end

function mutate!(spring_data::Vector, strength=0.1)
    for i in eachindex(spring_data)
        spring_data[i] += strength * (rand() - 0.5)
    end
    return spring_data
end

function create_mutation(spring_data::Vector{Float64}, strength=0.1)
    return mutate!(deepcopy(spring_data), strength)
end

function create_mutations!(
    spring_data::Vector{Float64}, mutations::Vector{Vector{Float64}}; strength=0.1
)
    for i in eachindex(mutations)
        mutations[i] = create_mutation(spring_data, strength)
    end
end

# function train!(network::Network, epochs::Int, behaviours::Vector{Behaviour}, opt::Evolution)
#     if parallel && Threads.nthreads() <= 1
#         @warn "You have set parallel=true but don't have more than one thread assigned to julia!"
#         opt.parallel = false
#     end
#     loss_function! = parallel ? calc_losses_parallel! : calc_losses!

#     candidates = [rand(ne(network.graph)) for i in 1:opt.mutations]
#     best_candidate = deepcopy(get_spring_constants_vec(network))
#     best_loss = calc_loss(network, best_candidate, behaviours, epochs=simepochs)
#     candidate_losses = Vector{Float64}(undef, mutations)

#     for i = 1:epochs
#         #c = copy(candidates)
#         create_mutations!(best_candidate, candidates, strength=opt.mutation_strength)
#         #println(candidates-c)
#         loss_function!(network, candidates, candidate_losses, behaviours;
#             vis=vis, epochs=simepochs)
#         best_i = argmin(candidate_losses)
#         if candidate_losses[best_i] < best_loss
#             best_loss = candidate_losses[best_i]
#             best_candidate = deepcopy(candidates[best_i])
#             @info "\tUpdated!"
#         end
#         @info "Epoch: $i, best loss: $(best_loss)"
#     end
#     set_spring_data!(network, best_candidate)
#     # this call to calc_loss also saves best candidate in network
#     # @info "Final loss: $(best_los)"
# end

function mutation(spring_data, strength)
    return spring_data + (rand(length(spring_data)) .- 0.5) * strength
end

function get_parent_index(index)
    i = 1
    while rand() < 0.55 && i < length(index)
        i += 1
    end
    #println(i)
    return (index[i])
end

function cross_over(candidates, index, out_size)
    out = [zeros(length(candidates[1])) for i in 1:out_size]
    for i in 1:Int(ceil(out_size / 2))
        out1, out2 = single_point_crossover(
            candidates[get_parent_index(index)], candidates[get_parent_index(index)]
        )
        out[i * 2 - 1] = out1
        out[i * 2] = out2
    end
    return out
end

function single_point_crossover(can1, can2)
    l = length(can1)
    out1 = zeros(l)
    out2 = zeros(l)
    p = rand(2:(l - 1))
    out1[1:p] = can1[1:p]
    out2[1:p] = can2[1:p]
    out1[(p + 1):l] = can2[(p + 1):l]
    out2[(p + 1):l] = can1[(p + 1):l]
    return (out1, out2)
end

function train!(
    network::Network, epochs::Int, behaviours::Vector{T}, sim::Simulation, opt::Evolution
) where {T<:Behaviour}
    loss = calc_loss(network, sim, behaviours)
    spring_data = get_spring_constants_vec(network)

    for _ in 1:epochs
        # TODO: check if pop size ever changes; if no, no need to allocate new
        # vector every time
        losses = Vector{Float64}(undef, length(opt.candidates))
        for i in eachindex(opt.candidates)
            opt.candidates[i] = mutation(opt.candidates[i], opt.mutation_strength)
            set_spring_data!(network, opt.candidates[i])
            losses[i] = calc_loss(network, sim, behaviours)
        end

        index = sortperm(losses)
        if losses[index[1]] < loss
            spring_data = opt.candidates[index[1]]
            loss = losses[index[1]]
        else
            opt.mutation_strength *= 0.95
        end
        println(loss)

        next_gen = [zeros(length(spring_data)) for _ in 1:(opt.popsize)]
        next_gen[1:Int(floor(opt.popsize / 5))] = opt.candidates[index[1:Int(
            floor(opt.popsize / 5)
        )]]  # copying best 20%
        next_gen[(Int(floor(opt.popsize / 5)) + 1):length(next_gen)] = cross_over(
            opt.candidates, index, Int(ceil(opt.popsize * 0.8))
        )  # other 80% are crossover
        opt.candidates = copy(next_gen)
    end
    return set_spring_data!(network, spring_data)
end

#TODO
# function train2!(pop_size, network, trainer)
#     spring_data = get_spring_constants_vec(network)
#     loss = calc_loss(spring_data, network, trainer)

#     for i = 1:1
#         candidates = [mutation(spring_data, 0.0001) for i = 1:pop_size]
#         losses = [calc_loss(candidate, network, trainer) for candidate in candidates]
#         index = sortperm(losses)
#         if losses[index[1]] < loss
#             spring_data = candidates[index[1]]
#             loss = losses[index[1]]
#         end
#         println(loss)
#     end
#     set_spring_data!(n, spring_data)
# end
