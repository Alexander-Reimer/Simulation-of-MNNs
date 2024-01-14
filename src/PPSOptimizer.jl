mutable struct PPS <: Optimization
    initialized::Bool
    init::Float64
    increment::Float64
    # base_loss::Float64
    selected::Set
    # new_spring_data::Dict{Tuple{Int64,Int64},Spring}
    sim_epochs::Int
end

PPS() = PPS(false, 1.15, 1.0, Set(), 250)
PPS(net) = PPS()

function select_spring(spring_data, selected::Set)
    # spring_data = get_spring_constants(network)
    if length(spring_data) == length(selected)
        return nothing
    end
    local i
    while true
        i = rand(1:length(spring_data))
        if !(i in selected)
            push!(selected, i)
            break
        end
    end
    return i
end

function pps_init!(network::Network, opt::PPS)
    spring_data = get_spring_constants(network)
    for (k, _) in spring_data
        spring_data[k].spring_constant = opt.init
    end
end

function train!(
    network::Network, epochs::Int, behaviours::Vector{Behaviour}, sim::Simulation, opt::PPS
)
    if !opt.initialized
        pps_init!(network, opt)
        opt.initialized = true
    end

    base_loss = calc_loss(network, sim, behaviours)
    isnan(base_loss) && (@warn "Abandoning training because loss is NaN!"; return nothing)

    spring_data = deepcopy(get_spring_constants_vec(network))
    new_spring_data = deepcopy(spring_data)
    @info "Base loss: $base_loss"
    while true
        i = select_spring(new_spring_data, opt.selected)
        if i === nothing
            opt.increment > 0 ? opt.increment *= -1 : opt.increment *= -0.9
            opt.selected = Set()
            continue
        end
        spring = new_spring_data[i]
        spring = opt.increment + spring
        new_spring_data[i] = spring

        set_spring_data!(network, new_spring_data)
        loss = calc_loss(network, sim, behaviours)
        isnan(loss) && (@warn "Abandoning training because loss is NaN!"; break)

        if loss < base_loss
            # new_spring_data = deepcopy(spring_data)
            # while true
            #     increment *= 0.9
            #     spring = increment + spring
            #     loss = calc_loss(network, spring_data, trainer.behaviours)

            # end
            spring_data = deepcopy(new_spring_data)
            base_loss = loss
        else
            new_spring_data = deepcopy(spring_data)
        end
        epochs -= 1
        epochs % 20 == 0 &&
            @info "Epochs left: $epochs, base loss: $base_loss, loss: $loss, increment: $(opt.increment)"
        epochs == 0 && break
    end
    set_spring_data!(network, spring_data)
    return nothing
end
