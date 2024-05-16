mutable struct Resonance <: Behaviour
    goals::Dict{Int,Float64}
    modifiers::Dict{Int,Vector{Float64}}
end

function Resonance(net::Network, num_goals; amplitude=0.2, min_frq=0, max_frq=3)
    b = Vector{Resonance}()
    for i in 1:num_goals
        r = rand() / 5
        f = if num_goals == 1
            min_frq
        else
            min_frq + ((max_frq - min_frq) * (i - 1) / (num_goals - 1))
        end
        goals = Dict()
        for row in 1:net.row_counts[end]
            goals[get_neuron_index(net, net.columns, row)] = r
        end
        modifiers = Dict()
        for n in 1:net.row_counts[1]
            modifiers[n] = [f, amplitude]
        end
        push!(b, Resonance(goals, modifiers))
    end
    return b
end

function resonate!(network::Network, acc, modifiers, t)
    for (n, m) in modifiers
        f, a = m[1], m[2] # frequency and amplitude
        acc[1, n] += sin(t * f * 2π) * a # only affects x component
    end
end

function calc_amplitude(neuron, integ, portion)
    max_x = maximum(integ.sol.u[(end - portion):end]) do curr_solution
        positions = curr_solution.x[2]
        return positions[1, neuron] # only x component
    end
    min_x = minimum(integ.sol.u[(end - portion):end]) do curr_solution
        positions = curr_solution.x[2]
        return positions[1, neuron] # only x component
    end
    return (max_x - min_x) / 2
end

#TODO: implement for Euler
# function calc_loss(network::Network, sim::SecondOrderDiff, behaviour::Resonance)
#     reset!(network)
#     integ = simulate!(network, sim, behaviour)
#     # portion of the simulation to consider (0.2 --> last 20% of the simulation)
#     portion = round(Int, integ.t / 2)
#     l = 0.0
#     for (n, target_a) in behaviour.goals # neuron and target amplitude
#         max_x = maximum(integ.sol.u[(end - portion):end]) do curr_solution
#             positions = curr_solution.x[2]
#             return positions[1, n] # only x component
#         end
#         min_x = minimum(integ.sol.u[(end - portion):end]) do curr_solution
#             positions = curr_solution.x[2]
#             return positions[1, n] # only x component
#         end
#         actual_a = (max_x - min_x) / 2 # actual amplitude
#         l += (target_a - actual_a)^2
#     end
#     return l / length(behaviour.goals)
# end

function resonate!(network::Network, acc, behaviours::Vector, t)
    for behaviour in behaviours
        resonate!(network, acc, behaviour.modifiers, t)
    end
end

function find_closest_index(arr::AbstractArray, x::Number)
    closest = abs(arr[1] - x)
    i = 2
    while i <= length(arr)
        diff = abs(arr[i] - x)
        if (diff > closest)
            i -= 1
            break
        else
            closest = diff
            i += 1
        end
    end
    return i
end

# note: te modifier frqeuencies of each Resonance behaviour need to be the same across input neurons
function calc_loss(network::Network, sim::SecondOrderDiff, behaviours::Vector{T}) where {T<:Behaviour}
    @info "Correct func!"
    reset!(network)
    sim.modifier = (network, acc, t) -> MNN.resonate!(network, acc, behaviours, t)

    integ = simulate!(network, sim)
    l = 0.0 # initialize loss
    n_loss = 0
    fs = 100 # sampling frequency
    portion = 0.5 # portion of simulaiton to consider (here last 50%)
    ts = (integ.t * portion):(1 / fs):(integ.t)
    positions = Vector{Float64}(undef, length(ts))
    component = 1 # x component
    frequencies = 0
    amplitudes = 0
    avg_amplitudes = zeros(length(positions) ÷ 2 + 1)
    for b in behaviours
        frequency = (collect(values(b.modifiers)))[1][1]
        for (neuron, target_amplitude) in b.goals
            map!(positions, ts) do t
                # .x[2] accesses positions, .x[1] would be velocities
                return integ.sol(t).x[2][1, neuron]
            end
            frequencies = FFTW.rfftfreq(length(positions), fs)
            # scale with length / 2 beacuse rfft --> symmetric
            amplitudes = abs.(FFTW.rfft(positions) ./ (length(positions) / 2))
            amplitude_index = find_closest_index(frequencies, frequency)
            l += (amplitudes[amplitude_index] - target_amplitude)^2
            avg_amplitudes .+= amplitudes
            n_loss += 1
        end
    end
    return l / n_loss
    # return (frequencies, avg_amplitudes ./ n_loss)
end

function calc_res_curve(
    network::Network, sim::SecondOrderDiff, behaviours::Vector{T}
) where {T<:Behaviour}
    @info "Correct func!"
    reset!(network)
    sim.modifier = (network, acc, t) -> MNN.resonate!(network, acc, behaviours, t)

    integ = simulate!(network, sim)
    l = 0.0 # initialize loss
    n_loss = 0
    fs = 100 # sampling frequency
    portion = 0.5 # portion of simulaiton to consider (here last 50%)
    ts = (integ.t * portion):(1 / fs):(integ.t)
    positions = Vector{Float64}(undef, length(ts))
    component = 1 # x component
    frequencies = 0
    amplitudes = 0
    avg_amplitudes = zeros(length(positions) ÷ 2 + 1)
    for b in behaviours
        frequency = (collect(values(b.modifiers)))[1][1]
        for (neuron, target_amplitude) in b.goals
            map!(positions, ts) do t
                # .x[2] accesses positions, .x[1] would be velocities
                return integ.sol(t).x[2][1, neuron]
            end
            frequencies = FFTW.rfftfreq(length(positions), fs)
            # scale with length / 2 beacuse rfft --> symmetric
            amplitudes = abs.(FFTW.rfft(positions) ./ (length(positions) / 2))
            amplitude_index = find_closest_index(frequencies, frequency)
            l += (amplitudes[amplitude_index] - target_amplitude)^2
            avg_amplitudes .+= amplitudes
            n_loss += 1
        end
    end
    # return l / n_loss
    return (frequencies, avg_amplitudes ./ n_loss)
end

#=function calc_res(n, sim)
    frequencies = [ 0.01*1.065^i for i = 1:100]
    data = []

    for f in frequencies
        reset!(n)
        sim.modifier = (n, acc, t) -> resonate!(n, acc, Dict(1=>[f,0.1], 2=>[f,0.1], 3=>[f,0.1]), t)
        integ = simulate!(n, sim)
        foo = round(Int, integ.t / 5)

        ampli = []

        for neuro in 1:n.row_counts[n.rows]
            max_x = maximum(integ.sol.u[end-foo:end]) do curr_solution
                positions = curr_solution.x[2]
                return positions[1, get_neuron_index(n,n.columns,neuro)] # only x component
            end
            min_x = minimum(integ.sol.u[(end - foo + 1):end]) do curr_solution
                positions = curr_solution.x[2]
                return positions[1, get_neuron_index(n,n.columns,neuro)] # only x component
            end
            push!(ampli, (max_x - min_x) / 2)
        end
        println(mean(ampli))
        push!(data, mean(ampli))
    end

    return frequencies, data
end=#

function simulate!(
    network::Network,
    sim::Simulation,
    behaviour::Resonance;
    vis::Union{Visualizer,Nothing}=nothing,
)
    sim.modifier = (network, acc, t) -> resonate!(network, acc, behaviour.modifiers, t)
    return simulate!(network, sim; vis=vis)
end

function calculate_resonance_curve(network::Network, frequencies, amplitude, neuron)
    amplitudes = Vector{Float64}(undef, length(frequencies))
    sim = SecondOrderDiff(300, (network, acc, t) -> nothing)
    for i in eachindex(frequencies)
        f = frequencies[i]
        mods = Dict()
        for n in 1:network.row_counts[1]
            mods[n] = [f, amplitude]
        end
        sim.modifier = (network, acc, t) -> resonate!(network, acc, mods, t)
        reset!(network)
        integ = simulate!(network, sim)
        amplitudes[i] = calc_amplitude(neuron, integ, round(Int, integ.t / 2))
    end
    return amplitudes
end

function calc_res(n, sim)
    frequencies = [0.01 * 1.065^i for i in 1:100]
    data = []

    for f in frequencies
        reset!(n)
        sim.modifier =
            (n, acc, t) ->
                resonate!(n, acc, Dict(1 => [f, 0.1], 2 => [f, 0.1], 3 => [f, 0.1]), t)
        integ = simulate!(n, sim)
        foo = round(Int, integ.t / 5)

        ampli = []

        for neuro in 1:n.row_counts[n.rows]
            max_x = maximum(integ.sol.u[(end - foo):end]) do curr_solution
                positions = curr_solution.x[2]
                return positions[1, get_neuron_index(n, n.columns, neuro)] # only x component
            end
            min_x = minimum(integ.sol.u[(end - foo + 1):end]) do curr_solution
                positions = curr_solution.x[2]
                return positions[1, get_neuron_index(n, n.columns, neuro)] # only x component
            end
            push!(ampli, (max_x - min_x) / 2)
        end
        println(mean(ampli))
        push!(data, mean(ampli))
    end
    return frequencies, data
end
