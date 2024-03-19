mutable struct Resonance <: Behaviour
    goals::Dict{Int,Number}
    modifiers::Dict{Int,Vector{Number}}
end

function Resonance(num_goals)
    b = Vector{Resonance}()
    for i in 1:num_goals
        r = rand() / 4
        f = 0.4 + (1.8 * (i - 1) / (num_goals - 1))
        push!(
            b,
            Resonance(
                Dict(36 => r, 37 => r, 38 => r),
                Dict(1 => [f, 0.1], 2 => [f, 0.1], 3 => [f, 0.1]),
            ),
        )
    end
    return b
end

function resonate!(network::Network, acc, modifiers, t)
    for (n, m) in modifiers
        f, a = m[1], m[2] # frequency and amplitude
        acc[1, n] += sin(t * f / (2pi)) * a # only affects x component
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
function calc_loss(network::Network, sim::Diff, behaviour::Resonance)
    reset!(network)
    integ = simulate!(network, sim, behaviour)
    # portion of the simulation to consider (0.2 --> last 20% of the simulation)
    portion = round(Int, integ.t / 2)
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
    sim = Diff(500, (network, acc, t) -> nothing)
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
