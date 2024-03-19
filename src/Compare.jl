module Compare
using MNN
using DataFrames, CSV # for writing to csv
using Dates # for timestamp
using UUIDs # for uuid1
using Random # for setting random seed

function create_df()
    return DataFrame(;
        time=DateTime[],
        uuid=UInt128[],
        epochs=Int64[],
        rows=Int64[],
        columns=Int64[],
        behaviours=Int64[],
        min_angle=Float64[],
        mag_goals=Float64[],
        mag_modifier=Float64[],
        sim_type=String[],
        sim_time=Float64[],
        mutation_strength=Float64[],
        popsize=Int64[],
        behaviour_type=String[],
        num_goals_resonance=Int64[],
        loss=Float64[],
    )
end

macro init_comp()
    ex = quote
        global df = create_df()
        global lk = ReentrantLock()
        timestamp = Dates.format(now(), "yyyy-mm-ddTHH-MM-SS")
        filepath = "src/data/" * name * "_" * timestamp * ".csv"
        open(filepath; write=true, create=true) do io
            CSV.write(io, df)
        end
        num_behaviours = 1
        epochs = 1000
        sim_time = 1000
        rows = 5
        columns = 4
        mag_goals = 0.1
        mag_modifier = 0.1
        num_goals_resonance = 3
    end
    return esc(ex)
end

macro init_net()
    ex = quote
        id = uuid1()
        Random.seed!(id.value)
        # found through experimentation; this ensures that create_behaviours isn't caught in
        # infinite loop trying to get suffucuently distant vectors
        min_angle = num_behaviours <= 1 ? 0 : π / (num_behaviours + 0)
        net = MNN.Network(columns, rows)
    end
    return esc(ex)
end

function save_df(filepath, df)
    open(filepath; append=true, truncate=false) do io
        CSV.write(io, df; append=true)
    end
    return empty!(df)
end

macro save(x)
    # return esc(:(save_df($filepath, df)))
end

macro trainer(opt)
    ex = quote
        t = Trainer(
            MNN.create_behaviours(
                net,
                num_behaviours;
                min_angle=min_angle,
                m_goal=mag_goals,
                m_mod=mag_modifier,
            ),
            MNN.Diff(sim_time),
            $opt(net),
        )
        if $opt <: MNN.Evolution
            t.optimization.mutation_strength = 0.01
            t.optimization.popsize = 10
        end
    end
    return esc(ex)
end

macro makeentry()
    ex = quote
        lock(lk)
        if t.optimization isa MNN.Evolution
            push!(
                df,
                (
                    now(),
                    id.value,
                    epochs,
                    rows,
                    columns,
                    num_behaviours,
                    min_angle,
                    mag_goals,
                    mag_modifier,
                    typename(t.simulation),
                    t.simulation.time,
                    t.optimization.mutation_strength,
                    # missing, #TODO: saving mutation strength
                    t.optimization.popsize,
                    MNN.calc_loss(net, t.simulation, t.behaviours),
                ),
            )
        else
            push!(
                df,
                (
                    now(),
                    id.value,
                    epochs,
                    rows,
                    columns,
                    num_behaviours,
                    min_angle,
                    mag_goals,
                    mag_modifier,
                    typename(t.simulation),
                    t.simulation.time,
                    missing,
                    missing,
                    MNN.calc_loss(net, t.simulation, t.behaviours),
                ),
            )
        end
        save_df(filepath, df)
        unlock(lk)
    end
    return esc(ex)
end

macro train!()
    estep = 5
    ex = quote
        max_epochs = epochs
        for epochs in 0:($estep):max_epochs
            epochs > 0 && train!(net, $estep, t)
            @makeentry
        end
        # epochs = max_epochs
    end
    return esc(ex)
end

typename(t::DataType) = split(string(t), '.')[end]
typename(t) = typename(typeof(t))

function num_behaviours(opt_type)
    name = "$(typename(opt_type))NumBehaviours"
    @init_comp
    max_num_behaviours = 4
    @sync for num_behaviours in 1:max_num_behaviours, _ in 1:3
        Threads.@spawn begin
            @init_net()
            @info "Created network!"
            @trainer(opt_type)
            @info "Created trainer!"
            @train!
        end
    end
    @save("$(typename(opt_type))NumBehaviours")
end

function epochs(opt_type)
    name = "$(typename(opt_type))Epochs"
    @init_comp
    @sync for _ in 1:10
        Threads.@spawn begin
            @init_net
            @trainer(opt_type)
            @train!
        end
    end
end

function min_angle(opt_type)
    name = "$(typename(opt_type))MinAngle"
    @init_comp
    @sync for angle in 0:0.1:(2π / (5 + 3))
        for _ in 1:3
            Threads.@spawn begin
                @init_net
                min_angle = $angle
                @trainer(opt_type)
                @train!
            end
        end
    end
end

function num_columns(opt_type)
    name = "$(typename(opt_type))NumColumns"
    @init_comp
    # 2 columns is min; gets stuck in infinite loop otherwise
    @sync for columns in 2:3, _ in 1:1
        Threads.@spawn begin
            epochs = $epochs
            @init_net
            @trainer(opt_type)
            @train!
            # train!(net, epochs, t)
            # @makeentry
        end
    end
    # @save("$(typename(opt_type))NumColumns")
end

function num_rows(opt_type)
    @init_comp
    @sync for rows in 1:8
        for _ in 1:5
            Threads.@spawn begin
                @init_net
                @trainer(opt_type)
                train!(net, epochs, t)
                @makeentry
            end
        end
    end
    @save("$(typename(opt_type))NumRows")
end

function num_rows_columns(opt_type)
    name = "$(typename(opt_type))NumRowsColumns"
    @init_comp
    @sync for rows in 3:10, columns in 2:10, _ in 1:4
        Threads.@spawn begin
            epochs = $epochs
            @init_net
            @trainer(opt_type)
            @train!
        end
    end
end

function mag_modifier_goal(opt_type)
    name = "$(typename(opt_type))MagModifierGoal"
    @init_comp
    for _ in 1:2
        for i in 1:20
            # mag_goals = (0.05 * 2^(6 - 1)) - 0.05 * 2^(i - 1)
            # mag_goals = 0.05 * 2^(i - 1)
            mag_goals = 0.1 * i
            @sync for j in 1:20
                # mag_modifier = (0.05 * 2^(6 - 1)) - 0.05 * 2^(j - 1)
                # mag_modifier = 0.05 * 2^(j - 1)
                mag_modifier = 0.1 * j
                Threads.@spawn begin
                    local mag_goals = $mag_goals
                    local mag_modifier = $mag_modifier
                    @init_net
                    @trainer(opt_type)
                    @train!
                end
            end
        end
    end
end

function mutation_strength(opt_type)
    name = "$(typename(opt_type))MutationStrength"
    @init_comp
    @sync for i in 1:10, _ in 1:2
        strength = 0.0001 * i
        Threads.@spawn begin
            @init_net
            @trainer(opt_type)
            t.optimization.mutation_strength = $strength
            # t.optimization.mutation_strength = strength
            @train!
        end
    end
end

function compare_pps()
    num_behaviours(MNN.PPS)
    epochs(MNN.PPS)
    min_angle(MNN.PPS)
    return num_columns(MNN.PPS)
end

function compare_evolution()
    num_behaviours(MNN.Evolution)
    epochs(MNN.Evolution)
    min_angle(MNN.Evolution)
    return num_columns(MNN.Evolution)
end

function load(path)
    return CSV.read(path, DataFrame; types=Dict(:uuid => UInt128))
end

function get_start_mutation_strengths(df_e, df)
    result = Float64[]
    df = filter(:epochs => x -> x == 0, df)
    for row in eachrow(df_e)
        r = df[df.uuid .== row[:uuid], :]
        append!(result, r[!, :mutation_strength])
    end
    return result
end

function resonance_test(net=nothing, t=nothing; start_epochs = 0)
    df = DataFrame(;
        time=DateTime[],
        epochs=Int64[],
        amps=Vector{Float64}[],
        net=Network[],
        # t=Trainer[],
        # freqs=Vector{Float64}[],
    )
    if net === nothing
        net = MNN.Network(11, 4)
    else
        net = deepcopy(net)
    end
    foo = net.neuron_count
    amp = 0.2
    if t === nothing
        # r = MNN.Resonance(Dict(37 => 0.0), Dict(1 => [2, 0.1], 2 => [2, 0.1], 3 => [2, 0.1]))
        bar = 1.2
        r1 = MNN.Resonance(
            Dict(foo - 2 => 0.3, foo - 1 => 0.3, foo => 0.3),
            Dict(1 => [bar, amp], 2 => [bar, amp], 3 => [bar, amp]),
        )
        bar = 2.0
        r2 = MNN.Resonance(
            Dict(foo - 2 => 0.2, foo - 1 => 0.2, foo => 0.2),
            Dict(1 => [bar, amp], 2 => [bar, amp], 3 => [bar, amp]),
        )
        bar = 2.8
        r3 = MNN.Resonance(
            Dict(foo - 2 => 0.3, foo - 1 => 0.3, foo => 0.3),
            Dict(1 => [bar, amp], 2 => [bar, amp], 3 => [bar, amp]),
        )
        t = MNN.Trainer(
            [r1, r2, r3],
            Diff(200),
            # PPS(),
            Evolution(net)
        )
        # t.optimization.initialized = true
        t.optimization.mutation_strength = 0.1
        t.optimization.popsize = 10
    else
        t = deepcopy(t)
    end
    freqs = 0.0:0.4:3.2
    epochs = start_epochs
    amps =
        (
            MNN.calculate_resonance_curve(net, freqs, amp, foo - 2) +
            MNN.calculate_resonance_curve(net, freqs, amp, foo - 1) +
            MNN.calculate_resonance_curve(net, freqs, amp, foo)
        ) ./ 3.0
    push!(df, (now(), epochs, amps, deepcopy(net)))
    for i in 1:1
        MNN.train!(net, 50, t)
        epochs += 50
        amps =
            (
                MNN.calculate_resonance_curve(net, freqs, amp, foo - 2) +
                MNN.calculate_resonance_curve(net, freqs, amp, foo - 1) +
                MNN.calculate_resonance_curve(net, freqs, amp, foo)
            ) ./ 3.0
        push!(df, (now(), epochs, amps, deepcopy(net)))
        # push!(amps, MNN.calculate_resonance_curve(net, freqs, 0.1, 37))
    end
    return df, deepcopy(t), freqs
end

function resonance_test2()
    nets = []
    amps = []
    ts = []
    freqss = []
    Threads.@threads for i in 1:1
        b = MNN.Resonance(2)
        net, amp, t, freqs = resonance_test(nothing, MNN.Trainer(b, Diff(500), PPS()))
        push!(nets, net)
        push!(amps, amp)
        push!(ts, t)
        push!(freqss, freqs)
    end
    return nets, amps, ts, freqss
end

function test_influence_frequency()
    nets = []
    amps = []
    ts = []
    freqss = []
    Threads.@threads for f in 0.1:0.1:0.6
        net = Network(11, 4)
        b = MNN.Resonance(
            Dict(36 => 0.2, 37 => 0.2, 38 => 0.2),
            Dict(1 => [f, 0.1], 2 => [f, 0.1], 3 => [f, 0.1]),
        )
        net, amp, t, freqs = resonance_test(nothing, MNN.Trainer([b], Diff(500), PPS()))
        push!(nets, net)
        push!(amps, amp)
        push!(ts, t)
        push!(freqss, freqs)
    end
    return nets, amps, ts, freqss
end

function get_diffs(y)
    amps = y[2]
    for i in eachindex(amps)
        f = 0.2 * i
        println("Frequency: $f")
        println("Amplitude: $(amps[i][end][i+1])")
    end
end

end
