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
        epochs = 200
        sim_time = 500
        rows = 5
        columns = 4
        mag_goals = 0.1
        mag_modifier = 0.1
    end
    return esc(ex)
end

macro init_net()
    ex = quote
        id = uuid1()
        Random.seed!(id.value)
        # found through experimentation; this ensures that create_behaviours isn't caught in
        # infinite loop trying to get suffucuently distant vectors
        min_angle = π / 3
        # min_angle = num_behaviours <= 1 ? 0 : π / (num_behaviours + 0)
        net = MNN.TestNetwork(columns, rows)
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
            MNN.create_deformation_behaviours(
                net,
                num_behaviours;
                min_angle=min_angle,
                m_goal=mag_goals,
                m_mod=mag_modifier,
            ),
            MNN.SecondOrderDiff(sim_time),
            $opt(net),
        )
        if $opt <: MNN.Evolution
            t.optimization.mutation_strength = 0.01
            t.optimization.popsize = 10
        else
            allowmissing!(df, [:mutation_strength, :popsize])
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
    @sync for num_behaviours in 1:max_num_behaviours, _ in 1:4
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

function resonance_test(net=nothing, t=nothing)
    if net === nothing
        net = MNN.Network(11, 4)
    end
    if t === nothing
        # r = MNN.Resonance(Dict(37 => 0.0), Dict(1 => [2, 0.1], 2 => [2, 0.1], 3 => [2, 0.1]))
        r1 = MNN.Resonance(
            Dict(37 => 0.4), Dict(1 => [1.8, 0.1], 2 => [1.8, 0.1], 3 => [1.8, 0.1])
        )
        r2 = MNN.Resonance(
            Dict(37 => 0.4), Dict(1 => [0.4, 0.1], 2 => [0.4, 0.1], 3 => [0.4, 0.1])
        )
        r3 = MNN.Resonance(
            Dict(37 => 0.1), Dict(1 => [1, 0.1], 2 => [1, 0.1], 3 => [1, 0.1])
        )
        t = MNN.Trainer([r1, r2, r3], Diff(150), PPS())
    end
    freqs = 0.0:0.2:2.2
    amps = []
    push!(amps, MNN.calculate_resonance_curve(net, freqs, 0.1, 37))
    for i in 1:3
        MNN.train!(net, 100, t)
        push!(amps, MNN.calculate_resonance_curve(net, freqs, 0.1, 37))
    end
    return net, amps, t, freqs
end

end
