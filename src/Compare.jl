module Compare
using MNN
using DataFrames, CSV # for writing to csv
using Dates # for timestamp
using UUIDs # for uuid1
using Random # for setting random seed

function create_df()
    return DataFrame(time=DateTime[], uuid=UInt128[], epochs=Int64[], rows=Int64[], columns=Int64[], behaviours=Int64[], min_angle=Float64[], mag_goals=Float64[], mag_modifier=Float64[], sim_type=String[], sim_time=Float64[], loss=Float64[])
end

macro init_comp()
    ex = quote
        global df = create_df()
        filepath = "src/data/" * name * "_" * string(now()) * ".csv"
        open(filepath, write=true, create=true) do io
            CSV.write(io, df)
        end
        num_behaviours = 3
        epochs = 50
        sim_time = 100
        rows = 5
        columns = 5
        mag_goals = 1
        mag_modifier = 0.1
    end
    return esc(ex)
    # esc(Meta.parse(prog))
end

macro init_net()
    ex = quote
        id = uuid1()
        Random.seed!(id.value)
        min_angle = π / (num_behaviours + 1)
        net = MNN.Network(columns, rows)
    end
    return esc(ex)
end

function save_df(filepath, df)
    open(filepath, write=true) do io
        CSV.write(io, df)
        # flush(io)
    end
    # empty!(df)
end

macro save(x)
    # return esc(:(save_df($filepath, df)))
end

macro trainer(opt)
    ex = quote
        t = Trainer(
            MNN.create_behaviours(net, num_behaviours; min_angle=min_angle, m_goal=mag_goals, m_mod=mag_modifier),
            MNN.Diff(sim_time),
            $opt(net)
        )
    end
    return esc(ex)
end

macro makeentry()
    ex = quote
        push!(df, (
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
            MNN.calc_loss(net, t.simulation, t.behaviours)
        ))
        save_df(filepath, df)
    end
    return esc(ex)
end

macro train!()
    estep = 5
    ex = quote
        max_epochs = epochs
        for epochs in 0:$estep:max_epochs
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
    max_num_behaviours = 10
    @sync for num_behaviours in 1:max_num_behaviours, _ in 1:50
        Threads.@spawn begin
            @init_net()
            @trainer(opt_type)
            @train!
        end
    end
    @save("$(typename(opt_type))NumBehaviours")
end

function epochs(opt_type)
    @init_comp
    max_epochs = epochs
    step = 10
    # Threads.@threads 
    @sync for _ in 1:5
        Threads.@spawn begin
            @init_net()
            @trainer(opt_type)
            for epochs in 0:step:max_epochs
                epochs > 0 && train!(net, step, t)
                @makeentry
            end
        end
    end
    @save("$(typename(opt_type))Epochs")
end

function min_angle(opt_type)
    @init_comp
    @sync for angle in 0:0.1:π/3
        for _ in 1:5
            Threads.@spawn begin
                @init_net
                @trainer(opt_type)
                train!(net, epochs, t)
                @makeentry
            end
        end
    end
    @save("$(typename(opt_type))MinAngle")
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

function compare_pps()
    num_behaviours(MNN.PPS)
    epochs(MNN.PPS)
    min_angle(MNN.PPS)
    num_columns(MNN.PPS)
end

function compare_evolution()
    num_behaviours(MNN.Evolution)
    epochs(MNN.Evolution)
    min_angle(MNN.Evolution)
    num_columns(MNN.Evolution)
end

function load(path)
    return CSV.read(path, DataFrame, types=Dict(:uuid => UInt128))
end

end