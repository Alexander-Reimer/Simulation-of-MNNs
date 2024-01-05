module Compare
using MNN
using DataFrames, CSV # for writing to csv
using Dates # for timestamp
using UUIDs # for uuid1
using Random # for setting random seed

function create_df()
    return DataFrame(time=DateTime[], behaviours=Int64[], loss=Float64[], epochs=Int64[], min_angle=Float64[], uuid=UInt128[], rows=Int64[], columns=Int64[])
end

macro init_comp()
    ex = quote
        df = create_df()
        num_behaviours = 3
        epochs = 100
        min_angle = π / (num_behaviours + 1)
        rows=5
        columns=5
    end
    return esc(ex)
    # esc(Meta.parse(prog))
end

macro init_net()
    ex = quote
        id = uuid1()
        Random.seed!(id.value)
        net = MNN.Network(columns, rows)
    end
    return esc(ex)
end

function save_df(name, df)
    open("src/data/$(name)_$(now()).csv", write=true, create=true) do io
        CSV.write(io, df)
    end
end
macro save(name)
    return esc(:(save_df($name, df)))
end

macro trainer(opt)
    return esc(:(t = Trainer(MNN.create_behaviours(net, num_behaviours, min_angle), $opt(net)))) 
end

macro makeentry()
    return esc(:(push!(df, (now(), num_behaviours, MNN.calc_loss(net, t.behaviours), epochs, min_angle, id.value, rows, columns))))
end

function typename(t)
    return split(string(t), '.')[end]
end

function num_behaviours(opt_type)
    @init_comp
    max_num_behaviours = 5
    @sync for num_behaviours in 1:max_num_behaviours
        Threads.@spawn begin
            @init_net()
            @trainer(opt_type)
            train!(net, epochs, t)
            @makeentry
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
    @init_comp
    @sync for columns in 1:5
        for _ in 1:5
            Threads.@spawn begin
                @init_net
                @trainer(opt_type)
                train!(net, epochs, t)
                @makeentry
            end
        end
    end
    @save("$(typename(opt_type))NumColumns")
end

function num_rows(opt_type)
    @init_comp
    @sync for rows in 1:5
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

end