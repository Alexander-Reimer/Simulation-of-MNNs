module Compare
using MNN
using DataFrames, CSV # for writing to csv
using Dates # for timestamp

function pps_num_behaviours()
    df = DataFrame(time=DateTime[], behaviours=Int64[], loss=Float64[], epochs=Int64[])
    max_nums = 5
    epochs = 150
    Threads.@threads for i in 1:max_nums*4
        net = MNN.Network(5, 5)
        num = i % max_nums + 1
        println("Number of behaviours: ", num)
        t = Trainer(MNN.create_behaviours(net, num, π / (num + 1)), MNN.PPS())
        train!(net, epochs, t)
        push!(df, (now(), num, MNN.calc_loss(net, t.behaviours), epochs))
    end
    open("src/data/PPSNumBehaviours_$(now()).csv", write=true, create=true) do io
        CSV.write(io, df)
    end
end

function pps_epochs()
    df = DataFrame(time=DateTime[], behaviours=Int64[], loss=Float64[], epochs=Int64[])
    max_epochs = 150
    step = 10
    num_b = 3
    Threads.@threads for i in 1:8
        net = MNN.Network(5, 5)
        t = Trainer(MNN.create_behaviours(net, num_b, π / (num_b + 1)), MNN.PPS())
        for e in 0:step:max_epochs
            e > 0 && train!(net, step, t)
            push!(df, (now(), num_b, MNN.calc_loss(net, t.behaviours), e))
        end
    end
    open("src/data/PPSEpochs_$(now()).csv", write=true, create=true) do io
        CSV.write(io, df)
    end
end

function pps_diff()
    df = DataFrame(time=DateTime[], behaviours=Int64[], loss=Float64[], epochs=Int64[], min_angle=Float64[])
    max_epochs = 150
    num_b = 3
    # Threads.@threads
    tasks = []
    for angle in 0:0.1:π/3
        for _ in 1:5
            t = Threads.@spawn begin
                net = MNN.Network(5, 5)
                t = Trainer(MNN.create_behaviours(net, num_b, angle), MNN.PPS())
                train!(net, max_epochs, t)
                push!(df, (now(), num_b, MNN.calc_loss(net, t.behaviours), max_epochs, angle))
            end
            push!(tasks, t)
        end
    end
    wait.(tasks)
    open("src/data/PPSDiff_$(now()).csv", write=true, create=true) do io
        CSV.write(io, df)
    end
end

function ()
    
end

function compare_pps()
    pps_num_behaviours()
    pps_epochs()
    pps_diff()
end

function compare_evolution()
    
end

# pps_num_behaviours()
end