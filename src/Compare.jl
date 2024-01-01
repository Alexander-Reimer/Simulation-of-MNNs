module Compare
using MNN
using DataFrames, CSV # for writing to csv
using Dates # for timestamp

function pps_num_behaviours()
    df = DataFrame(time=DateTime[], behaviours=Int64[], loss=Float64[], epochs=Int64[])
    net = MNN.Network(5, 5)
    max_nums = 6
    epochs = 300
    Threads.@threads for i in 1:max_nums*15
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
    max_epochs = 300
    step = 10
    num_b = 3
    Threads.@threads for i in 1:16
        net = MNN.Network(5, 5)
        t = Trainer(MNN.create_behaviours(net, num_b, π / (num_b + 1)), MNN.PPS())
        for e in 0:step:max_epochs
            e > 0 && train!(net, e, t)
            push!(df, (now(), num_b, MNN.calc_loss(net, t.behaviours), e))
        end
    end
    open("src/data/PPSEpochs_$(now()).csv", write=true, create=true) do io
        CSV.write(io, df)
    end
end

function compare_pps()

end

# pps_num_behaviours()
end