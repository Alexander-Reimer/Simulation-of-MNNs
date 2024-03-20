module CompareResonance
using MNN
using DataFrames, CSV # for writing to csv
using Dates # for timestamp
using UUIDs # for uuid1
using Random # for setting random seed
using Arrow # for writing to Arrow

function create_df()
    df = DataFrame(;
        time=DateTime[],
        uuid=UUID[],
        epochs=Int64[],
        rows=Int64[],
        columns=Int64[],
        goal_num=Int64[],
        network=Network[],
        trainer=Trainer[],
        loss=Float64[],
    )
    timestamp = Dates.format(now(), "yyyy-mm-ddTHH-MM-SS")
    metadata!(df, "time", timestamp; style=:note)
    return df
end

function save_df(df)
    filepath =
        "src/data/ResonanceCurveOptimization/" *
        "ResonanceEpochs" *
        "_" *
        metadata(df)["time"] *
        ".arrow"
    return Arrow.write(filepath, df; maxdepth=7)
end

function make_entry(net, trainer, num_goals_resonance, loss, id)
    lock(lk)
    push!(
        df,
        (
            now(),
            id,
            trainer.optimization.epochs,
            net.rows,
            net.columns,
            num_goals_resonance,
            net,
            trainer,
            loss,
        ),
    )
    save_df(df)
    unlock(lk)
    return nothing
end

function main()
    global df = create_df()
    global lk = ReentrantLock()
    epochs = 1000
    network_number = 10
    number_goals = 3
    @sync for _ in 1:network_number
        Threads.@spawn begin
            id = uuid1()
            Random.seed!(id.value)
            net = Network(11, 4)
            b = MNN.Resonance(net, 3)
            t = Trainer(b, Diff(100), PPS())
            for _ in 1:(epochs / 20)
                loss = train!(net, 20, t)
                make_entry(net, t, number_goals, loss, id)
            end
        end
    end
end

function load(path::String)
    return DataFrame(Arrow.Table(path))
end

function load()
    files = filter(
        x -> length(x) > 15 && x[1:15] == "ResonanceEpochs",
        readdir("src/data/ResonanceCurveOptimization/"),
    )
    if isempty(files)
        return nothing
    end
    newest = files[1]
    newest_time = Dates.DateTime(newest[17:35], "yyyy-mm-ddTHH-MM-SS")
    if length(files) == 1
        return load("src/data/ResonanceCurveOptimization/" * newest)
    end

    for filename in files[2:end]
        time = Dates.DateTime(filename[17:35], "yyyy-mm-ddTHH-MM-SS")
        if time > newest_time
            newest = filename
            newest_time = time
        end
    end
    return load("src/data/ResonanceCurveOptimization/" * newest)
end

end
