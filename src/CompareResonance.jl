module CompareResonance
using DataFrames, CSV # for writing to csv
using Dates # for timestamp
using UUIDs # for uuid1
using Random # for setting random seed
using Arrow # for writing to Arrow
using GLMakie # for plotting
using Revise
using MNN

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

function make_entry(net, trainer, num_goals_resonance, loss, id, epochs)
    lock(lk)
    push!(
        df,
        (now(), id, epochs, net.rows, net.columns, num_goals_resonance, net, trainer, loss),
    )
    save_df(df)
    unlock(lk)
    return nothing
end

function main()
    global df = create_df()
    global lk = ReentrantLock()
    epochs_pps = 1000
    # epochs_evolution = 200
    network_number = 10
    number_goals = 3
    try
        @sync for _ in 1:network_number
            Threads.@spawn begin
                id = uuid1()
                Random.seed!(id.value)
                net = Network(11, 4)
                b = MNN.Resonance(net, 3)
                t = Trainer(b, Diff(200), PPS())
                epochs = 0
                for _ in 1:(epochs_pps / 20)
                    loss = train!(net, 20, t)
                    epochs += 20
                    make_entry(net, t, number_goals, loss, id, epochs)
                end
                # t.optimization = Evolution(net)
                # for _ in 1:(epochs_evolution / 20)
                #     loss = train!(net, 20, t)
                #     epochs += 20
                #     make_entry(net, t, number_goals, loss, id, epochs)
                # end
            end
        end
    catch e
        println(e)
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

function get_resonance_curve(net::Network, b::Resonance, frqs)
    neurons = collect(keys(b.goals))
    amplitude = b.modifiers[1][2]
    amps = MNN.calculate_resonance_curve(net, frqs, amplitude, neurons[1])
    for n in neurons[2:end]
        amps .+= MNN.calculate_resonance_curve(net, frqs, amplitude, n)
    end
    return amps ./ length(neurons)
end

function get_resonance_curve(net::Network, trainer::Trainer, step=0.5)
    return get_resonance_curve(net, trainer.behaviours[1], 0.0:step:3.5)
end

function show_results(df)
    global amps_res = []
    global xs_res = []
    global ys_res = []
    max_epochs = maximum(df.epochs)
    df = filter(:epochs => ==(max_epochs), df)
    fig = Figure()
    ax = Axis(fig[1, 1])
    Threads.@threads for row in eachrow(df)
        net = row.network
        trainer = row.trainer
        amps = get_resonance_curve(net, trainer, 0.25)
        push!(amps_res, amps)
        lines!(ax, 0.0:0.25:3.5, amps)
        xs = Float64[]
        ys = Float64[]
        for b in trainer.behaviours
            push!(xs, collect(values(b.modifiers))[1][1])
            push!(ys, collect(values(b.goals))[1][1])
        end
        scatter!(xs, ys)
        push!(xs_res, xs)
        push!(ys_res, ys)
    end
    return fig
end

function fix(filepath)
    df = load(filepath)
    rm(filepath)
    Arrow.write(filepath, df; maxdepth=7)
    df = nothing
    return nothing
end

function fix()
    files = filter(
        x -> length(x) > 15 && x[1:15] == "ResonanceEpochs",
        readdir("src/data/ResonanceCurveOptimization/"),
    )
    for file in files
        fix("src/data/ResonanceCurveOptimization/" * file)
    end
end

end
