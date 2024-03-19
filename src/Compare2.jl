module Compare
using MNN
using DataFrames, CSV # for writing to csv
using Dates # for timestamp
using UUIDs # for uuid1
using Random # for setting random seed

function extra_fields(opt::MNN.Evolution)

end

function create_df(t::Trainer)
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

end
