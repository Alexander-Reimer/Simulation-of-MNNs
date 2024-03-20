# ==========================
#           NETWORK
# ==========================

struct NetworkSerialized
    columns::Int
    rows::Int # how many rows fixed columns have
    spring_constants::Vector{Float64}
end

function serialize(network::Network)
    return NetworkSerialized(
        network.columns, network.rows, get_spring_constants_vec(network)
    )
end

function deserialize(network_serialized::NetworkSerialized)
    network = Network(network_serialized.columns, network_serialized.rows)
    set_spring_data!(network, network_serialized.spring_constants)
    return network
end

ArrowTypes.arrowname(::Type{Network}) = :NetworkSerialized
ArrowTypes.ArrowType(::Type{Network}) = NetworkSerialized
ArrowTypes.ArrowKind(::Type{Network}) = ArrowTypes.StructKind()

ArrowTypes.toarrow(network::Network) = serialize(network)

ArrowTypes.JuliaType(::Val{:NetworkSerialized}) = Network
function ArrowTypes.fromarrow(::Type{Network}, columns, rows, spring_constants)
    network_serialized = NetworkSerialized(columns, rows, spring_constants)
    return deserialize(network_serialized)
end

# ==========================
#           TRAINER
# ==========================

ArrowTypes.arrowname(::Type{Trainer}) = :Trainer
ArrowTypes.ArrowKind(::Type{Trainer}) = ArrowTypes.StructKind()
ArrowTypes.JuliaType(::Val{:Trainer}) = Trainer

function Base.convert(
    ::Type{Behaviour},
    x::NamedTuple{
        (:goals, :modifiers),Tuple{Dict{Int64,Float64},Dict{Int64,Vector{Float64}}}
    },
)
    return Resonance(x.goals, x.modifiers)
end

function Base.convert(
    ::Type{Simulation},
    x::NamedTuple{(:time, :modifier),Tuple{Int64,NamedTuple{(),Tuple{}}}},
)
    return Diff(x.time)
end

function Base.convert(::Type{Simulation}, x::NamedTuple)
    return Diff(x.time)
end

function Base.convert(
    ::Type{Simulation},
    x::NamedTuple{
        (:time, :modifier),
        Tuple{
            Int64,
            NamedTuple{
                (:behaviour,),
                Tuple{
                    NamedTuple{
                        (:goals, :modifiers),
                        Tuple{Dict{Int64,Float64},Dict{Int64,Vector{Float64}}},
                    },
                },
            },
        },
    },
)
    return Diff(x.time)
end

function Base.convert(
    ::Type{Optimization},
    x::NamedTuple{
        (:initialized, :init, :increment, :selected, :epochs),
        Tuple{Bool,Float64,Float64,Set{Int64},Int64},
    },
)
    return PPS(x.initialized, x.init, x.increment, x.selected, x.epochs)
end

function Base.convert(
    ::Type{Optimization},
    x::NamedTuple{
        (:initialized, :init, :increment, :selected, :epochs),
        Tuple{Bool,Float64,Float64,Set{Missing},Int64},
    },
)
    return PPS(x.initialized, x.init, x.increment, x.selected, x.epochs)
end

function Base.convert(
    ::Type{Optimization},
    x::NamedTuple{
        (:parallel, :mutation_strength, :popsize, :candidates, :epochs),
        Tuple{Bool,Float64,Int64,Vector{Vector{Float64}},Int64},
    },
)
    return Evolution(x.parallel, x.mutation_strength, x.popsize, x.candidates, x.epochs)
end

ArrowTypes.arrowname(::Type{Diff}) = :Diff
ArrowTypes.ArrowKind(::Type{Diff}) = ArrowTypes.StructKind()
ArrowTypes.JuliaType(::Val{:Diff}) = Diff
