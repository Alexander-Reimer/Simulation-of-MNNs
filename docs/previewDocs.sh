# using https://github.com/tlienart/LiveServer.jl/#serve-docs
julia --project=docs -ie 'using Revise, MNN, LiveServer; servedocs()'