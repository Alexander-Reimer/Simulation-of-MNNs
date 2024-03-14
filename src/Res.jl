function nlys_ressonance()
    n = Network(11,5)
    sim = Diff(1000)
    sim.modifier = resonate!
    frequencies = [ 0.01*1.065^i for i = 1:100]

    positions = []
    for f in frequencies
        println(f)
        sim.modifier = (network, acc, t) -> resonate!(network,acc,f,t)
        reset!(n)
        r = simulate!(n, sim)
        posx = []
        for i = 1:n.row_counts[n.columns]
            push!(posx, r[get_neuron_index(n, n.columns, i)*2-1 ,:])
        end
        push!(positions, posx)
    end


    for i = 1:length(positions)
        for j = 1:length(positions[1])
            positions[i][j] = maximum(positions[i][j][500:1000])-minimum(positions[i][j][500:1000])
        end
    end
    return positions
end
