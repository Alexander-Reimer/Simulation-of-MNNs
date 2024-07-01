function calculate_deltas(network, behaiviour)
    deltas = zeros(Float64, size(network.positions)[1], size(network.positions)[2])
    for goal in behaiviour.goals
        for (neuron_i, goal_pos) in behaiviour.goals
            neuron = get_neuron(network, neuron_i)
            if behaiviour.relative
                start_pos = network.start_positions[neuron_i]
                goal_pos += start_pos
            end
            deltas[:, neuron_i] = (goal_pos - network.positions[:, neuron_i])
        end
    end

    for col in ((network.columns) - 1):-1:1
        for row in 1:network.row_counts[col]
            i = get_neuron_index(network, col, row)
            if network.graph[i].movable
                neuros = neighbors(network.graph, i)
                for neuro in neuros
                    if i < neuro
                        deltas[:, i] += deltas[:, neuro] / 3
                    end
                end
            end
        end
    end

    return deltas
end

function calculate_vec_align(v1, v2)
    return clamp(v1 ⋅ v2 / (norm(v1) * norm(v2)), -1, 1)
end

function update_spring!(n, spring, deltas)
    if !n.graph[spring.dst].movable
        return nothing
    end

    delta = deltas[:, spring.dst]
    dif = n.positions[:, spring.src] - n.positions[:, spring.dst]
    dist = norm(dif)
    f =
        springforce(
            (dist - sqrt(n.xdist^2 + n.ydist^2)),
            n.graph[spring.src, spring.dst].spring_constant,
        ) * dif / dist
    align = calculate_vec_align(delta, f)
    #n.graph[spring.src, spring.dst].spring_constant += (dist - sqrt(n.xdist^2 + n.ydist^2)) * align * 0.00005
    return n.graph[spring.src, spring.dst].spring_constant += align * 0.00005
    #println((dist - sqrt(n.xdist^2 + n.ydist^2)) * align * 0.05)
end

function train_epoch!(network, trainer)
    behaviours = trainer.behaviours
    reset!(network)
    simulate!(network, (0.0, 100.0); modifier=behaviours[1].modifier)
    deltas = calculate_deltas(network, behaviours[1])
    for spring in edges(network.graph)
        update_spring!(n, spring, deltas)
    end
    l = loss(network, behaviours[1])
    println("Start loss: ", l)
    return l
end
