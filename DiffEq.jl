#module B

using DifferentialEquations, LinearAlgebra
using Graphs, MetaGraphsNext
using GLMakie
using PyPlot

mutable struct Spring
    k :: Float64  # spring constant
    l :: Float64  # spring length
end


function springforce(x, k)
    #return - k * x
    #return - k * x - x*x*x
    return -x * (k + x*x)
end

function simulate(accelerations, velocities, positions, p, t)
    (sim, gam) = p
    for v in vertices(sim)
        f = [0.0, 0.0]
        if sim[v]  # is movable
            neuros = neighbors(sim, v)
            springs = collect(sim[neighbor, v] for neighbor in neuros)
            for j = 1 : length(springs)
                dif = positions[:,v] - positions[:,neuros[j]]
                dist = norm(dif)
                f += springforce((dist - springs[j].l), springs[j].k)  * dif / norm(dif)
            end
            f -= gam*velocities[:,v] # damping
            #f[2] -= 0.1 #gravity
            accelerations[:,v] = f
        end
    end
end

function nlys()
    N = 3 # number of beads (neurons) 
    dim = 2  # physical dimension of the system (1,2, or 3)
    #pos = Vector{SVector{2}}(undef,N)   # we could use StaticArrays, maybe a bit faster
    
    gamma = 0.001

    pos = zeros(dim,N)
    vel = zeros(dim,N)
    dv  = zeros(dim,N)   # container for solve

    #pos = [0.0 1.0 0.5; 0.0 0.0 1.0]
    pos = hcat([0,0],[1,0], [0.5, 1.0]) 
    #pos = hcat([0,0], [1,0], [0.5, sqrt(3/4)]) 
    vel = [0.0 0.0 0.0; 0.0 0.0 0.0]
    
    #mov = [true true false]
    
    sim = MetaGraphsNext.MetaGraph(
        Graph();  # underlying graph structure
        label_type=Int,  # color name
        vertex_data_type = Bool,  # movable or not
        edge_data_type=Spring,    
        graph_data="Simulation",  # tag for the whole graph
    )
    sim[1] = true
    sim[2] = true
    sim[3] = false
    sim[1,2] = Spring(-1,  0.6)
    sim[1,3] = Spring(1, 0.6)
    sim[2,3] = Spring(1,  0.6)


    t_span = (0.0,1000.0)

    p = (sim,gamma)
    prob = SecondOrderODEProblem{true}(simulate, vel, pos, t_span, p)
    sol = solve(prob, reltol=1e-8)
    #Plots.plot(sol)
    tt = sol.t
    mat = hcat(sol.u[:]...)  # convert solution object to array
    v = mat[1:6,:]    # velocities
    r = mat[7:12,:]    # positions
    r1x = r[1,:]
    r1y = r[2,:]
    r2x = r[3,:]
    r2y = r[4,:]
    r3x = r[5,:]
    r3y = r[6,:]


    figure(1)
    clf()
    subplot(3,1,1)
    plot(tt, r1x)
    plot(tt, r1y)
    subplot(3,1,2)
    plot(tt, r2x)
    plot(tt, r2y)
    subplot(3,1,3)
    plot(tt, r3x)
    plot(tt, r3y)
    #plot(r1x, r1y)
    #sol
end

function nlys1()
    sim = MetaGraphsNext.MetaGraph(
        Graph();  # underlying graph structure
        label_type=Int,  # color name
        vertex_data_type = Bool,  # movable or not
        edge_data_type=Spring,    
        graph_data="Simulation",  # tag for the whole graph
    )
    sim[1] = false
    sim[2] = true
    sim[3] = true
    sim[1,2] = Spring(1,1)
    sim[2,3] = Spring(1,1)
    #sim[1,3] = Spring(1,1)

    N = 3 # number of beads (neurons) 
    dim = 2  # physical dimension of the system (1,2, or 3)
    #pos = Vector{SVector{2}}(undef,N)   # we could use StaticArrays, maybe a bit faster
    gamma = 0.1

    pos = zeros(dim,N)
    vel = zeros(dim,N)
    pos = hcat([0.0,0.0],[0.5,0.0], [1.0,0.0])
    vel = hcat([0.0,0.0],[0.0,0.1], [0.0,0.0])
    
    t_span = (0.0,100.0)

    p = (sim,gamma)
    prob = SecondOrderODEProblem{true}(simulate, vel, pos, t_span, p)
    sol = solve(prob, reltol=1e-7)
    #Plots.plot(sol)
    tt = sol.t


    mat = hcat(sol.u[:]...)  # convert solution object to array
    global v = mat[1:6,:]    # velocities
    global r = mat[7:12,:]    # positions
    r1x = r[1,:]
    r1y = r[2,:]
    r2x = r[3,:]
    r2y = r[4,:]
    r3x = r[5,:]
    r3y = r[6,:]


    figure(2)
    clf()
    subplot(3,1,1)
    PyPlot.plot(tt, r1x)
    PyPlot.plot(tt, r1y)
    subplot(3,1,2)
    PyPlot.plot(tt, r2x)
    PyPlot.plot(tt, r2y)
    subplot(3,1,3)
    PyPlot.plot(tt, r3x)
    PyPlot.plot(tt, r3y)
    sol
end

function draw!(r)
    fig = GLMakie.Figure()
    ax = Axis(fig[1,1]) 
    global ox = Observable(zeros(Float64, Int(size(r)[1] / 2 )))
    global oy = Observable(zeros(Float64, Int(size(r)[1] / 2 )))
    scatter!(ox,oy, marker = :circle, markersize = 25, color = :blue)
    display(fig)

    x = zeros(Float64, Int(size(r)[1] / 2 ))
    y = zeros(Float64, Int(size(r)[1] / 2 ))
    for i = 1:(size(r)[2])
        for j = 1:size(r)[1]
            if j%2 == 0
                y[Int(j/2)] = r[j,i]
            else
                x[Int(j/2+0.5)] = r[j,i]
            end
            
        end
        ox[] = x
        oy[] = y

        sleep(0.03)
    end
    
end

@time nlys1();
#draw!(r)

#end #module
