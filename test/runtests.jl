using MNN
using Test

@testset "MNN.jl" begin
    @testset "Network" begin
        net = Network(5, 4)
        @test net.columns == 5
        @test net.rows == 4
        @test net.neuron_count == 17
        @test (1, MNN.get_neuron(net, 1, 1)) == net.graph.vertex_properties[1]
        @test (17, MNN.get_neuron(net, 5, 3)) == net.graph.vertex_properties[net.neuron_count]
        @test (9, MNN.get_neuron(net, 3, 2)) == net.graph.vertex_properties[9]
    end
    @testset "Deformation" begin
        net = Network(5, 4)
    end
    # Write your tests here.
end
