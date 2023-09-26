module S


using GLMakie
using LinearAlgebra
using Graphs
using MetaGraphsNext

mutable struct Neuron
    pos::Vector{Float64}
    velocity::Vector{Float64}
    movable::Bool
end

mutable struct Spring
    spring_constant::Float64
    length::Float64
end

function calculate_force(neuro)
    f = [0, 0]
    #println(neuro)
    neuros = neighbors(sim, neuro)
    springs = collect(sim[neighbor, neuro] for neighbor in neuros)
    for i = 1:length(neuros)
        dif = sim[neuro].pos - sim[neuros[i]].pos
        f += displacement2force((springs[i].length - norm(dif)), springs[i].spring_constant) * dif / norm(dif)
        if isnan.(f) != [0,0]
            println(i, "     ", sim[neuros[i]].pos)
            throw("calculated force as NaN")
        end
    end


    return f
end

function displacement2force(dif, spring_constsant)
    #println(dif, spring_constsant)
    if abs(dif)>0.1
  
        return dif  * 100
    else
        return dif  * spring_constsant
    end
end

function update_position!(n, delta, _draw)
    #println("line 49: ", n)
    sim[n].velocity += calculate_force(n) * delta
    sim[n].velocity *= 0.95
    sim[n].pos += sim[n].velocity * delta

    if _draw
        ox[n][] = sim[n].pos[1]
        oy[n][] = sim[n].pos[2]
    end
end

function initialize2!(sim, columns, rows)
    count = 1
    #yoffset = sqrt(0.75) * (row%2==0)
    count = add_collumn!(sim, 1, rows, sqrt(0.75), count, false)
    for column = 1:columns
        count = add_collumn!(sim, column*2, rows + 1, 0, count, true)
        count = add_collumn!(sim, column*2+1, rows, sqrt(0.75), count, false)
    end
    add_edges!(sim, rows, columns)
end

function add_edges!(sim, rows, columns)
    for i = (2*rows + 2):(2*rows +1)*(columns)+rows
        add_edge!(sim, i, i-(rows*2+1), Spring(r_spring_constant(),1))
    end

    for i = rows + 1:(2*rows +1)*(columns)+rows
        if norm(sim[i].pos - sim[i-rows].pos) < 1.1
            add_edge!(sim, i, i-rows, Spring(r_spring_constant(),1))
        end
        if (i-rows-1 > 0 && norm(sim[i].pos - sim[i-rows-1].pos) < 1.1)
            add_edge!(sim, i, i-rows-1, Spring(r_spring_constant(),1))
        end
    end
end

function r_spring_constant()
    return rand()-0.5
end

function add_collumn!(sim, column, rows, yoffset, count, fixed)
    for row = 1:rows
        movable = !((row == 1 || row == rows) && fixed)
        count = add_neuron!(sim, count, [(column-1) / 2,(row-1)*sqrt(0.75)*2+yoffset], movable)
    end
    return count
end

function add_neuron!(sim, count, pos, movable)
    sim[count] = Neuron(pos, [0,0], movable)
    return count + 1
end

function simulate!(sim, delta, epochs, _draw)
    for i = 1:epochs
        for vertex in vertices(sim)
            if _draw
                points[vertex] = Point2f(sim[vertex].pos[1], sim[vertex].pos[2])
            end
            if sim[vertex].movable
                update_position!(vertex, delta, _draw)
            end
        end
        for i = 1:3
            sim[i].velocity[1] += delta*0.08
        end

        if _draw
            sleep(0.001)
            for i2 = 1:length(springs)
                pp[i2][] = [points[springs[i2][1]], points[springs[i2][2]]]
            end
        end
    end
end

function set_edges!(sim, new_edges)
    springs = Tuple.(edges(sim))
    for i = 1:length(new_edges)
        sim[springs[i][1], springs[i][2]].spring_constant = new_edges[i]
    end
end

function cost(sim)
    return norm(abs.(sim[29].pos - [4, sqrt(0.75)]) + abs.(sim[30].pos - [4.2, 3*sqrt(0.75)]) + abs.(sim[31].pos-[4.2, 5*sqrt(0.75)]))
end

function mutate!(edges)
    for i = 1:length(edges)
        edges[i] += rand() * 0.003 - 0.0015
        #if edges[i] < 0
        #    edges[i] = 0.1
        #end
    end
end

function draw(sim)
    #fig = Figure()
    #ax = Axis(fig[1,1])
    for vertex in vertices(sim)
        scatter!(sim[vertex].pos[1],sim[vertex].pos[2], marker = :circle, markersize = 25, color = :blue)
    end
end

sim = MetaGraphsNext.MetaGraph(
    Graph();  # underlying graph structure
    label_type=Int,  # color name
    vertex_data_type=Neuron,  # RGB code
    edge_data_type=Spring,  # result of the addition between two colors
    graph_data="Simulation",  # tag for the whole graph
)
initialize2!(sim, 4,3)

_draw = true
if _draw
    ox = []
    oy = []
    points = []
    for vertex in vertices(sim)
        push!(ox, Observable(sim[vertex].pos[1]))
        push!(oy, Observable(sim[vertex].pos[2]))
        push!(points, Point2f(sim[vertex].pos[1], sim[vertex].pos[2]))
    end

    fig = Figure()
    ax = Axis(fig[1,1])
    for i = 1:length(ox)
        scatter!(ox[i],oy[i], marker = :circle, markersize = 25, color = :blue)
    end

    pp = []
    springs = Tuple.(edges(sim))
    for edge in springs
        push!(pp, Observable( [points[edge[1]], points[edge[2]]] ))
    end

    for p in pp
        lines!(p, color = :red, stroke = 10)
    end
    display(fig)
end

new_edges = zeros(ne(sim))
for i = 1:ne(sim)
    new_edges[i] = r_spring_constant()
end
#new_edges = [0.38132261262669004, 0.5807102886074409, 0.8466875598199755, 1.0975710722739225, 0.5284998461383046, 1.4534290099417584, 0.4747675542211766, 0.6854695357966796, 1.1668042820916256, 0.1, 1.777809390676027, 1.8522512446673245, 0.791026825945041, 0.6740339883506323, 0.1, 0.7235529006478539, 0.49210199358717, 0.1, 0.32964757548989027, 0.7114966625499006, 0.46785090815589375, 0.6856605330793495, 1.360441710848069, 1.0376886310154134, 0.5104947971313321, 0.26566981877789564, 1.0563145370133729, 0.7014995402733286, 1.1561525607347833, 0.5896322274737511, 0.6439126944840592, 0.30477911269298186, 1.0910979996304355, 0.40996133667692686, 0.25707390077231074, 0.5100330722939093, 0.2943478684800663, 1.1704724877666954, 0.48252271138617575, 0.6372653651264093, 1.1025413386977332, 0.44958796175997057, 1.6824460571040913, 1.2653925072333128, 0.21630546513029714, 0.8863165980851528, 0.7249136297438893, 0.5273197170728434, 0.03376023662311317, 0.9208154949230818, 0.3349976042979017, 0.2684697543122907, 0.8470524810916142, 1.0149613109557236, 0.09744040293297801, 0.26495495940718833, 1.0381622060783897, 0.1, 0.7354998283546883, 0.6561079830984046, 0.007038254854205284, 0.42535777242102596, 0.0034455518045951317, 0.7052457365680829, 0.1907329668519878, 1.2650603875798854, 0.4257257396277106, 0.13946398283971084, 0.3177085056946334, 1.047403840529849, 0.6061738085569367, 0.8699087027157486]
#new_edges = [0.27574162483754366, 0.6174561192355472, 0.35756173312637207, 0.6169514383235664, 0.17441984074596273, 1.6005090361212868, 0.6559195761408763, 0.3531626581819036, 1.0543350316453306, 0.5102343052594632, 1.259203354001298, 2.0317895202180036, 0.3769936091480457, 0.5177991807042138, 0.178483895158682, 0.0427098838299636, 0.765409176302714, 0.1, 0.22603771850883997, 1.0177903002946596, 0.46573838537628887, 1.0283957528760346, 1.214499192971946, 0.9830203751685898, 0.6084303563782993, 0.3857525837612466, 1.0201845700084764, 0.39621830503603933, 0.6155375367486472, 0.42766779502578733, 0.8124866199022722, 0.547246028554702, 1.3642307834291647, 1.0317841457552495, 0.1, 0.2638350390532406, 0.3258473628960232, 0.977885350455888, 0.6397398711153279, 0.23041261952090702, 0.659713691137854, 0.1, 1.798849391236874, 1.2228014266413316, 0.042823790413460544, 0.7402728555260021, 0.31356748134446344, 0.9226352433391757, 0.33228253723082646, 1.170686176199658, 0.3544998950283984, 0.027091685705336876, 1.0399818110224825, 1.0125458488507113, 0.1018864020828183, 0.5770375486829206, 0.98603131511798, 0.40217663689618355, 0.6087734268659248, 1.0290707663045298, 0.14441143195713707, 0.6545476445391539, 0.0048897927937085195, 1.0295716975008748, 0.05994274657637402, 1.4333708841016908, 1.032113822257745, 0.1, 0.42864711770153324, 1.391206939177696, 0.6163567873467569, 1.2057072725169466]
#new_edges = [0.2950541533795367, 0.6352166801854471, 0.4122351093500509, 0.6198557327696771, 0.17215098722049188, 1.6800397732727934, 0.6800683306192804, 0.34861556878961186, 1.0850562963868375, 0.5204326553803895, 1.271169712808774, 2.0490662032870626, 0.3835352954727257, 0.535262733902104, 0.1559209798474274, 0.012700639608200583, 0.7564583063617045, 0.14391203265422992, 0.2970109426040217, 1.0916471280303333, 0.4573866975681251, 1.0845324105084353, 1.2508068202227214, 0.9931985137935594, 0.6626593824236638, 0.42570494413050786, 1.042540519587074, 0.4196840825791615, 0.6594540305578368, 0.44566480665958347, 0.8045772979818068, 0.604648603968185, 1.3941748409198322, 1.0684993687614004, 0.14084166648207344, 0.28022254101502553, 0.35221833474997793, 1.0196125615100986, 0.6921869807337506, 0.24350273153927204, 0.6920640792249054, 0.12468475234102973, 1.8246717579666918, 1.246882153348338, 0.0696939237566623, 0.734809290842925, 0.3783541156423651, 0.9641452394219575, 0.39414468902095473, 1.216540186071032, 0.37543710218519294, 0.040866905075972224, 1.0695526815997354, 1.042925627260033, 0.15436802527789537, 0.6407497076730259, 1.0234129665231764, 0.45058691924677197, 0.63914573490715, 1.0572887229780772, 0.15275539231718444, 0.6768753452139357, 0.00015545295262374761, 1.0522778805996917, 0.051198098441448424, 1.4495613832717897, 1.0592171411220954, 0.0819655985678881, 0.3951688305990398, 1.4004914968872257, 0.660308709774502, 1.2806108932044284]
#new_edges = [0.8973350500195004, 0.6899741683856886, 1.1038014113354173, 0.38449721012071036, 0.36833400717323134, 0.4410162310719009, 0.7442464995961393, 0.24861511955711646, 0.9191022893736297, 0.3922213838676204, 0.4305035678409286, 1.008473328275058, 0.32306735604200854, 0.9499244893762604, 0.5842248858544512, 0.4436738962617731, 0.45311119613634104, 0.9977230284644415, 0.6583194148141828, 1.0473325636794477, 1.0436150422082215, 0.2979660714146187, 0.8258056645053024, 0.5107743581403772, 0.25200437060978004, 0.7912089645483376, 0.718836817362863, 0.348799133778521, 0.2656731681100634, 0.7493868990155437, 1.0628271974596195, 1.0337643790310895, 1.027242323233619, 0.8324905001174024, 0.7086312615804642, 0.925472545107825, 0.6893355798313773, 0.5104236492244547, 1.173870804251852, 0.28153697271181227, 1.027385085087369, 0.24745351855411618, 0.2305410300640029, 0.39296506092827765, 0.6149021230742451, 0.7051768300375787, 0.969677175507062, 0.47994831987697056, 0.33354621376855154, 0.8826430077764236, 0.8462022044828749, 0.6103292465240607, 0.8294837093746251, 0.6823461386769211, 0.541370052084719, 0.4651882320543434, 0.4940350389187659, 0.8973907215916808, 1.0540278537456003, 1.0672527223683872, 0.4344572505971478, 0.22414267518698466, 0.5164412983286613, 0.0464033798708755, 0.6987680865555632, 0.5166896309263271, 1.0471081738728267, 0.4197919976636207, 0.22140199060427698, 1.0607788752034462, 1.0598239216246803, 0.9683481340551356]
new_edeges = [-0.04030801835320083, -0.3297609714342504, -0.03651695383745145, 0.29948457223179054, -0.44426931205003933, -0.45257015880476514, -0.15068957225291993, -0.28981353739570515, -0.10254524521410693, -0.4580103930380496, 0.10409719056754525, 0.16747199612306915, -0.08455747893882841, 0.06409427632595688, 0.06998747807934141, -0.019426128632717234, -0.29764774169186486, 0.20157944384453935, 0.028914562535003734, 0.38636246105496796, -0.028524829388203606, 0.32049120049016333, -0.31563046301608644, 0.13525143353737065, -0.22315971874597212, -0.19609586095077655, -0.4491989153569902, 0.024890896787000055, -0.4546301629574665, -0.45522593777807235, -0.28307772732289344, 0.35955518285779153, 0.08757436198663116, 0.25858030587642616, -0.21286186224047643, -0.04286685375618207, -0.13840661758242823, -0.211101705912592, -0.3297106820426754, 0.2833388324366677, 0.07493230204641657, 0.29395145148715945, 0.283640566723892, 0.38214553757151937, 0.08619285411281223, -0.3563147104776316, 0.37163757152823906, 0.07278939311282552, -0.07320878129111391, -0.21942758856089484, 0.3979919248032747, 0.0023389742501031597, -0.19574296126688506, -0.17811022424033832, -0.3639597925934879, 0.4341431270532654, -0.05640883987654084, -0.10990849127893942, -0.1694390405816776, -0.13085049939350873, -0.3925266075579762, -0.2935693699453352, -0.23664586577576482, -0.20220231987836806, 0.20160645081366088, 0.12157964112829382, -0.25230]


set_edges!(sim, new_edges)
simulate!(sim, 0.01, 50000, _draw)
combinations = []
push!(combinations, copy(new_edges))
costs = []
push!(costs, cost(sim))


@time for i = 1:0
    index = findmin(costs)[2]
    new_edges = copy(combinations[index])
    initialize2!(sim, 4,3)
    mutate!(new_edges)
    set_edges!(sim, new_edges)
    simulate!(sim, 0.01, 5000, _draw)
    push!(combinations, copy(new_edges))
    push!(costs, cost(sim))
    println(i, " cost: ", cost(sim))
end

#draw(sim)

#=
ox = []
oy = []
points = []
for vertex in vertices(sim)
    push!(ox, Observable(sim[vertex].pos[1]))
    push!(oy, Observable(sim[vertex].pos[2]))
    push!(points, Point2f(sim[vertex].pos[1], sim[vertex].pos[2]))
end

fig = Figure()
ax = Axis(fig[1,1])
for i = 1:length(ox)
    scatter!(ox[i],oy[i], marker = :circle, markersize = 25, color = :blue)
end

pp = []
springs = Tuple.(edges(sim))
for edge in springs
    push!(pp, Observable( [points[edge[1]], points[edge[2]]] ))
end

for p in pp
    lines!(p, color = :red, stroke = 10)
end

#sim[40,48] = Spring(-100,1)
display(fig)


delta = 0.01
foo = []

@time for i = 1:1
    push!(foo, sim[1].pos[1])
    for vertex in vertices(sim)
        points[vertex] = Point2f(sim[vertex].pos[1], sim[vertex].pos[2])
        if sim[vertex].movable
            update_position!(vertex, delta)
        end
    end

    
    for i2 = 1:length(springs)
        pp[i2][] = [points[springs[i2][1]], points[springs[i2][2]]]
    end


    for i = 1:3
        sim[i].velocity[1] += delta*0.08
    end

    sleep(delta/200)
end
=#

end # module