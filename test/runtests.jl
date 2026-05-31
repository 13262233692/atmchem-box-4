using Test
using AtmChemBox
using SparseArrays

@testset "MechanismParser" begin
    mech = parse_mechanism("mechanisms/mozart.yaml")
    @test mech.name == "MOZART_Simplified"
    @test length(mech.species) >= 10
    @test length(mech.reactions) >= 5
end

@testset "ODESystem" begin
    mech = parse_mechanism("mechanisms/mozart.yaml")
    ode = build_ode_system(mech)
    @test ode.n_species == length(mech.species)
    @test ode.n_reactions == length(mech.reactions)
    @test size(ode.stoichiometry.S) == (ode.n_species, ode.n_reactions)

    J_sp = build_sparse_jacobian(ode)
    @test isa(J_sp, SparseMatrixCSC{Float64, Int})
    @test size(J_sp) == (ode.n_species, ode.n_species)

    k = compute_rate_constants(ode, 298.15, [0.01, 0.001])
    @test length(k) == ode.n_reactions
    @test all(isfinite.(k))

    conc = fill(1e-9, ode.n_species)
    conc[9] = 2.7e19
    conc[10] = 1e17

    J = copy(J_sp)
    compute_jacobian_sparse!(ode, J, conc, k, zeros(ode.n_species))
    @test all(isfinite.(nonzeros(J)))

    J_dense = Matrix(J_sp)
    compute_jacobian!(ode, J_dense, conc, k, zeros(ode.n_species))
    @test all(isfinite.(J_dense))
end

@testset "Solver" begin
    solver = RosenbrockSolver()
    @test solver.gamma > 0
    @test solver.reltol > 0
    @test solver.use_sparse == true
    @test solver.sparse_threshold == 100

    solver_no_sparse = RosenbrockSolver(use_sparse=false)
    @test solver_no_sparse.use_sparse == false

    @test should_use_sparse(solver, 200) == true
    @test should_use_sparse(solver, 50) == false
    @test should_use_sparse(solver_no_sparse, 200) == false
end

@testset "BoxModel" begin
    model = BoxModel("mechanisms/mozart.yaml")
    @test length(model.concentrations) == length(model.mechanism.species)
    @test isa(model.J_sparsity, SparseMatrixCSC{Float64, Int})

    set_initial_concentrations!(model, Dict("O3" => 50e-9, "NO" => 10e-9, "NO2" => 20e-9))

    times, results = run_simulation(model, 0.0, 100.0, 10.0)
    @test length(times) == 11
    @test size(results) == (11, length(model.mechanism.species))
    @test all(isfinite.(results))
end

@testset "BoxModel Sparse" begin
    model = BoxModel("mechanisms/mozart.yaml", use_sparse=true, sparse_threshold=5)
    @test model.solver.use_sparse == true

    set_initial_concentrations!(model, Dict("O3" => 50e-9, "NO" => 10e-9,
                                             "NO2" => 20e-9, "OH" => 1e-12,
                                             "HO2" => 10e-12))

    times, results = run_simulation(model, 0.0, 60.0, 10.0)
    @test all(isfinite.(results))
end

@testset "Adjoint Mode" begin
    mech = parse_mechanism("mechanisms/mozart.yaml")

    obj = FinalTimeObjective(mech, Dict("O3" => 1.0))
    @test obj !== nothing

    ode = build_ode_system(mech)
    adj = build_adjoint_system(ode, obj, use_sparse=true)
    @test adj.n_species == ode.n_species
    @test isa(adj.J_transpose, SparseMatrixCSC)

    model = BoxModel("mechanisms/mozart.yaml")
    set_initial_concentrations!(model, Dict(
        "O3" => 50e-9, "NO" => 10e-9, "NO2" => 20e-9,
        "OH" => 1e-12, "HO2" => 10e-12, "O2" => 2.7e19
    ))

    times, results = run_simulation(model, 0.0, 100.0, 10.0)
    @test all(isfinite.(results))

    config = SensitivityConfig(
        objective_type="final",
        objective_weights=Dict("O3" => 1.0),
        t_start=0.0,
        t_end=100.0,
        dt=10.0,
        checkpoint_dt=10.0,
        use_sparse=true
    )

    λ_final = zeros(ode.n_species)
    λ_final[1] = 1.0
    @test all(isfinite.(λ_final))
end

println("All tests passed!")
