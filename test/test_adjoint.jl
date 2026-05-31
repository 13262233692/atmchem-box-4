using Pkg
Pkg.activate(".")

using AtmChemBox
using SparseArrays
using Test

println("=== Testing Adjoint Mode ===\n")

println("1. Testing objective functions:")
mech = parse_mechanism("mechanisms/mozart.yaml")

obj_final = FinalTimeObjective(mech, Dict("O3" => 1.0))
@test obj_final !== nothing
println("   FinalTimeObjective created successfully")

obj_int = TimeIntegratedObjective(mech, Dict("O3" => 1.0))
@test obj_int !== nothing
println("   TimeIntegratedObjective created successfully")

println("\n2. Testing adjoint system build:")
ode = build_ode_system(mech)
adj = build_adjoint_system(ode, obj_final, use_sparse=false)
@test adj.n_species == ode.n_species
println("   AdjointModel (dense) created successfully: $(adj.n_species) species")

adj_sparse = build_adjoint_system(ode, obj_final, use_sparse=true)
@test isa(adj_sparse.J_transpose, SparseMatrixCSC)
println("   AdjointModel (sparse) created successfully: $(nnz(adj_sparse.J_transpose)) nonzeros")

println("\n3. Testing sensitivity config:")
config = SensitivityConfig(
    objective_type="final",
    objective_weights=Dict("O3" => 1.0),
    t_start=0.0,
    t_end=300.0,
    dt=30.0,
    checkpoint_dt=30.0
)
@test config.objective_type == "final"
@test config.t_end == 300.0
println("   SensitivityConfig created successfully")

println("\n4. Testing gradient_to_dict:")
test_grad = [1.0, 2.0, 3.0]
test_names = ["O3", "NO", "NO2"]
grad_dict = gradient_to_dict(test_grad, test_names)
@test grad_dict["O3"] == 1.0
@test grad_dict["NO"] == 2.0
println("   gradient_to_dict works correctly")

println("\n5. Testing get_species_index:")
idx_o3 = get_species_index(mech, "O3")
@test idx_o3 > 0
idx_nonexist = get_species_index(mech, "NONEXIST")
@test idx_nonexist == -1
println("   get_species_index works correctly")

println("\n6. Testing BoxModel with adjoint support:")
model = BoxModel("mechanisms/mozart.yaml")
@test model.ode.n_species > 0
println("   BoxModel created successfully")

set_initial_concentrations!(model, Dict(
    "O3" => 50e-9,
    "NO" => 10e-9,
    "NO2" => 20e-9,
    "OH" => 1e-12,
    "HO2" => 10e-12,
    "O2" => 2.7e19,
    "M" => 2.5e19
))

println("\n7. Testing forward simulation (prerequisite for adjoint):")
times, results = run_simulation(model, 0.0, 300.0, 30.0)
@test size(results, 2) == model.ode.n_species
@test all(isfinite.(results))
println("   Forward simulation completed successfully")
println("   Final O3: $(results[end, 1]*1e9) ppb")

println("\n8. Testing adjoint RHS computation:")
y = results[1, :]
λ = zeros(model.ode.n_species)
λ[1] = 1.0
dλ = similar(λ)
k = compute_rate_constants(model.ode, model.temperature, [0.01, 0.001])
dep = zeros(model.ode.n_species)

compute_adjoint_rhs!(adj, dλ, λ, y, k, dep, 0.0)
@test all(isfinite.(dλ))
println("   Adjoint RHS computed successfully")

println("\n=== All adjoint tests passed! ===")
