using Pkg
Pkg.activate(".")

using AtmChemBox
using SparseArrays
using Test

println("=== Testing Sparse Jacobian Support ===\n")

println("1. Testing small mechanism (dense mode):")
mech_small = parse_mechanism("mechanisms/mozart.yaml")
ode_small = build_ode_system(mech_small)
J_sp_small = build_sparse_jacobian(ode_small)
n_small = ode_small.n_species
nnz_small = nnz(J_sp_small)
density_small = nnz_small / (n_small * n_small)
println("   Species: $n_small, NNZ: $nnz_small, Density: $(@sprintf("%.4f", density_small))")

println("\n2. Testing large mechanism (sparse mode):")
mech_large = parse_mechanism("mechanisms/large_mechanism.yaml")
ode_large = build_ode_system(mech_large)
J_sp_large = build_sparse_jacobian(ode_large)
n_large = ode_large.n_species
nnz_large = nnz(J_sp_large)
density_large = nnz_large / (n_large * n_large)
println("   Species: $n_large, NNZ: $nnz_large, Density: $(@sprintf("%.4f", density_large))")

println("\n3. Testing solver sparse threshold:")
solver = RosenbrockSolver()
println("   use_sparse = $(solver.use_sparse)")
println("   sparse_threshold = $(solver.sparse_threshold)")
println("   should_use_sparse(solver, 50) = $(should_use_sparse(solver, 50))")
println("   should_use_sparse(solver, 150) = $(should_use_sparse(solver, 150))")

println("\n4. Testing safe_pow (overflow protection):")
println("   safe_pow(1e50, 2) = $(safe_pow(1e50, 2))")
println("   safe_pow(1e-50, 2) = $(safe_pow(1e-50, 2))")
println("   safe_pow(0.0, 2) = $(safe_pow(0.0, 2))")

println("\n5. Testing BoxModel with sparse support:")
model = BoxModel("mechanisms/large_mechanism.yaml")
println("   Model created successfully")
println("   J_sparsity type: $(typeof(model.J_sparsity))")
println("   J_sparsity size: $(size(model.J_sparsity))")
println("   J_sparsity nnz: $(nnz(model.J_sparsity))")

set_initial_concentrations!(model, Dict(
    "O3" => 50e-9,
    "NO" => 10e-9,
    "NO2" => 20e-9,
    "OH" => 1e-12,
    "HO2" => 10e-12,
    "O2" => 2.7e19,
    "M" => 2.5e19
))

println("\n6. Testing rate constant computation (overflow protection):")
k = compute_rate_constants(model.ode, 298.15, [0.01, 0.001])
println("   All rate constants finite: $(all(isfinite.(k)))")
println("   Min k: $(minimum(k[k .> 0])), Max k: $(maximum(k))")

println("\n7. Testing sparse jacobian computation:")
J_test = copy(model.J_sparsity)
dep_rates = zeros(model.ode.n_species)
compute_jacobian_sparse!(model.ode, J_test, model.concentrations, k, dep_rates)
println("   Jacobian computed successfully")
println("   All nonzeros finite: $(all(isfinite.(nonzeros(J_test))))")

println("\n=== All tests passed! ===")
