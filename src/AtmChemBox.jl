module AtmChemBox

using LinearAlgebra
using SparseArrays
using Printf

export 
    ChemicalMechanism,
    Reaction,
    Species,
    parse_mechanism,
    ODEModel,
    build_ode_system,
    compute_rates!,
    compute_jacobian!,
    compute_jacobian_sparse!,
    compute_rate_constants,
    build_sparse_jacobian,
    safe_pow,
    RosenbrockSolver,
    solve,
    should_use_sparse,
    Photolysis,
    Emissions,
    Deposition,
    BoxModel,
    run_simulation,
    set_initial_concentrations!,
    set_photolysis_rates!,
    set_emissions!,
    set_deposition_rates!,
    update_photolysis!,
    FinalTimeObjective,
    TimeIntegratedObjective,
    AdjointModel,
    build_adjoint_system,
    compute_adjoint_rhs!,
    compute_gradient_initial,
    compute_gradient_emissions,
    compute_gradient_rate_constants,
    SensitivityConfig,
    SensitivityResult,
    run_sensitivity_analysis,
    gradient_to_dict,
    get_gradient_for_species,
    get_gradient_for_reaction,
    print_sensitivity_summary,
    get_species_index

include("mechanism_parser/MechanismParser.jl")
include("ode_system/ODESystem.jl")
include("solver/RosenbrockSolver.jl")
include("physics/Physics.jl")
include("adjoint/AdjointSystem.jl")
include("adjoint/AdjointSolver.jl")
include("adjoint/SensitivityAnalysis.jl")
include("bindings/PythonInterface.jl")

using .MechanismParser
using .ODESystem
using .Solver
using .Physics
using .AdjointSystem
using .AdjointSolver
using .SensitivityAnalysis
using .PythonBindings

mutable struct BoxModel
    mechanism::ChemicalMechanism
    ode::ODEModel
    solver::RosenbrockSolver
    photolysis::Photolysis
    emissions::Emissions
    deposition::Deposition
    concentrations::Vector{Float64}
    temperature::Float64
    pressure::Float64
    J_sparsity::SparseMatrixCSC{Float64, Int}
end

function BoxModel(mechanism_file::String; T=298.15, P=101325.0,
                  use_sparse::Bool=true, sparse_threshold::Int=100)
    mech = parse_mechanism(mechanism_file)
    ode = build_ode_system(mech)
    solver = RosenbrockSolver(use_sparse=use_sparse,
                              sparse_threshold=sparse_threshold)
    photo = Photolysis(mech)
    emis = Emissions(mech)
    dep = Deposition(mech)
    conc = zeros(length(mech.species))
    J_sp = build_sparse_jacobian(ode)
    return BoxModel(mech, ode, solver, photo, emis, dep, conc, T, P, J_sp)
end

function set_initial_concentrations!(model::BoxModel, conc_dict::Dict)
    for (name, value) in conc_dict
        idx = findfirst(s -> s.name == name, model.mechanism.species)
        if idx !== nothing
            model.concentrations[idx] = value
        end
    end
end

function run_simulation(model::BoxModel, t_start::Float64, t_end::Float64, dt::Float64)
    times = collect(t_start:dt:t_end)
    n_times = length(times)
    n_species = length(model.mechanism.species)
    results = zeros(n_times, n_species)

    y = copy(model.concentrations)
    results[1, :] = y

    for i in 2:n_times
        t = times[i-1]
        update_photolysis!(model.photolysis, t, model.temperature)
        k = compute_rate_constants(model.ode, model.temperature, model.photolysis.rates)

        f_ode = (dy, y, p, t) -> begin
            compute_rates!(model.ode, dy, y, k, model.emissions.rates,
                           model.deposition.rates, t)
        end

        jac_sparse! = (J, y, p, t) -> begin
            compute_jacobian_sparse!(model.ode, J, y, k, model.deposition.rates)
        end

        y = solve(model.solver, f_ode, y, t, t + dt, model.temperature;
                  jacobian_sparse!=jac_sparse!,
                  J_sparsity=model.J_sparsity)
        results[i, :] = y
    end

    return times, results
end

end
