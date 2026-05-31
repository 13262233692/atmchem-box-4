module PythonBindings

using PyCall
using YAML
using SparseArrays
using LinearAlgebra
using Printf

include("../mechanism_parser/MechanismParser.jl")
include("../ode_system/ODESystem.jl")
include("../solver/RosenbrockSolver.jl")
include("../physics/Physics.jl")
include("../adjoint/AdjointSystem.jl")
include("../adjoint/AdjointSolver.jl")
include("../adjoint/SensitivityAnalysis.jl")

using .MechanismParser
using .ODESystem
using .Solver
using .Physics
using .AdjointSystem
using .AdjointSolver
using .SensitivityAnalysis

export BoxModelPython, run_simulation_py, load_config, BoxModel,
       set_initial_concentrations!, run_simulation, set_emissions!,
       set_photolysis_rates!, set_deposition_rates!, safe_pow, should_use_sparse,
       run_sensitivity_analysis, SensitivityConfig, SensitivityResult,
       gradient_to_dict, print_sensitivity_summary

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

const pyconfig = PyNULL()

function __init__()
    copy!(pyconfig, pyimport("sys").modules)
end

mutable struct BoxModelPython
    jl_model::BoxModel
    config::Dict
end

function load_config(config_file::String)
    return YAML.load_file(config_file)
end

function BoxModelPython(config_file::String)
    config = load_config(config_file)

    mechanism_file = get(config, "mechanism", "mechanisms/mozart.yaml")
    T = get(config, "temperature", 298.15)
    P = get(config, "pressure", 101325.0)
    use_sparse = get(config, "use_sparse", true)
    sparse_threshold = get(config, "sparse_threshold", 100)

    model = BoxModel(mechanism_file, T=T, P=P,
                     use_sparse=use_sparse,
                     sparse_threshold=sparse_threshold)

    initial_conds = get(config, "initial_conditions", Dict())
    set_initial_concentrations!(model, Dict(string(k) => float(v) for (k, v) in initial_conds))

    emissions = get(config, "emissions", Dict())
    set_emissions!(model.emissions, Dict(string(k) => float(v) for (k, v) in emissions))

    photolysis = get(config, "photolysis", Dict())
    if !isempty(photolysis)
        rates_dict = Dict(string(k) => float(v) for (k, v) in photolysis)
        set_photolysis_rates!(model.photolysis, rates_dict)
    end

    deposition = get(config, "deposition", Dict())
    if !isempty(deposition)
        set_deposition_rates!(model.deposition, Dict(string(k) => float(v) for (k, v) in deposition))
    end

    return BoxModelPython(model, config)
end

function run_simulation_py(model_py::BoxModelPython, t_start::Float64=0.0,
                            t_end::Float64=3600.0, dt::Float64=60.0)
    times, results = run_simulation(model_py.jl_model, t_start, t_end, dt)

    species_names = [s.name for s in model_py.jl_model.mechanism.species]
    result_dict = Dict(name => results[:, i] for (i, name) in enumerate(species_names))

    return times, result_dict
end

function get_species_names(model_py::BoxModelPython)
    return [s.name for s in model_py.jl_model.mechanism.species]
end

function get_concentrations(model_py::BoxModelPython)
    return model_py.jl_model.concentrations
end

function run_sensitivity_analysis_py(model_py::BoxModelPython;
                                       objective_type::String="final",
                                       objective_weights::Dict=Dict(),
                                       t_start::Float64=0.0,
                                       t_end::Float64=3600.0,
                                       dt::Float64=60.0,
                                       checkpoint_dt::Float64=60.0,
                                       target_species::Vector{String}=String[],
                                       target_reactions::Vector{String}=String[],
                                       target_emissions::Vector{String}=String[],
                                       use_sparse::Bool=true,
                                       sparse_threshold::Int=100)
    config = SensitivityConfig(
        objective_type=objective_type,
        objective_weights=Dict(string(k) => float(v) for (k, v) in objective_weights),
        t_start=t_start,
        t_end=t_end,
        dt=dt,
        checkpoint_dt=checkpoint_dt,
        target_species=target_species,
        target_reactions=target_reactions,
        target_emissions=target_emissions,
        use_sparse=use_sparse,
        sparse_threshold=sparse_threshold
    )

    result = run_sensitivity_analysis(model_py.jl_model, config)

    species_names = result.species_names
    reaction_ids = result.reaction_ids

    grad_initial_dict = Dict(name => result.gradient_initial[i]
                             for (i, name) in enumerate(species_names))
    grad_emissions_dict = Dict(name => result.gradient_emissions[i]
                               for (i, name) in enumerate(species_names))
    grad_rates_dict = Dict(id => result.gradient_rate_constants[i]
                           for (i, id) in enumerate(reaction_ids))

    forward_dict = Dict(name => result.forward_results[:, i]
                        for (i, name) in enumerate(species_names))
    adjoint_dict = Dict(name => result.adjoint_results[:, i]
                        for (i, name) in enumerate(species_names))

    return Dict(
        "objective_value" => result.objective_value,
        "times" => result.times,
        "forward_results" => forward_dict,
        "adjoint_times" => result.adjoint_times,
        "adjoint_results" => adjoint_dict,
        "gradient_initial" => grad_initial_dict,
        "gradient_emissions" => grad_emissions_dict,
        "gradient_rate_constants" => grad_rates_dict,
        "species_names" => species_names,
        "reaction_ids" => reaction_ids
    )
end

end
