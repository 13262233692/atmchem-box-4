module SensitivityAnalysis

using SparseArrays
using LinearAlgebra
import ..AdjointSystem: AdjointModel, FinalTimeObjective, TimeIntegratedObjective,
                         build_adjoint_system, compute_gradient_initial,
                         compute_gradient_emissions, compute_gradient_rate_constants,
                         get_species_index
import ..AdjointSolver: create_checkpoint, solve_adjoint, get_checkpoint_data
import ..AtmChemBox: BoxModel, run_simulation, compute_rate_constants

export SensitivityConfig, SensitivityResult, run_sensitivity_analysis,
       gradient_to_dict, get_gradient_for_species, get_gradient_for_reaction,
       print_sensitivity_summary

struct SensitivityConfig
    objective_type::String
    objective_weights::Dict{String, Float64}
    t_start::Float64
    t_end::Float64
    dt::Float64
    checkpoint_dt::Float64
    target_species::Vector{String}
    target_reactions::Vector{String}
    target_emissions::Vector{String}
    use_sparse::Bool
    sparse_threshold::Int
end

function SensitivityConfig(;
    objective_type::String="final",
    objective_weights::Dict{String, Float64}=Dict(),
    t_start::Float64=0.0,
    t_end::Float64=3600.0,
    dt::Float64=60.0,
    checkpoint_dt::Float64=60.0,
    target_species::Vector{String}=String[],
    target_reactions::Vector{String}=String[],
    target_emissions::Vector{String}=String[],
    use_sparse::Bool=true,
    sparse_threshold::Int=100
)
    return SensitivityConfig(
        objective_type,
        objective_weights,
        t_start,
        t_end,
        dt,
        checkpoint_dt,
        target_species,
        target_reactions,
        target_emissions,
        use_sparse,
        sparse_threshold
    )
end

struct SensitivityResult
    times::Vector{Float64}
    forward_results::Matrix{Float64}
    adjoint_times::Vector{Float64}
    adjoint_results::Matrix{Float64}
    gradient_initial::Vector{Float64}
    gradient_emissions::Vector{Float64}
    gradient_rate_constants::Vector{Float64}
    species_names::Vector{String}
    reaction_ids::Vector{String}
    objective_value::Float64
    config::SensitivityConfig
end

function gradient_to_dict(grad::Vector{Float64}, names::Vector{String})
    return Dict(name => grad[i] for (i, name) in enumerate(names))
end

function get_gradient_for_species(result::SensitivityResult, species_name::String)
    idx = findfirst(==(species_name), result.species_names)
    if idx === nothing
        return Dict()
    end

    grad_initial = gradient_to_dict(result.gradient_initial, result.species_names)
    grad_emis = gradient_to_dict(result.gradient_emissions, result.species_names)

    return Dict(
        "objective_value" => result.objective_value,
        "gradient_wrt_initial" => grad_initial,
        "gradient_wrt_emissions" => grad_emis,
        "sensitivity_of" => species_name,
        "target_species_sensitivity" => get(grad_initial, species_name, 0.0),
        "target_emission_sensitivity" => get(grad_emis, species_name, 0.0)
    )
end

function get_gradient_for_reaction(result::SensitivityResult, reaction_id::String)
    idx = findfirst(==(reaction_id), result.reaction_ids)
    if idx === nothing
        return 0.0
    end
    return result.gradient_rate_constants[idx]
end

function print_sensitivity_summary(result::SensitivityResult; top_n::Int=10)
    println("="^60)
    println("Sensitivity Analysis Summary")
    println("="^60)
    println("Objective value: $(result.objective_value)")
    println("Time window: $(result.config.t_start) - $(result.config.t_end) s")
    println()

    grad_initial = gradient_to_dict(result.gradient_initial, result.species_names)
    grad_emis = gradient_to_dict(result.gradient_emissions, result.species_names)

    println("Top $top_n initial condition sensitivities:")
    sorted_initial = sort(collect(grad_initial), by=x->abs(x[2]), rev=true)
    for (name, val) in sorted_initial[1:min(top_n, end)]
        @printf("  %-15s: %+.4e\n", name, val)
    end
    println()

    println("Top $top_n emission sensitivities:")
    sorted_emis = sort(collect(grad_emis), by=x->abs(x[2]), rev=true)
    for (name, val) in sorted_emis[1:min(top_n, end)]
        if abs(val) > 1e-20
            @printf("  %-15s: %+.4e\n", name, val)
        end
    end
    println()

    if length(result.gradient_rate_constants) > 0
        println("Top $top_n rate constant sensitivities:")
        grad_rxn = Dict(result.reaction_ids[i] => result.gradient_rate_constants[i]
                        for i in 1:length(result.reaction_ids))
        sorted_rxn = sort(collect(grad_rxn), by=x->abs(x[2]), rev=true)
        for (name, val) in sorted_rxn[1:min(top_n, end)]
            if abs(val) > 1e-20
                @printf("  %-15s: %+.4e\n", name, val)
            end
        end
    end

    println("="^60)
end

function run_sensitivity_analysis(model::BoxModel, config::SensitivityConfig)
    mech = model.mechanism
    ode = model.ode

    if config.objective_type == "final"
        if isempty(config.objective_weights)
            error("Must provide objective_weights for final time objective")
        end
        objective = FinalTimeObjective(mech, config.objective_weights)
    elseif config.objective_type == "integrated"
        if isempty(config.objective_weights)
            error("Must provide objective_weights for time integrated objective")
        end
        objective = TimeIntegratedObjective(mech, config.objective_weights)
    else
        error("Unknown objective type: $(config.objective_type). Use 'final' or 'integrated'.")
    end

    adj = build_adjoint_system(ode, objective,
                               use_sparse=config.use_sparse,
                               sparse_threshold=config.sparse_threshold)

    println("Running forward simulation...")
    forward_times, forward_results = run_simulation(model, config.t_start, config.t_end, config.dt)

    y_final = forward_results[end, :]
    J_value = isa(objective, FinalTimeObjective) ? objective(y_final) : 0.0

    if isa(objective, TimeIntegratedObjective)
        for i in 1:length(forward_times)-1
            dt = forward_times[i+1] - forward_times[i]
            J_value += objective(forward_results[i, :], forward_times[i]) * dt
        end
    end

    println("Creating checkpoints...")
    photo_rates = copy(model.photolysis.rates)
    y_checkpoints, t_checkpoints, k_checkpoints = create_checkpoint(
        adj, config.t_start, config.t_end, config.checkpoint_dt,
        model.concentrations, model.temperature, photo_rates,
        model.emissions.rates, model.deposition.rates
    )

    println("Running adjoint simulation...")
    adjoint_times, adjoint_results = solve_adjoint(
        adj, y_final, config.t_start, config.t_end, config.dt,
        y_checkpoints, t_checkpoints, k_checkpoints, model.deposition.rates
    )

    λ_initial = adjoint_results[1, :]

    println("Computing gradients...")
    grad_initial = compute_gradient_initial(λ_initial)

    emission_indices = Int[]
    for name in config.target_emissions
        idx = get_species_index(mech, name)
        if idx > 0
            push!(emission_indices, idx)
        end
    end

    if isempty(emission_indices)
        grad_emissions = compute_gradient_emissions(adj, adjoint_results, adjoint_times)
    else
        grad_emissions = compute_gradient_emissions(adj, adjoint_results, adjoint_times, emission_indices)
    end

    reaction_indices = Int[]
    for id in config.target_reactions
        idx = findfirst(==(id), ode.reaction_ids)
        if idx !== nothing
            push!(reaction_indices, idx)
        end
    end

    if isempty(reaction_indices)
        grad_rates = compute_gradient_rate_constants(
            adj, adjoint_results, y_checkpoints, t_checkpoints,
            collect(1:ode.n_reactions)
        )
    else
        grad_rates = compute_gradient_rate_constants(
            adj, adjoint_results, y_checkpoints, t_checkpoints, reaction_indices
        )
    end

    return SensitivityResult(
        forward_times,
        forward_results,
        adjoint_times,
        adjoint_results,
        grad_initial,
        grad_emissions,
        grad_rates,
        ode.species_names,
        ode.reaction_ids,
        J_value,
        config
    )
end

end
