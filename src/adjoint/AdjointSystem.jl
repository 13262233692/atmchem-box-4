module AdjointSystem

using SparseArrays
using LinearAlgebra
import ..ODESystem: ODEModel, compute_jacobian!, compute_jacobian_sparse!
import ..MechanismParser: ChemicalMechanism, Species

export AbstractObjective, FinalTimeObjective, TimeIntegratedObjective,
       AdjointModel, build_adjoint_system, compute_adjoint_rhs!,
       compute_gradient_initial, compute_gradient_emissions,
       compute_gradient_rate_constants, get_species_index

abstract type AbstractObjective end

struct FinalTimeObjective <: AbstractObjective
    species_weights::Dict{String, Float64}
    species_indices::Vector{Int}
    weights::Vector{Float64}
end

function FinalTimeObjective(mechanism::ChemicalMechanism, weights::Dict{String, Float64})
    indices = Int[]
    w = Float64[]
    for (name, weight) in weights
        idx = findfirst(s -> s.name == name, mechanism.species)
        if idx !== nothing
            push!(indices, idx)
            push!(w, weight)
        end
    end
    return FinalTimeObjective(weights, indices, w)
end

function (obj::FinalTimeObjective)(y::Vector{Float64})
    val = 0.0
    for (idx, w) in zip(obj.species_indices, obj.weights)
        val += w * y[idx]
    end
    return val
end

function gradient!(obj::FinalTimeObjective, grad::Vector{Float64}, y::Vector{Float64})
    fill!(grad, 0.0)
    for (idx, w) in zip(obj.species_indices, obj.weights)
        grad[idx] = w
    end
    return grad
end

struct TimeIntegratedObjective <: AbstractObjective
    species_weights::Dict{String, Float64}
    species_indices::Vector{Int}
    weights::Vector{Float64}
end

function TimeIntegratedObjective(mechanism::ChemicalMechanism, weights::Dict{String, Float64})
    indices = Int[]
    w = Float64[]
    for (name, weight) in weights
        idx = findfirst(s -> s.name == name, mechanism.species)
        if idx !== nothing
            push!(indices, idx)
            push!(w, weight)
        end
    end
    return TimeIntegratedObjective(weights, indices, w)
end

function (obj::TimeIntegratedObjective)(y::Vector{Float64}, t::Float64)
    val = 0.0
    for (idx, w) in zip(obj.species_indices, obj.weights)
        val += w * y[idx]
    end
    return val
end

function gradient!(obj::TimeIntegratedObjective, grad::Vector{Float64}, y::Vector{Float64}, t::Float64)
    fill!(grad, 0.0)
    for (idx, w) in zip(obj.species_indices, obj.weights)
        grad[idx] = w
    end
    return grad
end

struct AdjointModel
    ode::ODEModel
    n_species::Int
    n_reactions::Int
    objective::AbstractObjective
    J_transpose::Union{Matrix{Float64}, SparseMatrixCSC{Float64, Int}}
    use_sparse::Bool
end

function build_adjoint_system(ode::ODEModel, objective::AbstractObjective;
                              use_sparse::Bool=true, sparse_threshold::Int=100)
    n = ode.n_species
    use_sparse_actual = use_sparse && n >= sparse_threshold

    if use_sparse_actual
        J_transpose = copy(ode.jac_sparsity.pattern)
    else
        J_transpose = Matrix{Float64}(undef, n, n)
    end

    return AdjointModel(
        ode,
        n,
        ode.n_reactions,
        objective,
        J_transpose,
        use_sparse_actual
    )
end

function compute_adjoint_rhs!(adj::AdjointModel, dλ::Vector{Float64},
                              λ::Vector{Float64}, y::Vector{Float64},
                              k::Vector{Float64}, deposition::Vector{Float64},
                              t::Float64)
    n = adj.n_species

    if adj.use_sparse
        J = copy(adj.ode.jac_sparsity.pattern)
        compute_jacobian_sparse!(adj.ode, J, y, k, deposition)
        J_T = transpose(J)
        mul!(dλ, J_T, λ, -1.0, 0.0)
    else
        J = Matrix{Float64}(undef, n, n)
        compute_jacobian!(adj.ode, J, y, k, deposition)
        mul!(dλ, transpose(J), λ, -1.0, 0.0)
    end

    if isa(adj.objective, TimeIntegratedObjective)
        grad = similar(dλ)
        gradient!(adj.objective, grad, y, t)
        dλ .-= grad
    end

    for i in 1:n
        if !isfinite(dλ[i])
            dλ[i] = 0.0
        end
    end

    return dλ
end

function compute_final_adjoint_condition!(adj::AdjointModel, λ::Vector{Float64},
                                           y_final::Vector{Float64})
    fill!(λ, 0.0)

    if isa(adj.objective, FinalTimeObjective)
        gradient!(adj.objective, λ, y_final)
    end

    return λ
end

function compute_gradient_initial(λ_initial::Vector{Float64})
    return copy(λ_initial)
end

function compute_gradient_emissions(adj::AdjointModel, λ_history::Matrix{Float64},
                                     times::Vector{Float64}, emission_indices::Vector{Int})
    n_times = length(times)
    n_params = length(emission_indices)
    grad = zeros(n_params)

    for i in 1:n_params
        idx = emission_indices[i]
        for j in 1:n_times
            dt = j > 1 ? times[j] - times[j-1] : 0.0
            grad[i] += λ_history[j, idx] * dt
        end
    end

    return grad
end

function compute_gradient_emissions(adj::AdjointModel, λ_history::Matrix{Float64},
                                     times::Vector{Float64})
    n_times = length(times)
    n = adj.n_species
    grad = zeros(n)

    for idx in 1:n
        for j in 1:n_times
            dt = j > 1 ? times[j] - times[j-1] : 0.0
            grad[idx] += λ_history[j, idx] * dt
        end
    end

    return grad
end

function compute_gradient_rate_constants(adj::AdjointModel,
                                          λ_history::Matrix{Float64},
                                          y_history::Matrix{Float64},
                                          times::Vector{Float64},
                                          reaction_indices::Vector{Int})
    n_times = length(times)
    n_params = length(reaction_indices)
    grad = zeros(n_params)

    for p in 1:n_params
        j_rxn = reaction_indices[p]
        reactants = adj.ode.stoichiometry.reactant_indices[j_rxn]
        coeffs = adj.ode.stoichiometry.reactant_coeffs[j_rxn]
        products = adj.ode.stoichiometry.product_indices[j_rxn]
        p_coeffs = adj.ode.stoichiometry.product_coeffs[j_rxn]

        for t_idx in 1:n_times
            dt = t_idx > 1 ? times[t_idx] - times[t_idx-1] : 0.0
            y = y_history[t_idx, :]
            λ = λ_history[t_idx, :]

            rate = 1.0
            for (r_idx, c) in zip(reactants, coeffs)
                rate *= y[r_idx]^c
            end

            dJ_dk = 0.0
            for (p_idx, pc) in zip(products, p_coeffs)
                dJ_dk += pc * λ[p_idx] * rate
            end
            for (r_idx, rc) in zip(reactants, coeffs)
                dJ_dk -= rc * λ[r_idx] * rate
            end

            grad[p] += dJ_dk * dt
        end
    end

    return grad
end

function get_species_index(mechanism::ChemicalMechanism, name::String)
    idx = findfirst(s -> s.name == name, mechanism.species)
    return idx !== nothing ? idx : -1
end

end
