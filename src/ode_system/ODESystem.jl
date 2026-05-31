module ODESystem

using SparseArrays
using LinearAlgebra
import ..MechanismParser: ChemicalMechanism, Reaction, Species, RateCoefficient

export ODEModel, build_ode_system, compute_rates!, compute_jacobian!,
       compute_jacobian_sparse!, compute_rate_constants,
       jacobian_sparsity_pattern, safe_pow

function safe_pow(x::Float64, p::Int)
    if p == 0
        return 1.0
    elseif p == 1
        return x
    elseif p == 2
        return x * x
    elseif x == 0.0
        return 0.0
    else
        ax = abs(x)
        if ax > 1e100 && p > 1
            return sign(x) * Inf
        elseif ax < 1e-100 && p > 1
            return 0.0
        end
        result = x^p
        if !isfinite(result)
            if x > 0
                return Inf
            elseif x < 0 && iseven(p)
                return Inf
            else
                return -Inf
            end
        end
        return result
    end
end

function safe_pow(x::Float64, p::Float64)
    if p == 0.0
        return 1.0
    elseif p == 1.0
        return x
    elseif x <= 0.0
        return 0.0
    else
        ax = abs(x)
        if ax > 1e100 && p > 1.0
            return Inf
        elseif ax < 1e-100 && p > 1.0
            return 0.0
        end
        result = x^p
        if !isfinite(result)
            return Inf
        end
        return result
    end
end

struct StoichiometryMatrix
    S::SparseMatrixCSC{Float64, Int}
    reactant_indices::Vector{Vector{Int}}
    product_indices::Vector{Vector{Int}}
    reactant_coeffs::Vector{Vector{Float64}}
    product_coeffs::Vector{Vector{Float64}}
end

struct JacobianSparsity
    I::Vector{Int}
    J::Vector{Int}
    pattern::SparseMatrixCSC{Float64, Int}
    col_ptr::Vector{Int}
    row_val::Vector{Int}
end

struct ODEModel
    n_species::Int
    n_reactions::Int
    stoichiometry::StoichiometryMatrix
    rate_coefficients::Vector{RateCoefficient}
    photolysis_indices::Vector{Int}
    species_names::Vector{String}
    reaction_ids::Vector{String}
    jac_sparsity::JacobianSparsity
end

function jacobian_sparsity_pattern(ode_sys)
    I_idx = Int[]
    J_idx = Int[]

    for j in 1:ode_sys.n_reactions
        reactants = ode_sys.stoichiometry.reactant_indices[j]
        products = ode_sys.stoichiometry.product_indices[j]

        affected = Set{Int}()
        for idx in reactants
            push!(affected, idx)
        end
        for idx in products
            push!(affected, idx)
        end

        for r_idx in reactants
            for aff_idx in affected
                push!(I_idx, aff_idx)
                push!(J_idx, r_idx)
            end
        end
    end

    for i in 1:ode_sys.n_species
        push!(I_idx, i)
        push!(J_idx, i)
    end

    pattern = sparse(I_idx, J_idx, ones(Float64, length(I_idx)),
                     ode_sys.n_species, ode_sys.n_species)
    pattern = droppattern!(pattern)

    I_final = rowvals(pattern)
    J_final = copy(I_final)
    col_ptr = pattern.colptr

    return JacobianSparsity(collect(I_final), collect(J_final), pattern, collect(col_ptr), collect(rowvals(pattern)))
end

function build_ode_system(mechanism::ChemicalMechanism)
    n_species = length(mechanism.species)
    n_reactions = length(mechanism.reactions)

    I = Int[]
    J = Int[]
    V = Float64[]

    reactant_indices = [Int[] for _ in 1:n_reactions]
    product_indices = [Int[] for _ in 1:n_reactions]
    reactant_coeffs = [Float64[] for _ in 1:n_reactions]
    product_coeffs = [Float64[] for _ in 1:n_reactions]

    photolysis_indices = Int[]

    for (j, reaction) in enumerate(mechanism.reactions)
        for (spec_name, coeff) in reaction.reactants
            i = get(mechanism.species_index, spec_name, 0)
            if i > 0
                push!(I, i)
                push!(J, j)
                push!(V, -Float64(coeff))
                push!(reactant_indices[j], i)
                push!(reactant_coeffs[j], Float64(coeff))
            end
        end

        for (spec_name, coeff) in reaction.products
            i = get(mechanism.species_index, spec_name, 0)
            if i > 0
                push!(I, i)
                push!(J, j)
                push!(V, Float64(coeff))
                push!(product_indices[j], i)
                push!(product_coeffs[j], Float64(coeff))
            end
        end

        if reaction.is_photolysis
            push!(photolysis_indices, j)
        end
    end

    S = sparse(I, J, V, n_species, n_reactions)

    stoich = StoichiometryMatrix(
        S,
        reactant_indices,
        product_indices,
        reactant_coeffs,
        product_coeffs
    )

    rate_coeffs = [r.rate for r in mechanism.reactions]
    species_names = [s.name for s in mechanism.species]
    reaction_ids = [r.id for r in mechanism.reactions]

    temp_ode = ODEModel(
        n_species,
        n_reactions,
        stoich,
        rate_coeffs,
        photolysis_indices,
        species_names,
        reaction_ids,
        JacobianSparsity(Int[], Int[], spzeros(n_species, n_species), Int[], Int[])
    )

    jac_sparsity = jacobian_sparsity_pattern(temp_ode)

    return ODEModel(
        n_species,
        n_reactions,
        stoich,
        rate_coeffs,
        photolysis_indices,
        species_names,
        reaction_ids,
        jac_sparsity
    )
end

function compute_rate_constants(ode::ODEModel, T::Float64, photo_rates::Vector{Float64})
    k = zeros(ode.n_reactions)
    R = 8.314

    photo_idx = 1
    for j in 1:ode.n_reactions
        if j in ode.photolysis_indices && photo_idx <= length(photo_rates)
            k[j] = photo_rates[photo_idx]
            photo_idx += 1
        else
            rc = ode.rate_coefficients[j]
            if rc.type == "arrhenius"
                A, B, E = rc.parameters
                exponent = -E / (R * T)
                if abs(exponent) > 500.0
                    k[j] = exponent > 0 ? A * (T/300)^B * 1e300 : 1e-300
                else
                    k[j] = A * (T/300)^B * exp(exponent)
                end
            elseif rc.type == "constant"
                k[j] = rc.parameters[1]
            elseif rc.type == "exp"
                val = rc.parameters[1]
                if val > 500.0
                    k[j] = 1e300
                elseif val < -500.0
                    k[j] = 0.0
                else
                    k[j] = exp(val)
                end
            else
                k[j] = 1e-10
            end
        end

        if !isfinite(k[j])
            k[j] = k[j] > 0 ? 1e300 : 0.0
        end
    end

    return k
end

function compute_reaction_rates!(rates::Vector{Float64}, ode::ODEModel,
                                  concentrations::Vector{Float64}, k::Vector{Float64})
    fill!(rates, 0.0)

    for j in 1:ode.n_reactions
        rate = k[j]
        for (idx, coeff) in zip(ode.stoichiometry.reactant_indices[j],
                                ode.stoichiometry.reactant_coeffs[j])
            c = max(concentrations[idx], 0.0)
            rate *= safe_pow(c, Int(coeff))
            if !isfinite(rate)
                rate = rate > 0 ? 1e300 : -1e300
                break
            end
        end
        rates[j] = rate
    end

    return rates
end

function compute_rates!(ode::ODEModel, dy::Vector{Float64},
                        concentrations::Vector{Float64}, k::Vector{Float64},
                        emissions::Vector{Float64}, deposition::Vector{Float64}, t::Float64)
    reaction_rates = zeros(ode.n_reactions)
    compute_reaction_rates!(reaction_rates, ode, concentrations, k)

    mul!(dy, ode.stoichiometry.S, reaction_rates)

    dy .+= emissions
    dy .-= deposition .* concentrations

    for i in 1:length(dy)
        if !isfinite(dy[i])
            dy[i] = dy[i] > 0 ? 1e300 : -1e300
        end
    end

    return dy
end

function compute_jacobian!(ode::ODEModel, J::Matrix{Float64},
                            concentrations::Vector{Float64}, k::Vector{Float64},
                            deposition::Vector{Float64})
    fill!(J, 0.0)

    for j in 1:ode.n_reactions
        rate = k[j]
        reactants = ode.stoichiometry.reactant_indices[j]
        coeffs = ode.stoichiometry.reactant_coeffs[j]

        for (r_idx, r_coeff) in zip(reactants, coeffs)
            d_rate = rate * r_coeff
            for (r2_idx, r2_coeff) in zip(reactants, coeffs)
                if r2_idx == r_idx
                    if r2_coeff > 1
                        c = max(concentrations[r2_idx], 0.0)
                        d_rate *= safe_pow(c, Int(r2_coeff) - 1)
                    end
                else
                    c = max(concentrations[r2_idx], 0.0)
                    d_rate *= safe_pow(c, Int(r2_coeff))
                end
                if !isfinite(d_rate)
                    d_rate = d_rate > 0 ? 1e300 : -1e300
                    break
                end
            end

            for (p_idx, p_coeff) in zip(ode.stoichiometry.product_indices[j],
                                         ode.stoichiometry.product_coeffs[j])
                J[p_idx, r_idx] += p_coeff * d_rate
            end
            for (rr_idx, rr_coeff) in zip(reactants, coeffs)
                J[rr_idx, r_idx] -= rr_coeff * d_rate
            end
        end
    end

    for i in 1:ode.n_species
        J[i, i] -= deposition[i]
    end

    for i in 1:size(J, 1)
        for j in 1:size(J, 2)
            if !isfinite(J[i, j])
                J[i, j] = 0.0
            end
        end
    end

    return J
end

function compute_jacobian_sparse!(ode::ODEModel, J::SparseMatrixCSC{Float64, Int},
                                   concentrations::Vector{Float64}, k::Vector{Float64},
                                   deposition::Vector{Float64})
    nz = nonzeros(J)
    fill!(nz, 0.0)

    col_ptr = J.colptr
    row_vals = rowvals(J)

    col_idx_map = Dict{Tuple{Int, Int}, Int}()
    for col in 1:ode.n_species
        for k_idx in col_ptr[col]:(col_ptr[col+1]-1)
            row = row_vals[k_idx]
            col_idx_map[(row, col)] = k_idx
        end
    end

    for j in 1:ode.n_reactions
        rate = k[j]
        reactants = ode.stoichiometry.reactant_indices[j]
        coeffs = ode.stoichiometry.reactant_coeffs[j]

        for (r_idx, r_coeff) in zip(reactants, coeffs)
            d_rate = rate * r_coeff
            for (r2_idx, r2_coeff) in zip(reactants, coeffs)
                if r2_idx == r_idx
                    if r2_coeff > 1
                        c = max(concentrations[r2_idx], 0.0)
                        d_rate *= safe_pow(c, Int(r2_coeff) - 1)
                    end
                else
                    c = max(concentrations[r2_idx], 0.0)
                    d_rate *= safe_pow(c, Int(r2_coeff))
                end
                if !isfinite(d_rate)
                    d_rate = d_rate > 0 ? 1e300 : -1e300
                    break
                end
            end

            for (p_idx, p_coeff) in zip(ode.stoichiometry.product_indices[j],
                                         ode.stoichiometry.product_coeffs[j])
                key = (p_idx, r_idx)
                if haskey(col_idx_map, key)
                    nz[col_idx_map[key]] += p_coeff * d_rate
                end
            end
            for (rr_idx, rr_coeff) in zip(reactants, coeffs)
                key = (rr_idx, r_idx)
                if haskey(col_idx_map, key)
                    nz[col_idx_map[key]] -= rr_coeff * d_rate
                end
            end
        end
    end

    for i in 1:ode.n_species
        key = (i, i)
        if haskey(col_idx_map, key)
            nz[col_idx_map[key]] -= deposition[i]
        end
    end

    for i in 1:length(nz)
        if !isfinite(nz[i])
            nz[i] = 0.0
        end
    end

    return J
end

function build_sparse_jacobian(ode::ODEModel)
    return copy(ode.jac_sparsity.pattern)
end

end
