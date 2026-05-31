module AdjointSolver

using LinearAlgebra
using SparseArrays
import ..AdjointSystem: AdjointModel, compute_adjoint_rhs!, compute_final_adjoint_condition!
import ..ODESystem: compute_rate_constants
import ..Physics: update_photolysis!
import ..Solver: RosenbrockSolver, should_use_sparse

export AdjointSolverState, solve_adjoint, solve_adjoint_sparse,
       create_checkpoint, get_checkpoint_data

struct AdjointSolverState
    adj::AdjointModel
    solver::RosenbrockSolver
    t_start::Float64
    t_end::Float64
    n_checkpoints::Int
    y_checkpoints::Matrix{Float64}
    t_checkpoints::Vector{Float64}
    k_checkpoints::Matrix{Float64}
end

function create_checkpoint(ode::AdjointModel, t_start::Float64, t_end::Float64,
                            dt::Float64, y_initial::Vector{Float64},
                            T::Float64, photo_rates::Vector{Float64},
                            emissions::Vector{Float64}, deposition::Vector{Float64})
    n_times = Int(ceil((t_end - t_start) / dt)) + 1
    n_species = ode.n_species
    n_reactions = ode.n_reactions

    y_checkpoints = zeros(n_times, n_species)
    t_checkpoints = zeros(n_times)
    k_checkpoints = zeros(n_times, n_reactions)

    y = copy(y_initial)
    t = t_start

    y_checkpoints[1, :] = y
    t_checkpoints[1] = t
    k_checkpoints[1, :] = compute_rate_constants(ode.ode, T, photo_rates)

    for i in 2:n_times
        t_current = t_checkpoints[i-1]
        k = k_checkpoints[i-1, :]

        f_forward = (dy, y, p, t) -> begin
            reaction_rates = zeros(n_reactions)
            for j in 1:n_reactions
                rate = k[j]
                for (idx, coeff) in zip(ode.ode.stoichiometry.reactant_indices[j],
                                        ode.ode.stoichiometry.reactant_coeffs[j])
                    rate *= max(y[idx], 0.0)^Int(coeff)
                    if !isfinite(rate)
                        rate = rate > 0 ? 1e300 : -1e300
                        break
                    end
                end
                reaction_rates[j] = rate
            end

            mul!(dy, ode.ode.stoichiometry.S, reaction_rates)
            dy .+= emissions
            dy .-= deposition .* y

            for idx in 1:n_species
                if !isfinite(dy[idx])
                    dy[idx] = dy[idx] > 0 ? 1e300 : -1e300
                end
            end
            return dy
        end

        h = min(dt, t_end - t_current)
        k1 = zeros(n_species)
        f_forward(k1, y, nothing, t_current)
        y = y + h * k1

        if i > 1
            t_checkpoints[i] = t_checkpoints[i-1] + min(dt, t_end - t_checkpoints[i-1])
        else
            t_checkpoints[i] = t_checkpoints[i-1] + dt
        end

        y_checkpoints[i, :] = y

        photo_rates_current = copy(photo_rates)
        photo_scaling = max(0.0, cos((t_checkpoints[i] / 3600.0 - 12.0) * pi / 12.0))^0.5
        photo_rates_current .*= photo_scaling
        k_checkpoints[i, :] = compute_rate_constants(ode.ode, T, photo_rates_current)

        if t_checkpoints[i] >= t_end
            break
        end
    end

    return y_checkpoints, t_checkpoints, k_checkpoints
end

function get_checkpoint_data(t::Float64, y_checkpoints::Matrix{Float64},
                              t_checkpoints::Vector{Float64}, k_checkpoints::Matrix{Float64})
    idx = searchsortedfirst(t_checkpoints, t)

    if idx <= 1
        return y_checkpoints[1, :], k_checkpoints[1, :]
    elseif idx > length(t_checkpoints)
        return y_checkpoints[end, :], k_checkpoints[end, :]
    else
        t0 = t_checkpoints[idx-1]
        t1 = t_checkpoints[idx]
        frac = (t - t0) / (t1 - t0)

        y0 = y_checkpoints[idx-1, :]
        y1 = y_checkpoints[idx, :]
        y_interp = y0 + frac * (y1 - y0)

        k0 = k_checkpoints[idx-1, :]
        k1 = k_checkpoints[idx, :]
        k_interp = k0 + frac * (k1 - k0)

        return y_interp, k_interp
    end
end

function solve_adjoint(adj::AdjointModel, y_final::Vector{Float64},
                        t_start::Float64, t_end::Float64, dt::Float64,
                        y_checkpoints::Matrix{Float64}, t_checkpoints::Vector{Float64},
                        k_checkpoints::Matrix{Float64},
                        deposition::Vector{Float64};
                        saveat::Vector{Float64}=Float64[])
    n = adj.n_species
    solver = adj.use_sparse ? RosenbrockSolver(use_sparse=true) : RosenbrockSolver(use_sparse=false)

    λ = zeros(n)
    compute_final_adjoint_condition!(adj, λ, y_final)

    times_rev = collect(t_end:-dt:t_start)
    if times_rev[end] != t_start
        push!(times_rev, t_start)
    end
    reverse!(times_rev)

    n_times = length(times_rev)
    λ_history = zeros(n_times, n)
    t_history = zeros(n_times)

    λ_history[end, :] = λ
    t_history[end] = t_end

    current_idx = n_times

    for i in (n_times-1):-1:1
        t_prev = times_rev[i+1]
        t_curr = times_rev[i]
        h = t_curr - t_prev

        y_curr, k_curr = get_checkpoint_data(t_prev, y_checkpoints, t_checkpoints, k_checkpoints)

        f_adjoint = (dλ, λ, p, t) -> begin
            y_t, k_t = get_checkpoint_data(t, y_checkpoints, t_checkpoints, k_checkpoints)
            compute_adjoint_rhs!(adj, dλ, λ, y_t, k_t, deposition, t)
        end

        λ = reverse_step(solver, f_adjoint, λ, t_prev, h, y_curr, k_curr, deposition, adj)

        λ_history[i, :] = λ
        t_history[i] = t_curr
    end

    reverse!(λ_history, dims=1)
    reverse!(t_history)

    return t_history, λ_history
end

function reverse_step(solver::RosenbrockSolver, f, λ::Vector{Float64},
                      t::Float64, h::Float64, y::Vector{Float64},
                      k::Vector{Float64}, deposition::Vector{Float64},
                      adj::AdjointModel)
    n = length(λ)
    gamma = solver.gamma

    dλdt = similar(λ)
    f(dλdt, λ, nothing, t)

    if adj.use_sparse
        J = copy(adj.ode.jac_sparsity.pattern)
        compute_jacobian_sparse!(adj.ode, J, y, k, deposition)
        J_T = transpose(J)

        I_sparse = sparse(1.0I, n, n)
        W = I_sparse + h * gamma * J_T
        lu_W = lu(W)

        k1 = lu_W \ dλdt
    else
        J = Matrix{Float64}(undef, n, n)
        compute_jacobian!(adj.ode, J, y, k, deposition)

        M = I + h * gamma * transpose(J)
        lu_M = lu(M)

        k1 = lu_M \ dλdt
    end

    λ2 = λ + solver.a21 * h * k1
    dλdt2 = similar(λ)
    f(dλdt2, λ2, nothing, t - h/2)

    rhs2 = dλdt2 + solver.c21 * k1
    k2 = adj.use_sparse ? (lu_W \ rhs2) : (lu_M \ rhs2)

    λ3 = λ + solver.a31 * h * k1 + solver.a32 * h * k2
    dλdt3 = similar(λ)
    f(dλdt3, λ3, nothing, t - h)

    rhs3 = dλdt3 + solver.c31 * k1 + solver.c32 * k2
    k3 = adj.use_sparse ? (lu_W \ rhs3) : (lu_M \ rhs3)

    λ_new = λ + h * (solver.b1 * k1 + solver.b2 * k2 + solver.b3 * k3)

    return λ_new
end

function compute_adjoint_rhs_alt(adj::AdjointModel, dλ::Vector{Float64},
                                  λ::Vector{Float64}, y::Vector{Float64},
                                  k::Vector{Float64}, deposition::Vector{Float64},
                                  t::Float64)
    n = adj.n_species

    if adj.use_sparse
        J = copy(adj.ode.jac_sparsity.pattern)
        compute_jacobian_sparse!(adj.ode, J, y, k, deposition)
        mul!(dλ, transpose(J), λ, -1.0, 0.0)
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

import ..ODESystem: compute_jacobian_sparse!, compute_jacobian!
import ..AdjointSystem: TimeIntegratedObjective, gradient!

end
