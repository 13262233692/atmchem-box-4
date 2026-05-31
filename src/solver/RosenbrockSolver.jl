module Solver

using LinearAlgebra
using SparseArrays

export RosenbrockSolver, solve, solve_adaptive, should_use_sparse

struct RosenbrockSolver
    gamma::Float64
    a21::Float64
    a31::Float64
    a32::Float64
    c21::Float64
    c31::Float64
    c32::Float64
    b1::Float64
    b2::Float64
    b3::Float64
    e1::Float64
    e2::Float64
    e3::Float64
    max_steps::Int
    reltol::Float64
    abstol::Float64
    use_sparse::Bool
    sparse_threshold::Int
end

function RosenbrockSolver(; method::String="rodas3", max_steps::Int=10000,
                          reltol::Float64=1e-6, abstol::Float64=1e-9,
                          use_sparse::Bool=true, sparse_threshold::Int=100)
    if method == "rosenbrock2"
        gamma = 1.0 + sqrt(2.0)/2.0
        return RosenbrockSolver(
            gamma,
            1.0/gamma,
            0.0, 0.0,
            0.0, 0.0, 0.0,
            1.0/(2.0*gamma), 1.0/(2.0*gamma), 0.0,
            1.0/(2.0*gamma), 1.0/(2.0*gamma), 0.0,
            max_steps, reltol, abstol, use_sparse, sparse_threshold
        )
    elseif method == "rodas3"
        gamma = 0.435866521508459
        a21 = gamma
        a31 = gamma
        a32 = 0.0
        c21 = -3.0/2.0
        c31 = 5.0/2.0
        c32 = -5.0/2.0
        b1 = 3.0/4.0
        b2 = 0.0
        b3 = 1.0/4.0
        e1 = 1.0/4.0
        e2 = 0.0
        e3 = -1.0/4.0
        return RosenbrockSolver(
            gamma, a21, a31, a32, c21, c31, c32, b1, b2, b3, e1, e2, e3,
            max_steps, reltol, abstol, use_sparse, sparse_threshold
        )
    else
        error("Unknown Rosenbrock method: $method")
    end
end

function should_use_sparse(solver::RosenbrockSolver, n::Int)
    return solver.use_sparse && n >= solver.sparse_threshold
end

function solve_step_sparse(solver::RosenbrockSolver, f, y::Vector{Float64}, t::Float64,
                            h::Float64, params, J_sparsity::SparseMatrixCSC{Float64, Int},
                            jacobian_sparse!::Union{Function, Nothing}=nothing)
    n = length(y)
    gamma = solver.gamma

    dydt = similar(y)
    f(dydt, y, params, t)

    J = copy(J_sparsity)
    if jacobian_sparse! !== nothing
        jacobian_sparse!(J, y, params, t)
    else
        compute_jacobian_finite_difference_sparse!(J, f, y, t, params, dydt)
    end

    I_sparse = sparse(1.0I, n, n)
    W = I_sparse - h * gamma * J

    lu_W = lu(W)
    k1 = lu_W \ dydt

    y2 = y + solver.a21 * h * k1
    dydt2 = similar(y)
    f(dydt2, y2, params, t + h/2)

    rhs2 = dydt2 + solver.c21 * k1
    k2 = lu_W \ rhs2

    y3 = y + solver.a31 * h * k1 + solver.a32 * h * k2
    dydt3 = similar(y)
    f(dydt3, y3, params, t + h)

    rhs3 = dydt3 + solver.c31 * k1 + solver.c32 * k2
    k3 = lu_W \ rhs3

    y_new = y + h * (solver.b1 * k1 + solver.b2 * k2 + solver.b3 * k3)
    err = h * (solver.e1 * k1 + solver.e2 * k2 + solver.e3 * k3)

    return y_new, err
end

function solve_step_dense(solver::RosenbrockSolver, f, y::Vector{Float64}, t::Float64,
                           h::Float64, params, jacobian!::Union{Function, Nothing}=nothing)
    n = length(y)
    gamma = solver.gamma

    dydt = similar(y)
    f(dydt, y, params, t)

    J = Matrix{Float64}(undef, n, n)
    if jacobian! !== nothing
        jacobian!(J, y, params, t)
    else
        compute_jacobian_finite_difference!(J, f, y, t, params, dydt)
    end

    M = I - h * gamma * J
    lu_M = lu(M)

    k1 = lu_M \ dydt

    y2 = y + solver.a21 * h * k1
    dydt2 = similar(y)
    f(dydt2, y2, params, t + h/2)

    rhs2 = dydt2 + solver.c21 * k1
    k2 = lu_M \ rhs2

    y3 = y + solver.a31 * h * k1 + solver.a32 * h * k2
    dydt3 = similar(y)
    f(dydt3, y3, params, t + h)

    rhs3 = dydt3 + solver.c31 * k1 + solver.c32 * k2
    k3 = lu_M \ rhs3

    y_new = y + h * (solver.b1 * k1 + solver.b2 * k2 + solver.b3 * k3)
    err = h * (solver.e1 * k1 + solver.e2 * k2 + solver.e3 * k3)

    return y_new, err
end

function compute_jacobian_finite_difference_sparse!(J::SparseMatrixCSC{Float64, Int}, f,
                                                     y::Vector{Float64}, t::Float64,
                                                     params, dydt::Vector{Float64})
    n = length(y)
    nz = nonzeros(J)

    col_ptr = J.colptr
    row_vals = rowvals(J)

    y_perturbed = similar(y)
    dydt_perturbed = similar(y)

    for col in 1:n
        eps_j = sqrt(eps(Float64)) * max(abs(y[col]), 1e-8)
        if eps_j == 0.0
            eps_j = sqrt(eps(Float64))
        end

        y_perturbed .= y
        y_perturbed[col] += eps_j

        f(dydt_perturbed, y_perturbed, params, t)

        for k_idx in col_ptr[col]:(col_ptr[col+1]-1)
            row = row_vals[k_idx]
            nz[k_idx] = (dydt_perturbed[row] - dydt[row]) / eps_j
            if !isfinite(nz[k_idx])
                nz[k_idx] = 0.0
            end
        end
    end

    return J
end

function compute_jacobian_finite_difference!(J::Matrix{Float64}, f, y::Vector{Float64},
                                              t::Float64, params, dydt::Vector{Float64})
    n = length(y)
    y_perturbed = similar(y)
    dydt_perturbed = similar(y)

    for j in 1:n
        eps_j = sqrt(eps(Float64)) * max(abs(y[j]), 1e-8)
        if eps_j == 0.0
            eps_j = sqrt(eps(Float64))
        end

        y_perturbed .= y
        y_perturbed[j] += eps_j

        f(dydt_perturbed, y_perturbed, params, t)

        for i in 1:n
            J[i, j] = (dydt_perturbed[i] - dydt[i]) / eps_j
            if !isfinite(J[i, j])
                J[i, j] = 0.0
            end
        end
    end

    return J
end

function solve(solver::RosenbrockSolver, f, y0::Vector{Float64},
               t0::Float64, t_end::Float64, params=nothing;
               jacobian!::Union{Function, Nothing}=nothing,
               jacobian_sparse!::Union{Function, Nothing}=nothing,
               J_sparsity::Union{SparseMatrixCSC{Float64, Int}, Nothing}=nothing)
    h = min(1.0, (t_end - t0) / 100.0)
    t = t0
    y = copy(y0)
    n = length(y)

    use_sparse = should_use_sparse(solver, n) && J_sparsity !== nothing

    if use_sparse
        return solve_impl_sparse(solver, f, y, t, t_end, h, params,
                                  J_sparsity, jacobian_sparse!)
    else
        return solve_impl_dense(solver, f, y, t, t_end, h, params, jacobian!)
    end
end

function solve_impl_sparse(solver::RosenbrockSolver, f, y::Vector{Float64},
                            t::Float64, t_end::Float64, h::Float64, params,
                            J_sparsity::SparseMatrixCSC{Float64, Int},
                            jacobian_sparse!::Union{Function, Nothing})
    n_steps = 0
    while t < t_end && n_steps < solver.max_steps
        h = min(h, t_end - t)

        y_new, err = solve_step_sparse(solver, f, y, t, h, params,
                                        J_sparsity, jacobian_sparse!)

        scale = solver.abstol .+ solver.reltol .* max.(abs.(y), abs.(y_new))
        err_norm = sqrt(mean((err ./ scale).^2))

        if isnan(err_norm) || isinf(err_norm)
            h *= 0.1
            if h < 1e-15
                y = y_new
                break
            end
            continue
        end

        if err_norm < 1.0
            y = y_new
            t += h
            n_steps += 1
        end

        h *= min(5.0, max(0.2, 0.9 * (1.0 / max(err_norm, 1e-30))^(1/3)))
    end

    return y
end

function solve_impl_dense(solver::RosenbrockSolver, f, y::Vector{Float64},
                           t::Float64, t_end::Float64, h::Float64, params,
                           jacobian!::Union{Function, Nothing})
    n_steps = 0
    while t < t_end && n_steps < solver.max_steps
        h = min(h, t_end - t)

        y_new, err = solve_step_dense(solver, f, y, t, h, params, jacobian!)

        scale = solver.abstol .+ solver.reltol .* max.(abs.(y), abs.(y_new))
        err_norm = sqrt(mean((err ./ scale).^2))

        if isnan(err_norm) || isinf(err_norm)
            h *= 0.1
            if h < 1e-15
                y = y_new
                break
            end
            continue
        end

        if err_norm < 1.0
            y = y_new
            t += h
            n_steps += 1
        end

        h *= min(5.0, max(0.2, 0.9 * (1.0 / max(err_norm, 1e-30))^(1/3)))
    end

    return y
end

function solve_adaptive(solver::RosenbrockSolver, f, y0::Vector{Float64},
                        t0::Float64, t_end::Float64, params=nothing;
                        jacobian!::Union{Function, Nothing}=nothing,
                        jacobian_sparse!::Union{Function, Nothing}=nothing,
                        J_sparsity::Union{SparseMatrixCSC{Float64, Int}, Nothing}=nothing,
                        saveat::Vector{Float64}=Float64[])
    h = min(1.0, (t_end - t0) / 100.0)
    t = t0
    y = copy(y0)
    n = length(y)

    use_sparse = should_use_sparse(solver, n) && J_sparsity !== nothing

    if isempty(saveat)
        saveat = [t_end]
    end
    sort!(saveat)

    times = Float64[]
    solution = Vector{Vector{Float64}}()

    save_idx = 1

    if !isempty(saveat) && saveat[1] == t0
        push!(times, t0)
        push!(solution, copy(y0))
        save_idx += 1
    end

    n_steps = 0
    while t < t_end && n_steps < solver.max_steps
        h = min(h, t_end - t)

        if save_idx <= length(saveat) && saveat[save_idx] < t + h
            h_save = saveat[save_idx] - t
            if h_save <= 0
                save_idx += 1
                continue
            end
            if use_sparse
                y_temp, _ = solve_step_sparse(solver, f, y, t, h_save, params,
                                               J_sparsity, jacobian_sparse!)
            else
                y_temp, _ = solve_step_dense(solver, f, y, t, h_save, params, jacobian!)
            end
            t = saveat[save_idx]
            y = y_temp
            push!(times, t)
            push!(solution, copy(y))
            save_idx += 1
            continue
        end

        if use_sparse
            y_new, err = solve_step_sparse(solver, f, y, t, h, params,
                                            J_sparsity, jacobian_sparse!)
        else
            y_new, err = solve_step_dense(solver, f, y, t, h, params, jacobian!)
        end

        scale = solver.abstol .+ solver.reltol .* max.(abs.(y), abs.(y_new))
        err_norm = sqrt(mean((err ./ scale).^2))

        if isnan(err_norm) || isinf(err_norm)
            h *= 0.1
            if h < 1e-15
                y = y_new
                t += h
                n_steps += 1
            end
            continue
        end

        if err_norm < 1.0
            y = y_new
            t += h
            n_steps += 1
        end

        h *= min(5.0, max(0.2, 0.9 * (1.0 / max(err_norm, 1e-30))^(1/3)))
    end

    return times, solution
end

end
