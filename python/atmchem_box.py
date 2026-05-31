import julia
from julia.api import Julia
import numpy as np
import yaml
import os

class AtmChemBox:
    def __init__(self, config_file=None, mechanism_file=None,
                 use_sparse=True, sparse_threshold=100):
        self.jl = Julia(compiled_modules=False)
        self.jl.eval('using Pkg; Pkg.activate(".")')
        self.jl.eval('using AtmChemBox')

        self.config_file = config_file
        self.mechanism_file = mechanism_file
        self.use_sparse = use_sparse
        self.sparse_threshold = sparse_threshold

        if config_file:
            self.config = self._load_config(config_file)
            mechanism_file = self.config.get('mechanism', 'mechanisms/mozart.yaml')
            self.use_sparse = self.config.get('use_sparse', use_sparse)
            self.sparse_threshold = self.config.get('sparse_threshold', sparse_threshold)

        if mechanism_file:
            sp_flag = 'true' if self.use_sparse else 'false'
            self.jl.eval(f'model = BoxModel("{mechanism_file}", use_sparse={sp_flag}, sparse_threshold={int(self.sparse_threshold)})')
            if config_file:
                self._apply_config()

    def _load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _apply_config(self):
        initial_conds = self.config.get('initial_conditions', {})
        self.set_initial_concentrations(initial_conds)

        emissions = self.config.get('emissions', {})
        self.set_emissions(emissions)

        photolysis = self.config.get('photolysis', {})
        if photolysis:
            self.set_photolysis_rates(photolysis)

        deposition = self.config.get('deposition', {})
        if deposition:
            self.set_deposition_rates(deposition)

    def set_initial_concentrations(self, conc_dict):
        for name, value in conc_dict.items():
            self.jl.eval(f'set_initial_concentrations!(model, Dict("{name}" => {float(value)}))')

    def set_emissions(self, emis_dict):
        for name, value in emis_dict.items():
            self.jl.eval(f'set_emissions!(model.emissions, Dict("{name}" => {float(value)}))')

    def set_photolysis_rates(self, photo_dict):
        for name, value in photo_dict.items():
            self.jl.eval(f'set_photolysis_rates!(model.photolysis, Dict("{name}" => {float(value)}))')

    def set_deposition_rates(self, dep_dict):
        for name, value in dep_dict.items():
            self.jl.eval(f'set_deposition_rates!(model.deposition, Dict("{name}" => {float(value)}))')

    def set_temperature(self, T):
        self.jl.eval(f'model.temperature = {float(T)}')

    def run_simulation(self, t_start=0.0, t_end=3600.0, dt=60.0):
        result = self.jl.eval(f'run_simulation(model, {float(t_start)}, {float(t_end)}, {float(dt)})')
        times = np.array(result[0])
        species_names = self.get_species_names()
        results_dict = {name: np.array(result[1][:, i]) for i, name in enumerate(species_names)}
        return times, results_dict

    def get_species_names(self):
        return self.jl.eval('[s.name for s in model.mechanism.species]')

    def get_reactions(self):
        return self.jl.eval('[r.id for r in model.mechanism.reactions]')

    def get_concentrations(self):
        return np.array(self.jl.eval('model.concentrations'))

    def get_sparsity_info(self):
        n = self.jl.eval('model.ode.n_species')
        nnz = self.jl.eval('nnz(model.J_sparsity)')
        density = nnz / (n * n) if n > 0 else 0
        return {'n_species': n, 'nnz': nnz, 'density': density}

    def run_sensitivity_analysis(self, objective_type='final',
                                   objective_weights=None,
                                   t_start=0.0, t_end=3600.0, dt=60.0,
                                   checkpoint_dt=60.0,
                                   target_species=None,
                                   target_reactions=None,
                                   target_emissions=None,
                                   use_sparse=None,
                                   sparse_threshold=None):
        if objective_weights is None:
            objective_weights = {'O3': 1.0}

        weights_str = ', '.join([f'"{k}" => {float(v)}' for k, v in objective_weights.items()])

        target_species = target_species or []
        target_reactions = target_reactions or []
        target_emissions = target_emissions or []

        species_str = '[' + ', '.join([f'"{s}"' for s in target_species]) + ']'
        reactions_str = '[' + ', '.join([f'"{r}"' for r in target_reactions]) + ']'
        emissions_str = '[' + ', '.join([f'"{e}"' for e in target_emissions]) + ']'

        use_sparse = use_sparse if use_sparse is not None else self.use_sparse
        sparse_threshold = sparse_threshold if sparse_threshold is not None else self.sparse_threshold

        sp_flag = 'true' if use_sparse else 'false'

        jl_cmd = f'''
        result = run_sensitivity_analysis_py(
            BoxModelPython(model, Dict()),
            objective_type="{objective_type}",
            objective_weights=Dict({weights_str}),
            t_start={float(t_start)},
            t_end={float(t_end)},
            dt={float(dt)},
            checkpoint_dt={float(checkpoint_dt)},
            target_species={species_str},
            target_reactions={reactions_str},
            target_emissions={emissions_str},
            use_sparse={sp_flag},
            sparse_threshold={int(sparse_threshold)}
        )
        result
        '''

        result = self.jl.eval(jl_cmd)

        # Convert Julia arrays to numpy arrays
        result['times'] = np.array(result['times'])
        result['adjoint_times'] = np.array(result['adjoint_times'])

        for key in ['forward_results', 'adjoint_results']:
            if key in result:
                result[key] = {k: np.array(v) for k, v in result[key].items()}

        return result

    def print_sensitivity_summary(self, result, top_n=10):
        print("=" * 60)
        print("Sensitivity Analysis Summary")
        print("=" * 60)
        print(f"Objective value: {result['objective_value']:.6e}")
        print()

        grad_initial = result['gradient_initial']
        grad_emissions = result['gradient_emissions']
        grad_rates = result['gradient_rate_constants']

        print(f"Top {top_n} initial condition sensitivities:")
        sorted_initial = sorted(grad_initial.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_initial[:top_n]:
            if abs(val) > 1e-20:
                print(f"  {name:15s}: {val:+.4e}")
        print()

        print(f"Top {top_n} emission sensitivities:")
        sorted_emis = sorted(grad_emissions.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, val in sorted_emis[:top_n]:
            if abs(val) > 1e-20:
                print(f"  {name:15s}: {val:+.4e}")
        print()

        if grad_rates:
            print(f"Top {top_n} rate constant sensitivities:")
            sorted_rxn = sorted(grad_rates.items(), key=lambda x: abs(x[1]), reverse=True)
            for name, val in sorted_rxn[:top_n]:
                if abs(val) > 1e-20:
                    print(f"  {name:15s}: {val:+.4e}")

        print("=" * 60)

def run_example():
    import matplotlib.pyplot as plt

    model = AtmChemBox(config_file='examples/config.yaml')

    info = model.get_sparsity_info()
    print(f"Jacobian sparsity: {info['n_species']}x{info['n_species']} matrix, "
          f"{info['nnz']} nonzeros, density={info['density']:.4f}")

    times, results = model.run_simulation(t_end=3*3600, dt=60)

    plt.figure(figsize=(12, 6))
    plt.plot(times/3600, results['O3']*1e9, label='O3')
    plt.plot(times/3600, results['NO']*1e9, label='NO')
    plt.plot(times/3600, results['NO2']*1e9, label='NO2')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (ppb)')
    plt.legend()
    plt.title('Atmospheric Chemistry Box Model Simulation')
    plt.grid(True)
    plt.savefig('simulation_results.png')
    plt.close()

    print("Simulation completed. Results saved to simulation_results.png")

def run_sensitivity_example():
    import matplotlib.pyplot as plt

    model = AtmChemBox(config_file='examples/config.yaml')

    print("\nRunning sensitivity analysis for O3 concentration...")
    result = model.run_sensitivity_analysis(
        objective_type='final',
        objective_weights={'O3': 1.0},
        t_end=3*3600,
        dt=60
    )

    model.print_sensitivity_summary(result, top_n=5)

    grad_initial = result['gradient_initial']
    o3_sensitivity = grad_initial.get('NOx', 0) if 'NOx' in grad_initial else grad_initial.get('NO', 0) + grad_initial.get('NO2', 0)
    print(f"\nO3 sensitivity to initial NOx: {o3_sensitivity:.4e}")
    print(f"O3 sensitivity to NO emissions: {result['gradient_emissions'].get('NO', 0):.4e}")

    # Plot adjoint variables
    times = result['times'] / 3600
    adjoint_o3 = result['adjoint_results']['O3']
    adjoint_no2 = result['adjoint_results']['NO2']

    plt.figure(figsize=(12, 6))
    plt.plot(times, adjoint_o3, label='Adjoint O3')
    plt.plot(times, adjoint_no2, label='Adjoint NO2')
    plt.xlabel('Time (hours)')
    plt.ylabel('Adjoint Variable')
    plt.legend()
    plt.title('Adjoint Variables for O3 Sensitivity')
    plt.grid(True)
    plt.savefig('adjoint_results.png')
    plt.close()

    print("\nAdjoint analysis completed. Results saved to adjoint_results.png")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'sensitivity':
        run_sensitivity_example()
    else:
        run_example()
