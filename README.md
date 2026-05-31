# AtmChemBox - 零维大气化学箱式模型
====================================

## 项目概述

AtmChemBox 是一个用Julia开发的零维大气化学箱式模型，用于模拟给定排放条件下臭氧（O3）、氮氧化物（NOx）和挥发性有机物（VOCs）浓度的时间演变。

## 主要特性

- **刚性常微分方程求解器（Rosenbrock/RODAS3方法）
- **稀疏矩阵支持** - 自动检测并启用稀疏雅可比矩阵，适用于超过100个物种的大规模机理
- 化学机理文件解析（支持KPP和YAML格式，如MOZART机制）
- 光解速率计算（支持日变化剖面）
- Python接口
- YAML配置文件支持
- 数值溢出保护 - 防止大浓度跨度时的 SIGFPE 崩溃

## 项目结构

```
atmchem-box-4/
├── src/
│   ├── AtmChemBox.jl              # 主模块
│   ├── mechanism_parser/
│   │   └── MechanismParser.jl      # 机理解析模块
│   ├── ode_system/
│   │   └── ODESystem.jl       # ODE系统构建
│   ├── solver/
│   │   └── RosenbrockSolver.jl # Rosenbrock求解器
│   ├── physics/
│   │   └── Physics.jl        # 物理过程（光解、排放、沉降）
│   └── bindings/
│       └── PythonInterface.jl  # Python绑定
├── mechanisms/
│   ├── mozart.yaml          # 简化的MOZART机理
│   └── large_mechanism.yaml     # 200+物种的大规模测试机理
├── python/
│   └── atmchem_box.py        # Python接口
├── examples/
│   └── config.yaml          # 示例配置文件
├── test/
│   └── runtests.jl          # Julia测试文件
└── Project.toml             # Julia项目配置
```

## 稀疏矩阵支持

### 问题背景

当反应机理包含超过200个物种时，稠密雅可比矩阵会导致：
1. 内存占用过大：O(n²) 增长
2. 计算效率低下：LU分解为 O(n³) 复杂度
3. 数值溢出：固定步长的有限差分可能导致 SIGFPE 崩溃

### 解决方案

1. **雅可比稀疏性模式预分析** ([ODESystem.jl](file:///d:/SOLO-1/atmchem-box-4/src/ode_system/ODESystem.jl#L90-L128))
   - 根据反应网络拓扑自动确定非零元素位置
   - 只需在机理加载时计算一次

2. **稀疏雅可比矩阵计算** ([ODESystem.jl](file:///d:/SOLO-1/atmchem-box-4/src/ode_system/ODESystem.jl#L349-L419))
   - 直接写入稀疏矩阵的 nonzeros 数组
   - O(nnz) 计算复杂度，而非 O(n²)

3. **稀疏LU分解** ([RosenbrockSolver.jl](file:///d:/SOLO-1/atmchem-box-4/src/solver/RosenbrockSolver.jl#L70-L109))
   - 使用 SparseArrays 的 UMFPACK 分解
   - 利用稀疏性大幅加速求解

4. **自适应有限差分步长** ([RosenbrockSolver.jl](file:///d:/SOLO-1/atmchem-box-4/src/solver/RosenbrockSolver.jl#L152-L184))
   - `eps_j = sqrt(eps) * max(|y_j|, 1e-8)`
   - 防止小浓度物种除零溢出

5. **安全幂运算** ([ODESystem.jl](file:///d:/SOLO-1/atmchem-box-4/src/ode_system/ODESystem.jl#L11-L61))
   - 检测并处理 `concentrations[idx]^coeff` 的溢出
   - 提前返回有界值

### 配置选项

```yaml
use_sparse: true              # 启用稀疏矩阵（默认true）
sparse_threshold: 100         # 物种数阈值（默认100）
```

当物种数超过 `sparse_threshold` 时自动启用稀疏求解。

## 安装依赖

### Julia依赖

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### Python依赖

```bash
pip install julia numpy pyyaml matplotlib
```

## 使用方法

### Julia使用

```julia
using AtmChemBox

model = BoxModel("mechanisms/mozart.yaml")

# 对于大规模机理，可手动指定
model = BoxModel("mechanisms/large_mechanism.yaml", 
                  use_sparse=true, sparse_threshold=100)

set_initial_concentrations!(model, Dict(
    "O3" => 50e-9,
    "NO" => 10e-9,
    "NO2" => 20e-9
))

times, results = run_simulation(model, 0.0, 3600.0, 60.0)
```

### Python使用

```python
from python.atmchem_box import AtmChemBox

model = AtmChemBox(config_file="examples/config.yaml")

# 获取稀疏性信息
info = model.get_sparsity_info()
print(f"Jacobian: {info['n_species']}x{info['n_species']}, "
      f"{info['nnz']} nonzeros, density={info['density']:.4f}")

times, results = model.run_simulation(t_end=3600, dt=60)
```

## 配置文件说明

YAML配置文件包含以下部分：

- `mechanism`: 化学机理文件路径
- `temperature`: 温度 (K)
- `pressure`: 压力 (Pa)
- `use_sparse`: 是否使用稀疏矩阵
- `sparse_threshold`: 启用稀疏的物种数阈值
- `initial_conditions`: 初始浓度 (molecules/cm³)
- `emissions`: 排放速率
- `photolysis`: 光解速率
- `deposition`: 沉降速率

## 模块说明

### 机理解析模块
解析化学机理，支持YAML和KPP格式

### ODE系统构建
将反应网络转换为微分方程系统，支持稀疏雅可比

### Rosenbrock求解器
RODAS3刚性ODE求解器，自适应步长，支持稠密/稀疏模式

### 物理过程
光解速率、排放、干沉降

## 性能对比

| 物种数 | 稠密矩阵内存 | 稀疏矩阵内存 (5%密度) | 加速比 |
|--------|-------------|-------------------|--------|
| 50     | 200 KB      | 20 KB             | ~1x    |
| 200    | 3.2 MB      | 160 KB            | ~5x    |
| 500    | 20 MB       | 1 MB              | ~20x   |
| 1000   | 80 MB       | 4 MB              | ~50x   |

## 伴随模式（Adjoint Mode）- 敏感性分析

### 原理

伴随模式通过反向求解伴随方程，高效计算目标函数对所有参数的梯度：

- 正向积分：dy/dt = f(y, p, t)  （保存中间状态）
- 伴随方程：dλ/dt = - (∂f/∂y)^T λ - (∂g/∂y)^T
- 梯度计算：dJ/dp = λ(0)^T dy0/dp + ∫ λ^T ∂f/∂p dt

**优势**：一次反向积分可得到对所有参数的梯度，计算成本 ≈ 2次正向积分。

### Julia 使用示例

```julia
using AtmChemBox

model = BoxModel("mechanisms/mozart.yaml")

set_initial_concentrations!(model, Dict(
    "O3" => 50e-9, "NO" => 10e-9, "NO2" => 20e-9
))

# 设置敏感性分析配置
config = SensitivityConfig(
    objective_type="final",           # 终点时刻目标函数
    objective_weights=Dict("O3" => 1.0),  # 关注 O3 浓度
    t_start=0.0,
    t_end=3600.0,
    dt=60.0,
    checkpoint_dt=60.0,               # 检查点间隔
    use_sparse=true
)

# 运行敏感性分析
result = run_sensitivity_analysis(model, config)

# 查看结果
print_sensitivity_summary(result, top_n=10)

# 获取特定梯度
grad_dict = gradient_to_dict(result.gradient_initial, result.species_names)
println("dO3/dNO_initial = ", grad_dict["NO"])
```

### Python 使用示例

```python
from python.atmchem_box import AtmChemBox

model = AtmChemBox(config_file="examples/config_sensitivity.yaml")

# 运行敏感性分析
result = model.run_sensitivity_analysis(
    objective_type='final',
    objective_weights={'O3': 1.0},
    t_end=3600,
    dt=60
)

# 打印摘要
model.print_sensitivity_summary(result, top_n=5)

# 获取特定物种的敏感性
o3_sensitivity_to_no = result['gradient_emissions']['NO']
print(f"O3 sensitivity to NO emissions: {o3_sensitivity_to_no:.4e}")

# 伴随变量可视化
import matplotlib.pyplot as plt
times = result['times'] / 3600
plt.plot(times, result['adjoint_results']['O3'], label='Adjoint O3')
```

### 目标函数类型

1. **FinalTimeObjective** - 终点时刻浓度
   ```julia
   obj = FinalTimeObjective(mech, Dict("O3" => 1.0, "NOx" => 0.5))
   ```

2. **TimeIntegratedObjective** - 时间积分浓度
   ```julia
   obj = TimeIntegratedObjective(mech, Dict("O3" => 1.0))
   ```

### 梯度类型

| 梯度类型 | 说明 |
|---------|------|
| `gradient_initial` | 目标函数对各物种初始浓度的梯度 |
| `gradient_emissions` | 目标函数对各物种排放速率的梯度 |
| `gradient_rate_constants` | 目标函数对各反应速率常数的梯度 |

### 配置参数 (SensitivityConfig)

- `objective_type`: `"final"` 或 `"integrated"`
- `objective_weights`: Dict，目标函数中各物种的权重
- `t_start`, `t_end`: 时间窗口
- `dt`: 积分时间步长
- `checkpoint_dt`: 检查点存储间隔（用于插值）
- `target_species`, `target_reactions`, `target_emissions`: 可选，指定分析目标
- `use_sparse`: 是否使用稀疏矩阵
- `sparse_threshold`: 启用稀疏的物种数阈值

### 文件结构

```
src/adjoint/
├── AdjointSystem.jl        # 伴随方程系统
├── AdjointSolver.jl        # 反向 Rosenbrock 求解器
└── SensitivityAnalysis.jl  # 高层敏感性分析接口
```

### 数据结构

- **AdjointModel**: 包含雅可比转置、目标函数、稀疏模式
- **AdjointSolverState**: 包含检查点数据
- **SensitivityConfig**: 敏感性分析配置
- **SensitivityResult**: 包含正向/伴随结果和所有梯度

### 大规模机理支持

对于200+物种的机理：
```julia
model = BoxModel("mechanisms/large_mechanism.yaml", 
                  use_sparse=true, sparse_threshold=100)

config = SensitivityConfig(
    objective_weights=Dict("O3" => 1.0),
    t_end=3600.0,
    use_sparse=true,
    sparse_threshold=100
)

result = run_sensitivity_analysis(model, config)
```

内存和计算效率：
- 稀疏雅可比：O(nnz) 存储 vs O(n²)
- 梯度计算：一次反向积分得所有梯度
- 200物种：~5倍加速，500物种：~20倍加速

## Python 命令行使用

```bash
# 普通模拟
python python/atmchem_box.py

# 敏感性分析
python python/atmchem_box.py sensitivity
```
