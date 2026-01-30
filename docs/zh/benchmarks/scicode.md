# SciCode


## 概述

SciCode 是一个具有挑战性的基准测试，旨在评估语言模型在生成代码以解决现实科学研究问题方面的能力。它涵盖 5 个主要领域（物理学、数学、材料科学、生物学和化学）中的 16 个子领域。

## 任务描述

- **任务类型**：科学代码生成
- **输入**：包含多个子问题的多步骤科学问题
- **输出**：解决每个子问题的 Python 代码
- **领域**：物理学、数学、材料科学、生物学、化学

## 主要特点

- 问题源自真实科研场景
- 覆盖 5 个科学领域的 16 个子领域
- 多步骤问题，需结合知识回忆、推理与综合能力
- 可选提供科学背景信息
- 需要沙箱环境以安全执行代码

## 评估说明

- **必须使用沙箱环境**执行代码
- 请参考 [沙箱文档](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/sandbox.html)
- 跟踪两个指标：
  - `main_problem_pass_rate`：完整问题解决率
  - `subproblem_pass_rate`：各子问题成功率
- 通过配置 `provide_background` 参数决定是否包含科学背景上下文
- 使用自定义 Docker 镜像：`scicode-benchmark:latest`


## 属性

| 属性 | 值 |
|----------|-------|
| **基准测试名称** | `scicode` |
| **数据集 ID** | [evalscope/SciCode](https://modelscope.cn/datasets/evalscope/SciCode/summary) |
| **论文** | N/A |
| **标签** | `Coding` |
| **指标** | `main_problem_pass_rate`, `subproblem_pass_rate` |
| **默认示例数** | 0-shot |
| **评估划分** | `test` |


## 数据统计

| 指标 | 值 |
|--------|-------|
| 总样本数 | 65 |
| 提示词长度（平均） | 1055.94 字符 |
| 提示词长度（最小/最大） | 1055 / 1056 字符 |

## 样例示例

**子集**: `default`

```json
{
  "input": [
    {
      "id": "127a601e",
      "content": "\nPROBLEM DESCRIPTION:\nYou will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier su ... [TRUNCATED] ... olve it as follows:\n 1. Generate scientific background required for the next step, in a comment\n 2. Implement a function to solve the problem provided, using the provided header and return line\n\nThe response must be formatted as ```python```\n"
    },
    {
      "id": "a3effde1",
      "content": "77"
    }
  ],
  "id": 0,
  "group_id": 0,
  "metadata": {
    "problem_name": "Berendsen_thermostat",
    "problem_id": "77",
    "problem_description_main": "Write a Script to integrate the Berendsen thermalstat and barostat into molecular dynamics calculation through velocity Verlet algorithm. The particles are placed in a periodic cubic system, interacting with each other through truncated and s ... [TRUNCATED] ... nd positions of particles in our simulation to control the system's temperature and pressure, respectively. The implementation should enable switching the thermostat and barostat on or off with a condition on their respective time constants.\n",
    "problem_background_main": "",
    "problem_io": "\"\"\"\nIntegrate the equations of motion using the velocity Verlet algorithm, with the inclusion of the Berendsen thermostat\nand barostat for temperature and pressure control, respectively.\n\nParameters:\nN : int\n    The number of particles in the ... [TRUNCATED] ... ometers/ps.\nL : float\n    Updated length of the cubic simulation box's side, units: nanometers.\n\nRaises:\n-------\nException:\n    If the Berendsen barostat has shrunk the box such that the side length L is less than twice the cutoff radius.\n\"\"\"",
    "required_dependencies": "import math\nimport numpy as np\nimport scipy as sp\nfrom scipy.constants import  Avogadro",
    "sub_steps": [
      {
        "step_number": "77.1",
        "step_description_prompt": "Wrap to periodic boundaries\nImplementing a Python function named `wrap`. This function should apply periodic boundary conditions to the coordinates of a particle inside a cubic simulation box.",
        "step_background": "Background:\nTo implement PBC, the unit cell is surrounded by translated copies in all directions to approximate an infinitely large system. When one molecule diffuses across the boundary of the simulation box it reappears on the opposite side. So each molecule always interacts with its neighbours even though they may be on opposite sides of the simulation box",
        "ground_truth_code": null,
        "function_header": "def wrap(r, L):\n    '''Apply periodic boundary conditions to a vector of coordinates r for a cubic box of size L.\n    Parameters:\n    r : The (x, y, z) coordinates of a particle.\n    L (float): The length of each side of the cubic box.\n    Returns:\n    coord: numpy 1d array of floats, the wrapped coordinates such that they lie within the cubic box.\n    '''",
        "test_cases": [
          "particle_position = np.array([10.5, -1.2, 20.3])\nbox_length = 5.0\n# Applying the wrap function\nassert np.allclose(wrap(particle_position, box_length), target)",
          "particle_position1 = np.array([10.0, 5.5, -0.1])\nbox_length1 = 10.0\n# Applying the wrap function\nassert np.allclose(wrap(particle_position1, box_length1), target)",
          "particle_position2 = np.array([23.7, -22.1, 14.3])\nbox_length2 = 10.0\n# Applying the wrap function\nassert np.allclose(wrap(particle_position2, box_length2), target)"
        ],
        "return_line": "    return coord"
      },
      {
        "step_number": "77.2",
        "step_description_prompt": "Minimum Image Distance Function\n\nImplementing Python function named `dist` that calculates the minimum image distance between two atoms in a periodic cubic system.",
        "step_background": "Background:\nThe function should implement the minimum image convention, which is used in molecular dynamics simulations to consider the shortest distance between periodic images of particles.",
        "ground_truth_code": null,
        "function_header": "def dist(r1, r2, L):\n    '''Calculate the minimum image distance between two atoms in a periodic cubic system.\n    Parameters:\n    r1 : The (x, y, z) coordinates of the first atom.\n    r2 : The (x, y, z) coordinates of the second atom.\n    L (float): The length of the side of the cubic box.\n    Returns:\n    float: The minimum image distance between the two atoms.\n    '''",
        "test_cases": [
          "r1 = np.array([2.0, 3.0, 4.0])\nr2 = np.array([2.5, 3.5, 4.5])\nbox_length = 10.0\nassert np.allclose(dist(r1, r2, box_length), target)",
          "r1 = np.array([1.0, 1.0, 1.0])\nr2 = np.array([9.0, 9.0, 9.0])\nbox_length = 10.0\nassert np.allclose(dist(r1, r2, box_length), target)",
          "r1 = np.array([0.1, 0.1, 0.1])\nr2 = np.array([9.9, 9.9, 9.9])\nbox_length = 10.0\nassert np.allclose(dist(r1, r2, box_length), target)"
        ],
        "return_line": "    return distance"
      },
      {
        "step_number": "77.3",
        "step_description_prompt": "Minimum Image Vector Function\n\nImplementing Python function named `dist_v` that calculates the minimum image vector between two atoms in a periodic cubic system.",
        "step_background": "Background:\nThe function should implement the minimum image convention, which is used in molecular dynamics simulations to consider the shortest distance between periodic images of particles.",
        "ground_truth_code": null,
        "function_header": "def dist_v(r1, r2, L):\n    '''Calculate the minimum image vector between two atoms in a periodic cubic system.\n    Parameters:\n    r1 : The (x, y, z) coordinates of the first atom.\n    r2 : The (x, y, z) coordinates of the second atom.\n    L (float): The length of the side of the cubic box.\n    Returns:\n    float: The minimum image distance between the two atoms.\n    '''",
        "test_cases": [
          "r1 = np.array([2.0, 3.0, 4.0])\nr2 = np.array([2.5, 3.5, 4.5])\nbox_length = 10.0\nassert np.allclose(dist_v(r1, r2, box_length), target)",
          "r1 = np.array([1.0, 1.0, 1.0])\nr2 = np.array([9.0, 9.0, 9.0])\nbox_length = 10.0\nassert np.allclose(dist_v(r1, r2, box_length), target)",
          "r1 = np.array([0.1, 0.1, 0.1])\nr2 = np.array([9.9, 9.9, 9.9])\nbox_length = 10.0\nassert np.allclose(dist_v(r1, r2, box_length), target)"
        ],
        "return_line": "    return r12"
      },
      {
        "step_number": "77.4",
        "step_description_prompt": "Lennard-Jones Potential\n\nImplementing a Python function named `E_ij` to get Lennard-Jones potential with potential well depth epislon that reaches zero at distance sigma between pair of atoms with distance r. which is truncated and shifted to zero at a cutoff distance `rc`.",
        "step_background": "Background\nThe Lennard-Jones potential models soft repulsive and attractive (van der Waals) interactions. Hence, the Lennard-Jones potential describes electronically neutral atoms or molecules. The commonly used expression for the Lennard-Jon ... [TRUNCATED] ... article potential energy V is zero (often referred to as 'size of the particle').\n\nThe potential is truncated and shifted at a distance $ r_c $ to ensure the interaction energy becomes zero for $ r > r_c $, simplifying the force computations.",
        "ground_truth_code": null,
        "function_header": "def E_ij(r, sigma, epsilon, rc):\n    '''Calculate the combined truncated and shifted Lennard-Jones potential energy between two particles.\n    Parameters:\n    r (float): The distance between particles i and j.\n    sigma (float): The distance  ... [TRUNCATED] ... Jones potential.\n    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.\n    Returns:\n    float: The combined potential energy between the two particles, considering the specified potentials.\n    '''",
        "test_cases": [
          "r1 = 1.0  # Close to the sigma value\nsigma1 = 1.0\nepsilon1 = 1.0\nrc = 1\nassert np.allclose(E_ij(r1, sigma1, epsilon1, rc), target)",
          "r2 = 0.5  # Significantly closer than the effective diameter\nsigma2 = 1.0\nepsilon2 = 1.0\nrc = 2\nassert np.allclose(E_ij(r2, sigma2, epsilon2, rc), target)",
          "r3 = 2.0  # Larger than sigma\nsigma3 = 1.0\nepsilon3 = 1.0\nrc = 3\nassert np.allclose(E_ij(r3, sigma3, epsilon3, rc), target)"
        ],
        "return_line": "    return E"
      },
      {
        "step_number": "77.5",
        "step_description_prompt": "Lennard-Jones Force\n\n Based on Lennard-Jones potential with potential well depth epislon that reaches zero at distance sigma, write a function that calculates the forces between two particles whose three dimensional displacement is r.",
        "step_background": "Background\nTo get force, we just use the negative gradiant of Lennard-Jones potential (by definition):\n\n$\\vec{F}=-\\frac{\\partial V}{\\partial \\vec{r}}=-\\left(\\frac{\\partial V}{\\partial x} ; \\frac{\\partial V}{\\partial y} ; \\frac{\\partial V}{\\partial z}\\right)$",
        "ground_truth_code": null,
        "function_header": "def f_ij(r, sigma, epsilon, rc):\n    '''Calculate the force vector between two particles, considering the truncated and shifted\n    Lennard-Jones potential.\n    Parameters:\n    r (float): The distance between particles i and j.\n    sigma (flo ... [TRUNCATED] ... ential.\n    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.\n    Returns:\n    array_like: The force vector experienced by particle i due to particle j, considering the specified potentials\n    '''",
        "test_cases": [
          "sigma = 1\nepsilon = 1\nr = np.array([-3.22883506e-03,  2.57056485e+00,  1.40822287e-04])\nrc = 2\nassert np.allclose(f_ij(r,sigma,epsilon,rc), target)",
          "sigma = 2\nepsilon = 1\nr = np.array([3,  -4,  5])\nrc = 10\nassert np.allclose(f_ij(r,sigma,epsilon,rc), target)",
          "sigma = 3\nepsilon = 1\nr = np.array([5,  9,  7])\nrc = 20\nassert np.allclose(f_ij(r,sigma,epsilon,rc), target)"
        ],
        "return_line": "    return f"
      },
      {
        "step_number": "77.6",
        "step_description_prompt": "Tail Corrections for Energy with LJ\n\nImplementing Python functions named `E_tail` to calculate the tail correction  for a system of particles within a cubic simulation box. This correction accounts for the truncation of the Lennard-Jones potentials at a specific cutoff distance.",
        "step_background": "Background\n\nIn molecular dynamics simulations, long-range interactions are often neglected beyond a cutoff radius $ r_c $. To estimate the contribution of these neglected interactions to the system's energy, tail correction is applied. The energy tail correction per particle is given by:\n\n$\nu^{\\textbf{tail LJ}}_{i} = \\frac{8}{3} \\pi N^2 \\epsilon \\sigma^3 \\left[ \\frac{1}{3} \\left( \\frac{\\sigma}{r_c} \\right)^9 - \\left( \\frac{\\sigma}{r_c} \\right)^3 \\right]\n$",
        "ground_truth_code": null,
        "function_header": "def E_tail(N, L, sigma, epsilon, rc):\n    '''Calculate the energy tail correction for a system of particles, considering the truncated and shifted\n    Lennard-Jones potential.\n    Parameters:\n    N (int): The total number of particles in the  ... [TRUNCATED] ...     rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.\n    Returns:\n    float\n        The energy tail correction for the entire system (in zeptojoules), considering the specified potentials.\n    '''",
        "test_cases": [
          "N=2\nL=10\nsigma = 1\nepsilon = 1\nrc = 1\nassert np.allclose(E_tail(N,L,sigma,epsilon,rc), target)",
          "N=5\nL=10\nsigma = 1\nepsilon = 1\nrc = 5\nassert np.allclose(E_tail(N,L,sigma,epsilon,rc), target)",
          "N=10\nL=10\nsigma = 1\nepsilon = 1\nrc = 9\nassert np.allclose(E_tail(N,L,sigma,epsilon,rc), target)"
        ],
        "return_line": "    return E_tail_LJ"
      },
      {
        "step_number": "77.7",
        "step_description_prompt": "Tail Corrections for Pressure with LJ\n\nImplementing Python functions named `P_tail` to calculate the tail correction for a system of particles within a cubic simulation box. This correction accounts for the truncation of the Lennard-Jones potentials at a specific cutoff distance.",
        "step_background": "Background\n\nIn molecular dynamics simulations, long-range interactions are often neglected beyond a cutoff radius $ r_c $. To estimate the contribution of these neglected interactions to the system's pressure, tail correction is applied. The pressure tail correction for the system, considering all particles, is:\n\n$\np^{\\text{tail LJ}} = \\frac{16}{3} \\pi N^2 \\epsilon \\sigma^3 \\left[ \\frac{2}{3} \\left( \\frac{\\sigma}{r_c} \\right)^9 - \\left( \\frac{\\sigma}{r_c} \\right)^3 \\right]\n$",
        "ground_truth_code": null,
        "function_header": "def P_tail(N, L, sigma, epsilon, rc):\n    ''' Calculate the pressure tail correction for a system of particles, including\n     the truncated and shifted Lennard-Jones contributions.\n    P arameters:\n     N (int): The total number of particles ... [TRUNCATED] ...  for the Lennard-Jones potential.\n     rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.\n     Returns:\n     float\n         The pressure tail correction for the entire system (in bar).\n     \n    '''",
        "test_cases": [
          "N=2\nL=10\nsigma = 1\nepsilon = 1\nrc = 1\nassert np.allclose(P_tail(N,L,sigma,epsilon,rc), target)",
          "N=5\nL=10\nsigma = 1\nepsilon = 1\nrc = 5\nassert np.allclose(P_tail(N,L,sigma,epsilon,rc), target)",
          "N=10\nL=10\nsigma = 1\nepsilon = 1\nrc = 9\nassert np.allclose(P_tail(N,L,sigma,epsilon,rc), target)"
        ],
        "return_line": "    return P_tail_bar"
      },
      {
        "step_number": "77.8",
        "step_description_prompt": "Potential Energy\nImplementing a Python function named `E_pot` to calculate the total potential energy of a system of particles.",
        "step_background": "Background\n\nThe pairwise potential energy $ E_{ij} $ for particles separated by a distance less than the cutoff radius $ r_c $ is calculated using the `E_ij` function, which should be provided. A helper function `dist` should be used to calculate the distance between two particles, applying the minimum image convention.",
        "ground_truth_code": null,
        "function_header": "def E_pot(xyz, L, sigma, epsilon, rc):\n    '''Calculate the total potential energy of a system using the truncated and shifted Lennard-Jones potential.\n    Parameters:\n    xyz : A NumPy array with shape (N, 3) where N is the number of particl ... [TRUNCATED] ... tential well for the Lennard-Jones potential.\n    rc (float): The cutoff distance beyond which the potentials are truncated and shifted to zero.\n    Returns:\n    float\n        The total potential energy of the system (in zeptojoules).\n    '''",
        "test_cases": [
          "positions1 = np.array([[1, 1, 1], [1.1, 1.1, 1.1]])\nL1 = 10.0\nsigma1 = 1.0\nepsilon1 = 1.0\nrc=5\nassert np.allclose(E_pot(positions1, L1, sigma1, epsilon1,rc), target)",
          "positions2 = np.array([[1, 1, 1], [1, 9, 1], [9, 1, 1], [9, 9, 1]])\nL2 = 10.0\nsigma2 = 1.0\nepsilon2 = 1.0\nrc=5\nassert np.allclose(E_pot(positions2, L2, sigma2, epsilon2,rc), target)",
          "np.random.seed(0)\npositions3 = np.random.rand(10, 3) * 10  # 10 particles in a 10x10x10 box\nL3 = 10.0\nsigma3 = 1.0\nepsilon3 = 1.0\nrc=5\nassert np.allclose(E_pot(positions3, L3, sigma3, epsilon3,rc), target)"
        ],
        "return_line": "    return E"
      },
      {
        "step_number": "77.9",
        "step_description_prompt": "Temperature Calculation\n\nImplement Python function to calculate instantaneous temperature of a system of particles in molecular dynamics simulation. The temperature function, named `temperature`, should use the kinetic energy to determine the instantaneous temperature of the system according to the equipartition theorem, with the temperature returned in Kelvin. Note that the Boltzmann constant $k_B$ is 0.0138064852 zJ/K.",
        "step_background": "Background\n\nTemperature Calculation via Equipartition Theorem\n\nFor the `temperature` function, the temperature is calculated using the relationship:\n\n$$ E_{kinetic} = \\frac{3}{2}k_BT $$\n\nWhere \\(E_{kinetic}\\) is the average kinetic energy per ... [TRUNCATED] ...  particle is given by:\n\n$$ E_{kinetic} = \\frac{1}{2}mv^2 $$\n\nWhere $v^2 = v_x^2 + v_y^2 + v_z^2$ is the square of the velocity vector. Remember to account for Avogadro's number when converting from molar mass to the mass of a single particle.",
        "ground_truth_code": null,
        "function_header": "def temperature(v_xyz, m, N):\n    '''Calculate the instantaneous temperature of a system of particles using the equipartition theorem.\n    Parameters:\n    v_xyz : ndarray\n        A NumPy array with shape (N, 3) containing the velocities of ea ... [TRUNCATED] ... oat\n        The molar mass of the particles in the system, in grams per mole (g/mol).\n    N : int\n        The number of particles in the system.\n    Returns:\n    float\n        The instantaneous temperature of the system in Kelvin (K).\n    '''",
        "test_cases": [
          "v=np.array([1,2,3])\nm=1\nN=1\nassert np.allclose(temperature(v,m,N), target)",
          "v=np.array([[1,2,3],[1,1,1]])\nm=10\nN=2\nassert np.allclose(temperature(v,m,N), target)",
          "v=np.array([[1,2,3],[4,6,8],[6,1,4]])\nm=100\nN=3\nassert np.allclose(temperature(v,m,N), target)"
        ],
        "return_line": "    return T"
      },
      {
        "step_number": "77.10",
        "step_description_prompt": "Pressure Calculation Using Virial Equation\n\nImplementing a Python function named `pressure` to calculate the pressure of a molecular system using the virial equation. Note that the Boltzmann constant $k_B$ is 0.0138064852 zJ/K.",
        "step_background": "Background\nThe function should compute the pressure `P` using the formula:\n\n$ P_{kinetec} = \\frac{Nk_BT}{V} $\n\n$ P_{virial} = \\frac{1}{3V} \\sum_{\\text{pairs } i,j} \\mathbf{f}_{ij} \\cdot \\mathbf{r}_{ij} $\n\n$ P = P_{kinetec} + P_{virial} $\n\nwhe ... [TRUNCATED] ... particle \\( i \\) exerted by particle \\( j \\) (as computed by the `f_ij` function previously written), and $ \\mathbf{r}_{ij} $ is the displacement vector from particle \\( i \\) to \\( j \\).\nThe function should return the pressure in units of bar",
        "ground_truth_code": null,
        "function_header": "def pressure(N, L, T, xyz, sigma, epsilon, rc):\n    '''Calculate the pressure of a system of particles using the virial theorem, considering\n    the Lennard-Jones contributions.\n    Parameters:\n    N : int\n        The number of particles in t ... [TRUNCATED] ... ich the inter-particle potential is considered to be zero (in nanometers).\n    Returns:\n    tuple\n        The kinetic pressure (in bar), the virial pressure (in bar), and the total pressure (kinetic plus virial, in bar) of the system.\n    '''",
        "test_cases": [
          "from scicode.compare.cmp import cmp_tuple_or_list\nN = 2\nL = 10\nsigma = 1\nepsilon = 1\npositions = np.array([[3,  -4,  5],[0.1, 0.5, 0.9]])\nrc = 1\nT=300\nassert cmp_tuple_or_list(pressure(N, L, T, positions, sigma, epsilon, rc), target)",
          "from scicode.compare.cmp import cmp_tuple_or_list\nN = 2\nL = 10\nsigma = 1\nepsilon = 1\npositions = np.array([[.62726631, 5.3077771 , 7.29719649],\n       [2.25031287, 8.58926428, 4.71262908],\n          [3.62726631, 1.3077771 , 2.29719649]])\nrc = 2\nT=1\nassert cmp_tuple_or_list(pressure(N, L, T, positions, sigma, epsilon, rc), target)",
          "from scicode.compare.cmp import cmp_tuple_or_list\nN = 5\nL = 10\nsigma = 1\nepsilon = 1\npositions = np.array([[.62726631, 5.3077771 , 7.29719649],\n       [7.25031287, 7.58926428, 2.71262908],\n       [8.7866416 , 3.73724676, 9.22676027],\n       [0.89096788, 5.3872004 , 7.95350911],\n       [6.068183  , 3.55807037, 2.7965242 ]])\nrc = 3\nT=200\nassert cmp_tuple_or_list(pressure(N, L, T, positions, sigma, epsilon, rc), target)"
        ],
        "return_line": "    return P_kinetic, P_virial, P_kinetic + P_virial"
      },
      {
        "step_number": "77.11",
        "step_description_prompt": "Forces Calculation Function\n\nImplementing Python function titled `forces` that calculates the forces on each particle due to pairwise interactions with all its neighbors in a molecular simulation.  This function should compute the net force on each particle and return a NumPy array `f_xyz` of the same shape as `xyz`, where each element is the force vector (in zeptojoules per nanometer) for the corresponding particle.",
        "step_background": "Background\nNewton's third law:\nForce on j due to i is negative of i due to j",
        "ground_truth_code": null,
        "function_header": "def forces(N, xyz, L, sigma, epsilon, rc):\n    '''Calculate the net forces acting on each particle in a system due to all pairwise interactions.\n    Parameters:\n    N : int\n        The number of particles in the system.\n    xyz : ndarray\n     ... [TRUNCATED] ...  the inter-particle forces are considered negligible.\n    Returns:\n    ndarray\n        A NumPy array of shape (N, 3) containing the net force vectors acting on each particle in the system,\n        in zeptojoules per nanometer (zJ/nm).\n    '''",
        "test_cases": [
          "N = 2\nL = 10\nsigma = 1\nepsilon = 1\npositions = np.array([[3,  -4,  5],[0.1, 0.5, 0.9]])\nrc = 1\nassert np.allclose(forces(N, positions, L, sigma, epsilon, rc), target)",
          "N = 2\nL = 10\nsigma = 1\nepsilon = 1\npositions = np.array([[.62726631, 5.3077771 , 7.29719649],\n       [2.25031287, 8.58926428, 4.71262908],\n          [3.62726631, 1.3077771 , 2.29719649]])\nrc = 9\nassert np.allclose(forces(N, positions, L, sigma, epsilon, rc), target)",
          "N = 5\nL = 10\nsigma = 1\nepsilon = 1\npositions = np.array([[.62726631, 5.3077771 , 7.29719649],\n       [7.25031287, 7.58926428, 2.71262908],\n       [8.7866416 , 3.73724676, 9.22676027],\n       [0.89096788, 5.3872004 , 7.95350911],\n       [6.068183  , 3.55807037, 2.7965242 ]])\nrc = 3\nassert np.allclose(forces(N, positions, L, sigma, epsilon, rc), target)"
        ],
        "return_line": "    return f_xyz"
      },
      {
        "step_number": "77.12",
        "step_description_prompt": "Berendsen Thermostat and Barostat Integration into Velocity Verlet Algorithm\n\nWrite a fuction to integrate the Berendsen thermalstat and barostat into molecular dynamics calculation through velocity Verlet algorithm. The Berendsen thermalstat ... [TRUNCATED] ... and positions of particles in our simulation to control the system's temperature and pressure, respectively. The implementation should enable switching the thermostat and barostat on or off with a condition on their respective time constants.",
        "step_background": "Background\n\nThe Berendsen thermostat is described by the following additional force term:\n\n$$ f^{\\text{Berendsen}}_{i} = \\frac{m_i}{2\\tau_T} \\left( \\frac{T_{\\text{target}}}{T(t)} - 1 \\right) \\mathbf{v}_i(t), $$\n\nwhere $T_{\\text{target}}$ is t ... [TRUNCATED] ... ght) \\Delta t$.\n3. Derive $\\mathbf{a}(t+\\Delta t)$ from the interaction potential using $\\mathbf{x}(t+\\Delta t)$.\n4. Calculate $\\mathbf{v}(t+\\Delta t)=\\mathbf{v}\\left(t+\\frac{1}{2} \\Delta t\\right)+\\frac{1}{2} \\mathbf{a}(t+\\Delta t) \\Delta t$.",
        "ground_truth_code": null,
        "function_header": "def velocityVerlet(N, xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target):\n    '''Integrate the equations of motion using the velocity Verlet algorithm, with the inclusion of the Berendsen thermostat\n    and barostat f ... [TRUNCATED] ...        Updated length of the cubic simulation box's side, units: nanometers.\n    Raises:\n    -------\n    Exception:\n        If the Berendsen barostat has shrunk the box such that the side length L is less than twice the cutoff radius.\n    '''",
        "test_cases": [
          "np.random.seed(17896)\n# NPT simulation\nT_target = 298 # K\nP_target = 200 # bar\nL = 2.4 # nm\nN = 100\ndt = 0.005 # ps\nnSteps = 1200\nrc = 0.8 # nm\nprintModulus = 1 # steps\nsigma = 0.34 # nm\nepsilon = 1.65 # zJ\ntau_T = 0.1 # ps\ntau_P = 0.01 # ps\n ... [TRUNCATED] ... xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target, nSteps)\nthreshold = 0.3\nassert (np.abs(np.mean(T_sim-T_target)/T_target)<threshold and np.abs(np.mean(P_sim[int(0.2*nSteps):]-P_target)/P_target)<threshold) == target"
        ],
        "return_line": "    return xyz_full, v_xyz_full, L"
      }
    ],
    "general_solution": null,
    "general_tests": [
      "np.random.seed(17896)\n# NPT simulation\nT_target = 298 # K\nP_target = 200 # bar\nL = 2.4 # nm\nN = 100\ndt = 0.005 # ps\nnSteps = 1200\nrc = 0.8 # nm\nprintModulus = 1 # steps\nsigma = 0.34 # nm\nepsilon = 1.65 # zJ\ntau_T = 0.1 # ps\ntau_P = 0.01 # ps\n ... [TRUNCATED] ... xyz, v_xyz, L, sigma, epsilon, rc, m, dt, tau_T, T_target, tau_P, P_target, nSteps)\nthreshold = 0.3\nassert (np.abs(np.mean(T_sim-T_target)/T_target)<threshold and np.abs(np.mean(P_sim[int(0.2*nSteps):]-P_target)/P_target)<threshold) == target"
    ]
  }
}
```

*注：部分内容因显示原因已被截断。*

## 提示模板

**系统提示：**
```text

PROBLEM DESCRIPTION:
You will be provided with a description of a scientific problem. You will solve these problems by solving a sequence of *subproblems*. The solution to each subproblem may be implemented using your solutions to earlier subproblems. Each subproblem should be solved by providing a Python function that meets the specifications provided.

For each subproblem, you will be provided with the following
 1. a description of the subproblem
 2. a function header, which you must use in your solution implementation
 3. a return line, which you must use in your solution implementation

You must only use the following dependencies to implement your solution:
{required_dependencies}

You MUST NOT import these dependencies anywhere in the code you generate.

For each subproblem provided you must solve it as follows:
 1. Generate scientific background required for the next step, in a comment
 2. Implement a function to solve the problem provided, using the provided header and return line

The response must be formatted as ```python```

```

**提示模板：**
```text

Implement code to solve the following subproblem, using the description, function header, and return line provided.

Remember that you may use functions that you generated previously as solutions to previous subproblems to implement your answer.

Remember that you MUST NOT include code to import dependencies.

Remember to ensure your response is in the format of ```python``` and includes necessary background as a comment at the top.

SUBPROBLEM DESCRIPTION:
{step_description_prompt}

FUNCTION HEADER:
{function_header}

RETURN LINE:
{return_line}

Example:
```python
# Background: [Here, insert the necessary scientific knowledge required for the next step.]

[Insert the Python code here based on the provided function header and dependencies.]
```

```

## 额外参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `provide_background` | `bool` | `False` | 在模型提示中包含科学家撰写的科学背景信息。 |

## 沙箱配置

此基准测试需要沙箱环境来执行代码。

```json
{
  "image": "scicode-benchmark:latest",
  "tools_config": {
    "shell_executor": {},
    "python_executor": {}
  }
}
```

## 使用方法

### 使用 CLI

```bash
evalscope eval \
    --model YOUR_MODEL \
    --api-url OPENAI_API_COMPAT_URL \
    --api-key EMPTY_TOKEN \
    --datasets scicode \
    --use-sandbox \
    --limit 10  # 正式评估时请移除此行
```

### 使用 Python

```python
from evalscope import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='YOUR_MODEL',
    api_url='OPENAI_API_COMPAT_URL',
    api_key='EMPTY_TOKEN',
    datasets=['scicode'],
    use_sandbox=True,
    dataset_args={
        'scicode': {
            # extra_params: {}  # 使用默认额外参数
        }
    },
    limit=10,  # 正式评估时请移除此行
)

run_task(task_cfg=task_cfg)
```