# <p align="center">bioprocess-digital-twin-pinn</p>
<p align="center">
  <strong>Physics-Informed Neural Network (PINN) Bioprocess Digital Twin & Gray-Box Soft Sensor</strong><br>
  <em>Continuous State Estimation, Multi-Rate Sensor Fusion & Autograd Kinetic Inversion for 20 L Aerobic Stirred Tank Fermentation</em><br>
  Developed by <a href="https://www.linkedin.com/in/juanj-alva">Ing. Juan José Álvarez Morales</a> • <strong>Elevate Biotech</strong>
</p>

<p align="center">
  <a href="https://github.com/Juanj053/bioprocess-digital-twin-pinn/actions"><img src="https://img.shields.io/badge/CI-passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://www.linkedin.com/in/juanj-alva"><img src="https://img.shields.io/badge/LinkedIn-Juan%20Jos%C3%A9%20%C3%81lvarez-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/Bioreactor%20Scale-20%20L%20Stirred%20Tank-blueviolet?style=for-the-badge" alt="Scale">
  <img src="https://img.shields.io/badge/Accuracy-R%C2%B2%20%3E%200.997-success?style=for-the-badge" alt="Accuracy">
</p>

---

## 🔬 Executive Overview & The Gray-Box Paradigm

In modern precision biomanufacturing (recombinant therapeutic peptides, industrial enzymes, cell-free biopolymers), biological reactors operate under **highly non-linear, non-stationary regimes**. Traditional computational approaches face an engineering dilemma:

1. **Pure White-Box Mechanistic Models (ODEs):** Captures first-principles thermodynamics and microbial growth kinetics (Monod, Pirt, mass transfer). However, they suffer from batch-to-batch biological drift, parameter sensitivity, and rigid mathematical structures incapable of absorbing live sensor anomalies.
2. **Pure Black-Box Machine Learning (LSTMs, XGBoost):** Fits historical time-series data flexibly, but demands massive experimental training datasets, lacks thermodynamic explainability, violates stoichiometric mass balances (predicting negative concentrations or physically impossible yields), and fails catastrophically when extrapolating outside the calibration envelope.

**`bioprocess-digital-twin-pinn`** introduces a **Hybrid Gray-Box Digital Twin** powered by **Physics-Informed Neural Networks (PINNs)**. By embedding coupled ordinary differential equations (ODEs) directly into the neural network's loss functional via reverse-mode automatic differentiation (`torch.autograd`), the model:
- Reconstructs unmeasured continuous biological trajectories from **sparse, noisy, multi-rate physical sensors**.
- Strictly satisfies **stoichiometric carbon conservation** and **biological non-negativity** constraints ($X, S, C_L, P \ge 0$).
- Functions as an online **soft sensor** for automated Process Analytical Technology (PAT) and digital twin telemetry.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               HYBRID GRAY-BOX PINN DIGITAL TWIN ARCHITECTURE (20 L BATCH)              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
    [ PHYSICAL SENSOR DATA ]                                  [ FIRST-PRINCIPLES ODEs ]
    • Offline Lab HPLC: S(t), P(t)                            • Dual Monod Growth: μ(S, CL)
    • Offline Dry Cell Weight: X(t)                           • Pirt Substrate Balance: dS/dt
    • Online Optical Probe: CL(t)                             • Two-Film Mass Transfer: kla(C* - CL)
               │                                              • Luedeking-Piret Product: dP/dt
               ▼                                                         │
       ┌───────────────┐                                                 ▼
       │ L_data (MSE)  │                                       ┌───────────────────┐
       └───────┬───────┘                                       │ L_phys (Autograd) │
               │                                               └─────────┬─────────┘
               └────────────────────────────┬────────────────────────────┘
                                            ▼
                     TOTAL LOSS: L = L_data + λ_phys·L_phys + λ_ic·L_ic + λ_bal·L_balance
                                            │
                                            ▼
                   PyTorch MLP with Tanh Activations & Softplus Positivity
                                            │
                                            ▼
           Continuous State Trajectory: ŷ(t) = [ X(t), S(t), CL(t), P(t) ]  (R² > 0.997)
```

---

## 📐 Mathematical Formulation: Coupled Bioreactor ODE System

The state of the 20 L aerobic stirred-tank fermentation is governed by the state vector:
$$\mathbf{y}(t) = \begin{bmatrix} X(t) \\ S(t) \\ C_L(t) \\ P(t) \end{bmatrix} \in \mathbb{R}_+^4, \quad t \in [0, t_f], \quad t_f = 40.0 \text{ h}$$

### 1. Dual Monod Specific Growth Rate ($\mu$)
$$\mu(S, C_L) = \mu_{\max} \cdot \left( \frac{S}{K_S + S} \right) \cdot \left( \frac{C_L}{K_{O_2} + C_L} \right)$$
* $\mu_{\max} = 0.35 \text{ h}^{-1}$, $K_S = 2.0 \text{ g/L}$, $K_{O_2} = 0.0008 \text{ g/L}$ ($0.8 \text{ mg/L}$).

### 2. Coupled Dynamic Mass Balances
$$\frac{dX}{dt} = (\mu - k_d) X$$
$$\frac{dS}{dt} = - \left( \frac{\mu}{Y_{X/S}} + m_S + \frac{\alpha \mu + \beta}{Y_{P/S}} \right) X$$
$$\frac{dC_L}{dt} = k_L a \cdot (C^* - C_L) - \left( \frac{\mu}{Y_{X/O_2}} + m_{O_2} \right) X$$
$$\frac{dP}{dt} = (\alpha \mu + \beta) X$$

### 3. Bioreactor Scale & Mass Transfer Parameters (20 L Vessel)
| Parameter | Value | Unit | Engineering Description |
| :--- | :---: | :---: | :--- |
| **$V_L$** | `20.0` | $\text{L}$ | Working operational broth volume |
| **$V_T$** | `28.0` | $\text{L}$ | Total tank geometry (aspect ratio $H/D \approx 2.5$) |
| **Agitation** | `450` | $\text{RPM}$ | Dual 6-blade Rushton impellers ($D_i/D_t \approx 0.33$) |
| **Aeration** | `1.2` | $\text{vvm}$ | Gas flow rate ($24 \text{ L}_{\text{gas}}/\text{min} = 1.44 \text{ m}^3/\text{h}$) |
| **$k_L a$** | `65.0` | $\text{h}^{-1}$ | Volumetric oxygen mass transfer coefficient |
| **$C^*$** | `0.0078` | $\text{g/L}$ | Saturated dissolved oxygen solubility ($7.8 \text{ mg/L}$ at $30^\circ\text{C}$, $1.05 \text{ bar}$) |
| **$Y_{X/S}$** | `0.48` | $\text{g}_X/\text{g}_S$ | True stoichiometric biomass yield on carbon |
| **$m_S$** | `0.020` | $\text{g}_S/(\text{g}_X\cdot\text{h})$ | Pirt cellular maintenance energy coefficient |
| **$\alpha, \beta$** | `0.15, 0.010` | $-$ | Luedeking-Piret product synthesis constants |

---

## 🧠 Physics-Informed Neural Network Architecture

The neural network acts as a continuous non-linear interpolator mapping dimensionless time $\tau = t / t_{\max} \in [0, 1]$ to normalized state variables:

$$\hat{\mathbf{y}}_{\boldsymbol{\theta}}(\tau) = \text{Softplus}\left( \text{MLP}_{\boldsymbol{\theta}}(\tau) \right)$$

1. **$C^\infty$ Smooth Activations:** Uses `Tanh` activation across hidden layers. Unlike piecewise-linear functions (`ReLU`) whose second derivatives vanish everywhere ($\nabla^2 \text{ReLU} \equiv 0$), `Tanh` provides continuous, non-zero gradient landscapes for high-order autograd differentiation.
2. **Biological Positivity Guarantee:** The final output is transformed via `Softplus(z) = \ln(1 + e^z) \ge 0`, preventing negative physical concentrations.
3. **Multi-Objective Loss Formulation:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{ic}} \mathcal{L}_{\text{ic}} + \lambda_{\text{phys}} \mathcal{L}_{\text{collocation}} + \lambda_{\text{balance}} \mathcal{L}_{\text{carbon}}$$

$$\mathcal{L}_{\text{collocation}} = \frac{1}{N_c} \sum_{i=1}^{N_c} \left[ \left(\frac{d\hat{X}}{dt} - f_X\right)^2 + \left(\frac{d\hat{S}}{dt} - f_S\right)^2 + \left(\frac{d\hat{C}_L}{dt} - f_{CL}\right)^2 + \left(\frac{d\hat{P}}{dt} - f_P\right)^2 \right]$$

---

## 📊 Benchmark Validation & Convergence Telemetry

Evaluated against high-precision stiff Radau numerical integration ($N=401$ dense points) on an Apple Silicon CPU:

| State Variable | Physical Dimension | Ground Truth Range | PINN Metric ($R^2$) | Mean Absolute Error (MAE) | Root Mean Sq Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Biomass $X(t)$** | $\text{g/L}$ | $0.80 \to 10.95$ | **`0.9974`** | `0.1351 g/L` | `0.1653 g/L` |
| **Substrate $S(t)$** | $\text{g/L}$ | $35.00 \to 0.00$ | **`0.9970`** | `0.5419 g/L` | `0.6601 g/L` |
| **Dissolved $O_2$ $C_L(t)$** | $\text{mg/L}$ | $7.80 \to 1.15$ | **`0.9723`** | `0.2000 mg/L` | `0.5131 mg/L` |
| **Product $P(t)$** | $\text{g/L}$ | $0.00 \to 3.12$ | **`0.9975`** | `0.0493 g/L` | `0.0687 g/L` |

> ⚡ **Execution Speed:** Complete training (4,000 epochs, 150 collocation points) converges in **`7.00 seconds`** on CPU using Adam with Cosine Annealing learning rate scheduling.

---

## 🚀 Quick Start & Installation

### Option A: Using `uv` (Recommended)
```bash
# Clone the repository
git clone https://github.com/Juanj053/bioprocess-digital-twin-pinn.git
cd bioprocess-digital-twin-pinn

# Sync virtual environment and run the test suite
uv sync
uv run pytest -v tests/
```

### Option B: Using Standard `pip`
```bash
pip install -e .
pytest -v tests/
```

---

## 💻 Usage & CLI Interface

### 1. Train PINN Digital Twin via CLI
```bash
# Train PINN on synthetic 20L bioreactor multi-rate telemetry
pinn-bioprocess train --duration 40.0 --epochs 4000 --lr 0.002 --output results/metrics.json
```

### 2. Export Pure Mechanistic ODE Simulation
```bash
# Generate high-resolution ODE ground truth trajectory
pinn-bioprocess simulate --duration 40.0 --points 500 --output results/bioreactor_trajectory.csv
```

### 3. Launch Interactive Streamlit Telemetry Dashboard
```bash
uv run streamlit run src/pinn_bioprocess/app.py
```
*Opens an interactive multi-panel dashboard with Plotly charts, live operational sliders ($k_L a$, RPM, $S_0$), loss curves, and stoichiometric carbon recovery metrics.*

---

## 📂 Repository Structure

```
bioprocess-digital-twin-pinn/
├── .github/workflows/ci.yml       # Automated CI across Python 3.10, 3.11, 3.12
├── config/
│   └── bioreactor_20l.yaml        # Declarative vessel, kinetic & PINN configurations
├── src/
│   └── pinn_bioprocess/
│       ├── core/
│       │   ├── kinetics.py        # Dual Monod, Pirt, Luedeking-Piret, and mass transfer ODEs
│       │   ├── simulator.py       # Stiff Radau solver & multi-rate sensor noise generator
│       │   └── mass_balance.py    # Carbon incorporation and macroscopic yield auditor
│       ├── models/
│       │   └── neural_pinn.py     # PyTorch BioprocessPINN with C^inf Tanh & Softplus
│       ├── training/
│       │   ├── trainer.py         # Autograd collocation loss engine & Adam optimization
│       │   └── metrics.py         # R², MAE, and RMSE calculation routines
│       ├── app.py                 # Interactive Streamlit dashboard with Plotly telemetry
│       └── cli.py                 # Command-line interface (train, simulate)
├── tests/
│   ├── test_kinetics.py           # Unit tests for rate equations and ODE derivatives
│   ├── test_simulator.py          # Stiff integration and multi-rate noise testing
│   ├── test_pinn_architecture.py  # Autograd flow, output shapes, and positivity
│   └── test_training_loss.py      # Collocation loss verification and short fit loop
├── pyproject.toml                 # Modern PEP 517/621 packaging (Hatchling)
├── CITATION.cff                   # Academic citation metadata
├── LICENSE                        # MIT License
└── README.md                      # Engineering and scientific documentation
```

---

## 📜 Citation

If you utilize this digital twin architecture in your research, academic publications, or industrial biomanufacturing scale-up protocols, please cite:

```bibtex
@software{alvarez_morales_2026_bioprocess_pinn,
  author       = {Álvarez Morales, Juan José},
  title        = {bioprocess-digital-twin-pinn: Physics-Informed Neural Network Digital Twin for Aerobic Bioreactor State Estimation},
  month        = oct,
  year         = 2026,
  publisher    = {GitHub},
  version      = {0.1.0},
  url          = {https://github.com/Juanj053/bioprocess-digital-twin-pinn}
}
```

---

## 👨‍🔬 Author & Academic Inquiries

**Ing. Juan José Álvarez Morales**  
*Biotechnology Engineer (Universidad Técnica de Ambato)*  
*Founder & Chief Bioengineer at Elevate Biotech • Research Lead at Agromanper*  
Ambato, Ecuador 🇪🇨  

- **LinkedIn:** [linkedin.com/in/juanj-alva](https://www.linkedin.com/in/juanj-alva)  
- **Email:** [jjalvarezm77@gmail.com](mailto:jjalvarezm77@gmail.com)  
- **GitHub:** [@Juanj053](https://github.com/Juanj053)  
- **Research Collaboration:** Open for CYTED, NIH/Fogarty, and UK academic partnerships in computational bioprocess digital twins and therapeutic peptide discovery.
