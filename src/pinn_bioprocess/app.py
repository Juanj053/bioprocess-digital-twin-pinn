"""
Streamlit Web Dashboard for the PINN Bioprocess Digital Twin.
Interactive real-time visualization of hybrid gray-box state estimation for a 20 L bioreactor.
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pinn_bioprocess.core.kinetics import BioprocessParameters
from pinn_bioprocess.core.mass_balance import check_carbon_balance
from pinn_bioprocess.core.simulator import BioreactorSimulator
from pinn_bioprocess.models.neural_pinn import BioprocessPINN
from pinn_bioprocess.training.metrics import compute_regression_metrics
from pinn_bioprocess.training.trainer import BioprocessPINNTrainer

st.set_page_config(
    page_title="PINN Bioprocess Digital Twin | Elevate Biotech",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Bioprocess Digital Twin: Physics-Informed Neural Network (PINN)")
st.caption("Hybrid Gray-Box Soft Sensor for 20 L Aerobic Stirred Tank Fermentation • Elevate Biotech")

# Sidebar Controls
st.sidebar.header("⚙️ Bioreactor Operational Controls")
v_liq = st.sidebar.slider("Working Volume [L]", 10.0, 25.0, 20.0, 0.5)
agitation = st.sidebar.slider("Agitation Speed [RPM]", 200, 800, 450, 25)
aeration = st.sidebar.slider("Aeration Rate [vvm]", 0.5, 2.5, 1.2, 0.1)
kla = st.sidebar.slider("Mass Transfer Coeff kLa [h⁻¹]", 20.0, 120.0, 65.0, 5.0)
s0 = st.sidebar.slider("Initial Substrate S₀ [g/L]", 15.0, 60.0, 35.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.header("🧠 PINN Hyperparameters")
epochs = st.sidebar.select_slider("Training Epochs", options=[500, 1000, 2000, 4000], value=2000)
lr = st.sidebar.selectbox("Learning Rate", options=[0.001, 0.002, 0.005], index=1)
n_collocation = st.sidebar.slider("Physics Collocation Points", 50, 300, 150, 25)

params = BioprocessParameters(
    kla=kla,
    S0=s0,
    t_final_h=40.0,
)

# Simulation & Training Workflow
sim = BioreactorSimulator(params)
gt, obs = sim.generate_multirate_sensors(random_seed=42)

run_button = st.sidebar.button("🚀 Run Digital Twin Training", type="primary")

if "trained_model" not in st.session_state or run_button:
    with st.spinner("Training Physics-Informed Neural Network (evaluating autograd ODE residuals)..."):
        model = BioprocessPINN(hidden_dim=64, num_layers=3, t_max=40.0)
        trainer = BioprocessPINNTrainer(
            model=model,
            params=params,
            n_collocation=n_collocation,
            lambda_phys=0.05,
            lambda_ic=0.15,
            lambda_balance=0.01,
        )
        fit_res = trainer.fit(obs, epochs=epochs, lr=lr, verbose=False)
        pred = trainer.predict(gt["t"])
        metrics = compute_regression_metrics(gt, pred)
        c_bal = check_carbon_balance(params.S0, pred["S"][-1], params.X0, pred["X"][-1], params.P0, pred["P"][-1])

        st.session_state["trained_model"] = model
        st.session_state["trainer"] = trainer
        st.session_state["pred"] = pred
        st.session_state["metrics"] = metrics
        st.session_state["fit_res"] = fit_res
        st.session_state["c_bal"] = c_bal

pred = st.session_state["pred"]
metrics = st.session_state["metrics"]
fit_res = st.session_state["fit_res"]
c_bal = st.session_state["c_bal"]

# Metric Cards
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
m_col1.metric("Biomass R²", f"{metrics['Biomass_X']['r2']:.4f}", f"MAE {metrics['Biomass_X']['mae']:.3f} g/L")
m_col2.metric("Substrate R²", f"{metrics['Substrate_S']['r2']:.4f}", f"MAE {metrics['Substrate_S']['mae']:.3f} g/L")
m_col3.metric("Dissolved O₂ R²", f"{metrics['Oxygen_CL']['r2']:.4f}", f"MAE {metrics['Oxygen_CL']['mae']:.3f} mg/L")
m_col4.metric("Product R²", f"{metrics['Product_P']['r2']:.4f}", f"MAE {metrics['Product_P']['mae']:.3f} g/L")
m_col5.metric("Training Time", f"{fit_res['training_time_seconds']:.2f} s", f"Epochs {epochs}")

st.markdown("---")

# Main Multi-Panel Plotly Figure
fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Biomass Growth X(t) [g/L]",
        "Substrate Consumption S(t) [g/L]",
        "Dissolved Oxygen CL(t) [mg/L]",
        "Product Formation P(t) [g/L]",
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

# Colors
c_gt = "#6c757d"
c_pinn = "#0d6efd"
c_sensor = "#dc3545"

# Panel 1: Biomass
fig.add_trace(go.Scatter(x=gt["t"], y=gt["X"], mode="lines", name="Ground Truth (Radau)", line=dict(color=c_gt, dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=pred["t"], y=pred["X"], mode="lines", name="PINN State Estimate", line=dict(color=c_pinn, width=2.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=obs["t_offline"], y=obs["X_obs"], mode="markers", name="Offline Lab Assay", marker=dict(color=c_sensor, size=7)), row=1, col=1)

# Panel 2: Substrate
fig.add_trace(go.Scatter(x=gt["t"], y=gt["S"], mode="lines", name="Ground Truth (Radau)", line=dict(color=c_gt, dash="dash"), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=pred["t"], y=pred["S"], mode="lines", name="PINN State Estimate", line=dict(color="#198754", width=2.5), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=obs["t_offline"], y=obs["S_obs"], mode="markers", name="HPLC Glucose", marker=dict(color=c_sensor, size=7), showlegend=False), row=1, col=2)

# Panel 3: Dissolved Oxygen
fig.add_trace(go.Scatter(x=gt["t"], y=gt["CL"] * 1000.0, mode="lines", name="Ground Truth (Radau)", line=dict(color=c_gt, dash="dash"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=pred["t"], y=pred["CL"] * 1000.0, mode="lines", name="PINN State Estimate", line=dict(color="#0dcaf0", width=2.5), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=obs["t_online"], y=obs["CL_obs"] * 1000.0, mode="markers", name="Online DO Probe", marker=dict(color="#ffc107", size=5), showlegend=False), row=2, col=1)

# Panel 4: Product
fig.add_trace(go.Scatter(x=gt["t"], y=gt["P"], mode="lines", name="Ground Truth (Radau)", line=dict(color=c_gt, dash="dash"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=pred["t"], y=pred["P"], mode="lines", name="PINN State Estimate", line=dict(color="#6f42c1", width=2.5), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=obs["t_offline"], y=obs["P_obs"], mode="markers", name="HPLC Product", marker=dict(color=c_sensor, size=7), showlegend=False), row=2, col=2)

fig.update_xaxes(title_text="Fermentation Time [h]")
fig.update_layout(height=650, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# Convergence & Stoichiometry Tabs
t1, t2 = st.tabs(["📉 Training Loss Convergence", "⚖️ Carbon Balance & Stoichiometry"])

with t1:
    h = fit_res["history"]
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=h["epoch"], y=h["total_loss"], mode="lines+markers", name="Total Loss L_total"))
    fig_loss.add_trace(go.Scatter(x=h["epoch"], y=h["l_data"], mode="lines", name="Sensor Loss L_data"))
    fig_loss.add_trace(go.Scatter(x=h["epoch"], y=h["l_phys"], mode="lines", name="Physics Collocation L_phys"))
    fig_loss.add_trace(go.Scatter(x=h["epoch"], y=h["l_ic"], mode="lines", name="Initial Condition L_ic"))
    fig_loss.update_yaxes(type="log", title_text="Loss (Log Scale)")
    fig_loss.update_xaxes(title_text="Epoch")
    fig_loss.update_layout(height=380, template="plotly_white")
    st.plotly_chart(fig_loss, use_container_width=True)

with t2:
    st.markdown(f"""
    - **Initial Glucose $S_0$**: `{params.S0:.2f} g/L` | **Final Residual Substrate $S_f$**: `{pred['S'][-1]:.2f} g/L`
    - **Biomass Formed $\\Delta X$**: `{pred['X'][-1] - params.X0:.2f} g/L` | **Product Formed $\\Delta P$**: `{pred['P'][-1] - params.P0:.2f} g/L`
    - **Total Carbon Consumed**: `{c_bal['c_consumed_g_C_L']:.2f} g C/L`
    - **Carbon Fixed in Biomass & Product**: `{c_bal['c_fixed_biomass_product_g_C_L']:.2f} g C/L`
    - **Carbon Incorporation Fraction**: `{c_bal['carbon_incorporation_fraction'] * 100:.1f} %` *(Remainder emitted as metabolic CO₂ respiration)*
    """)
