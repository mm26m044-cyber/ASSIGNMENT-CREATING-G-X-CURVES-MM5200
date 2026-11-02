"""
Scientifically Accurate Determination and Plotting of the Binodal Regime in Si-Ge.

This script demonstrates two fundamental thermodynamic perspectives:
1. Liquid-Solid Binodal Regime (High-T Phase Diagram):
   - First Derivative condition:
       (dG_S / dx)|_{x_S} = (dG_L / dx)|_{x_L} = (G_L(x_L) - G_S(x_S)) / (x_L - x_S) = m_tangent
   - More accurate & numerically robust method:
       Simultaneous equality of chemical potentials:
       mu_Si^S(x_S) = mu_Si^L(x_L)
       mu_Ge^S(x_S) = mu_Ge^L(x_L)
   - Visualized via:
       (a) G-x curves with Common Tangent
       (b) dG/dx curves vs x showing the horizontal slope matching
       (c) T-x Phase diagram highlighting the complete Binodal coexistence regime.

2. Solid-State Miscibility Gap Binodal & Spinodal (Subcritical Low-T Regime):
   - For a regular solution with Omega_S = 3850 J/mol, Tc = Omega_S / (2*R) = 231.5 K (-41.6 °C).
   - Binodal (first derivative condition):
       d(Delta G_m) / dx = 0  =>  R*T*ln(x / (1-x)) + Omega_S*(1 - 2*x) = 0
   - Spinodal (second derivative condition):
       d^2(Delta G_m) / dx^2 = 0  =>  x*(1-x) = R*T / (2*Omega_S)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root, root_scalar

# Matplotlib styling for publication-grade scientific plots
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.35,
})

# ==============================================================================
# THERMODYNAMIC PARAMETERS (SGTE / Dinsdale 1991 / Mullis 2003)
# ==============================================================================
R = 8.314462618  # Universal gas constant [J / (mol * K)]

# Pure Si
TM_SI = 1687.0    # K (1414 °C)
DH_SI = 50210.0   # J / mol

# Pure Ge
TM_GE = 1211.4    # K (938.25 °C)
DH_GE = 36940.0   # J / mol

# Regular solution interaction parameters [J / mol]
OMEGA_S = 3850.0  # Solid (Diamond cubic A4)
def get_omega_L(T):
    return 6610.0 - 0.354 * T  # Liquid CALPHAD parameter


# ==============================================================================
# GIBBS ENERGIES, FIRST DERIVATIVES & CHEMICAL POTENTIALS
# ==============================================================================
def delta_G_fusion(T):
    """Pure element lattice stabilities relative to solid: dG = dH * (1 - T/Tm)"""
    dG_Si = DH_SI * (1.0 - T / TM_SI)
    dG_Ge = DH_GE * (1.0 - T / TM_GE)
    return dG_Si, dG_Ge

def G_solid(x, T):
    """Molar Gibbs energy of Solid phase relative to pure solids [J/mol]."""
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    G_ideal = R * T * ((1.0 - x) * np.log(1.0 - x) + x * np.log(x))
    G_ex = OMEGA_S * x * (1.0 - x)
    return G_ideal + G_ex

def G_liquid(x, T):
    """Molar Gibbs energy of Liquid phase relative to pure solids [J/mol]."""
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    dG_Si, dG_Ge = delta_G_fusion(T)
    omega_L = get_omega_L(T)
    G_ref = (1.0 - x) * dG_Si + x * dG_Ge
    G_ideal = R * T * ((1.0 - x) * np.log(1.0 - x) + x * np.log(x))
    G_ex = omega_L * x * (1.0 - x)
    return G_ref + G_ideal + G_ex

def dG_solid_dx(x, T):
    """
    First derivative of G_solid with respect to x = x_Ge:
    dG^S / dx = mu_Ge^S - mu_Si^S = R*T*ln(x / (1-x)) + Omega_S * (1 - 2*x)
    """
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    return R * T * np.log(x / (1.0 - x)) + OMEGA_S * (1.0 - 2.0 * x)

def dG_liquid_dx(x, T):
    """
    First derivative of G_liquid with respect to x = x_Ge:
    dG^L / dx = mu_Ge^L - mu_Si^L = (dG_Ge - dG_Si) + R*T*ln(x / (1-x)) + Omega_L * (1 - 2*x)
    """
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    dG_Si, dG_Ge = delta_G_fusion(T)
    omega_L = get_omega_L(T)
    return (dG_Ge - dG_Si) + R * T * np.log(x / (1.0 - x)) + omega_L * (1.0 - 2.0 * x)

def chemical_potentials_solid(x, T):
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    mu_Si = R * T * np.log(1.0 - x) + OMEGA_S * (x ** 2)
    mu_Ge = R * T * np.log(x) + OMEGA_S * ((1.0 - x) ** 2)
    return mu_Si, mu_Ge

def chemical_potentials_liquid(x, T):
    x = np.clip(x, 1e-15, 1.0 - 1e-15)
    dG_Si, dG_Ge = delta_G_fusion(T)
    omega_L = get_omega_L(T)
    mu_Si = dG_Si + R * T * np.log(1.0 - x) + omega_L * (x ** 2)
    mu_Ge = dG_Ge + R * T * np.log(x) + omega_L * ((1.0 - x) ** 2)
    return mu_Si, mu_Ge


# ==============================================================================
# ACCURATE SOLVER FOR LIQUID-SOLID BINODAL (COMMON TANGENT)
# ==============================================================================
def solve_liquid_solid_binodal(T, x_init=None):
    """
    Computes binodal boundary points (x_S, x_L) at temperature T.
    Uses the exact thermodynamic condition:
        mu_Si^S(x_S, T) == mu_Si^L(x_L, T)
        mu_Ge^S(x_S, T) == mu_Ge^L(x_L, T)
    This is mathematically equivalent to the common tangent condition:
        (dG_S/dx)|_{x_S} == (dG_L/dx)|_{x_L} == (G_L - G_S)/(x_L - x_S)
    but numerically superior because it avoids division by (x_L - x_S).
    """
    if not (TM_GE < T < TM_SI):
        return None

    if x_init is None:
        tau = (TM_SI - T) / (TM_SI - TM_GE)
        x_init = [np.clip(tau * 0.7, 0.05, 0.85), np.clip(tau * 1.2, 0.15, 0.95)]

    def residual(p):
        x_S, x_L = p
        if not (1e-6 < x_S < 1 - 1e-6 and 1e-6 < x_L < 1 - 1e-6):
            return [1e6, 1e6]
        mu_Si_S, mu_Ge_S = chemical_potentials_solid(x_S, T)
        mu_Si_L, mu_Ge_L = chemical_potentials_liquid(x_L, T)
        return [mu_Si_S - mu_Si_L, mu_Ge_S - mu_Ge_L]

    sol = root(residual, x_init, method='hybr')
    if sol.success and (0 < sol.x[0] < sol.x[1] < 1):
        return sol.x[0], sol.x[1]

    # Fallback to bounded minimization
    from scipy.optimize import minimize
    def obj(p):
        r = residual(p)
        return r[0]**2 + r[1]**2
    res = minimize(obj, x_init, bounds=[(1e-4, 0.999), (1e-4, 0.999)])
    if res.success and (0 < res.x[0] < res.x[1] < 1):
        return res.x[0], res.x[1]

    return None


# ==============================================================================
# PLOT 1: G-x CURVE & FIRST DERIVATIVE (dG/dx) SIDE-BY-SIDE / STACKED
# ==============================================================================
def plot_binodal_first_derivative(T=1450.0, save_path="si_ge_binodal_derivative.png"):
    """
    Plots:
    Top Panel: G_m(x) for Solid and Liquid with Common Tangent, shading the Binodal Regime.
    Bottom Panel: First derivative dG/dx for both phases, showing the horizontal tie line
                  and highlighting why dG/dx = m determines the binodal, NOT dG/dx = 0.
    """
    x = np.linspace(0.005, 0.995, 1000)
    G_S = G_solid(x, T) / 1000.0  # kJ/mol
    G_L = G_liquid(x, T) / 1000.0

    dG_S = dG_solid_dx(x, T) / 1000.0  # kJ/mol
    dG_L = dG_liquid_dx(x, T) / 1000.0

    binodal = solve_liquid_solid_binodal(T)
    if binodal is None:
        raise ValueError(f"No two-phase equilibrium at T = {T} K")

    x_S, x_L = binodal
    G_S_eq = G_solid(x_S, T) / 1000.0
    G_L_eq = G_liquid(x_L, T) / 1000.0

    # Common tangent slope m = (G_L - G_S) / (x_L - x_S) [kJ/mol]
    m_tangent = (G_L_eq - G_S_eq) / (x_L - x_S)

    # Tangent line: y(x) = G_S_eq + m_tangent * (x - x_S)
    tangent_y = G_S_eq + m_tangent * (x - x_S)

    # Intercepts
    mu_Si_eq, mu_Ge_eq = chemical_potentials_solid(x_S, T)
    mu_Si_eq /= 1000.0
    mu_Ge_eq /= 1000.0

    fig, (ax_G, ax_dG) = plt.subplots(2, 1, figsize=(10, 10), sharex=True,
                                      gridspec_kw={'height_ratios': [1.2, 1.0], 'hspace': 0.12})

    # --- TOP PANEL: G - x and Common Tangent ---
    ax_G.plot(x, G_S, label=r'Solid Solution ($G^S$, Diamond $\alpha$)', color='#1f77b4', lw=2.5)
    ax_G.plot(x, G_L, label=r'Liquid Solution ($G^L$)', color='#d62728', lw=2.5)
    ax_G.plot(x, tangent_y, '--', color='#2ca02c', lw=2.2, label=f'Common Tangent (Slope $m = {m_tangent:.2f}$ kJ/mol)')

    # Mark binodal contact points
    ax_G.plot([x_S], [G_S_eq], 'o', color='#1f77b4', markersize=9, zorder=5)
    ax_G.plot([x_L], [G_L_eq], 's', color='#d62728', markersize=9, zorder=5)

    ax_G.annotate(f'Solidus Binodal Point\n$x_S = {x_S:.3f}$', xy=(x_S, G_S_eq),
                  xytext=(x_S - 0.22, G_S_eq + 2.8),
                  arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.5, headwidth=6),
                  fontsize=10, fontweight='bold', color='#1f77b4',
                  bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", lw=1.2))

    ax_G.annotate(f'Liquidus Binodal Point\n$x_L = {x_L:.3f}$', xy=(x_L, G_L_eq),
                  xytext=(x_L + 0.05, G_L_eq + 2.8),
                  arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                  fontsize=10, fontweight='bold', color='#d62728',
                  bbox=dict(boxstyle="round,pad=0.3", fc="#fde8e8", ec="#d62728", lw=1.2))

    # Shading the Binodal Regime
    ax_G.axvspan(0.0, x_S, color='blue', alpha=0.08, label=r'Single-Phase Solid ($\alpha$)')
    ax_G.axvspan(x_S, x_L, color='green', alpha=0.18, label=r'Binodal Regime: Two-Phase Coexistence ($\alpha + L$)')
    ax_G.axvspan(x_L, 1.0, color='red', alpha=0.08, label=r'Single-Phase Liquid ($L$)')

    ax_G.set_ylabel(r'Molar Gibbs Energy, $G_m$ [kJ / mol]', fontweight='bold')
    ax_G.set_title(f'Si-Ge Thermodynamic Equilibrium at $T = {T:.0f}$ K ({T - 273.15:.1f} °C)\n'
                   r'Top: $G-x$ Curves & Common Tangent | Bottom: First Derivative $\frac{dG}{dx}$ Matching',
                   fontweight='bold', pad=12)
    ax_G.legend(loc='lower center', ncol=2, framealpha=0.92)
    ax_G.grid(True, linestyle='--', alpha=0.5)

    # --- BOTTOM PANEL: First Derivative (dG/dx) ---
    ax_dG.plot(x, dG_S, label=r'$\frac{dG^S}{dx} = \mu_{Ge}^S - \mu_{Si}^S$ (Solid)', color='#1f77b4', lw=2.5)
    ax_dG.plot(x, dG_L, label=r'$\frac{dG^L}{dx} = \mu_{Ge}^L - \mu_{Si}^L$ (Liquid)', color='#d62728', lw=2.5)

    # Horizontal line at common tangent slope m
    ax_dG.axhline(m_tangent, color='#2ca02c', linestyle='--', lw=2.0,
                  label=f'Common Slope $m = {m_tangent:.2f}$ kJ/mol')

    # Intersection points on dG/dx
    ax_dG.plot([x_S], [m_tangent], 'o', color='#1f77b4', markersize=9, zorder=5)
    ax_dG.plot([x_L], [m_tangent], 's', color='#d62728', markersize=9, zorder=5)

    # Vertical tie-lines connecting top and bottom plots
    ymin_G, ymax_G = ax_G.get_ylim()
    ymin_dG, ymax_dG = -25, 25
    ax_dG.set_ylim(ymin_dG, ymax_dG)

    ax_G.vlines(x_S, ymin_G, G_S_eq, color='#1f77b4', linestyle=':', lw=1.8)
    ax_G.vlines(x_L, ymin_G, G_L_eq, color='#d62728', linestyle=':', lw=1.8)
    ax_dG.vlines(x_S, ymin_dG, m_tangent, color='#1f77b4', linestyle=':', lw=1.8)
    ax_dG.vlines(x_L, ymin_dG, m_tangent, color='#d62728', linestyle=':', lw=1.8)

    # Horizontal line at zero (to illustrate the common misconception)
    ax_dG.axhline(0, color='gray', linestyle=':', lw=1.2, label=r'Zero Slope $\frac{dG}{dx} = 0$ (Local Minima)')

    # Shading on derivative plot
    ax_dG.axvspan(0.0, x_S, color='blue', alpha=0.08)
    ax_dG.axvspan(x_S, x_L, color='green', alpha=0.18)
    ax_dG.axvspan(x_L, 1.0, color='red', alpha=0.08)

    # Explanatory annotation
    ax_dG.annotate(r'Binodal Condition:' + '\n' +
                   r'$\left.\frac{dG^S}{dx}\right|_{x_S} = \left.\frac{dG^L}{dx}\right|_{x_L} = m$',
                   xy=(0.5 * (x_S + x_L), m_tangent), xytext=(0.5 * (x_S + x_L) - 0.12, m_tangent + 10.0),
                   arrowprops=dict(facecolor='#2ca02c', shrink=0.1, width=1.5, headwidth=6),
                   fontsize=10, fontweight='bold', color='#1e7e34',
                   bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ca02c", lw=1.2))

    ax_dG.set_xlim(0, 1)
    ax_dG.set_xlabel(r'Mole Fraction of Germanium, $x_{Ge}$ (Component B)', fontweight='bold')
    ax_dG.set_ylabel(r'First Derivative, $\frac{dG}{dx}$ [kJ / mol]', fontweight='bold')
    ax_dG.legend(loc='lower right', framealpha=0.92)
    ax_dG.grid(True, linestyle='--', alpha=0.5)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.96, hspace=0.10)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    return fig


# ==============================================================================
# PLOT 2: T-x PHASE DIAGRAM HIGHLIGHTING THE COMPLETE BINODAL REGIME
# ==============================================================================
def plot_binodal_phase_diagram(T_slice=1450.0, save_path="si_ge_binodal_phase_diagram.png"):
    """
    Calculates and plots the entire Si-Ge Phase Diagram, clearly labeling
    and shading the Binodal Regime (Liquid + Solid Coexistence region).
    """
    T_range = np.linspace(TM_GE + 1.5, TM_SI - 1.5, 150)
    x_S_list, x_L_list, T_valid = [], [], []

    last_sol = None
    for T in T_range:
        res = solve_liquid_solid_binodal(T, x_init=last_sol)
        if res is not None:
            x_S_list.append(res[0])
            x_L_list.append(res[1])
            T_valid.append(T)
            last_sol = [res[0], res[1]]

    x_S_all = [0.0] + x_S_list + [1.0]
    x_L_all = [0.0] + x_L_list + [1.0]
    T_all = [TM_SI] + T_valid + [TM_GE]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Plot Solidus and Liquidus Binodal lines
    ax.plot(x_S_all, T_all, 'b-', lw=2.8, label=r'Solidus Binodal Line ($x_S$)')
    ax.plot(x_L_all, T_all, 'r-', lw=2.8, label=r'Liquidus Binodal Line ($x_L$)')

    # Color the regions
    ax.fill_betweenx(T_all, 0, x_S_all, color='blue', alpha=0.10)
    ax.fill_betweenx(T_all, x_S_all, x_L_all, color='green', alpha=0.20,
                     label=r'Binodal Regime: Two-Phase Coexistence ($\alpha + L$)')
    ax.fill_betweenx(T_all, x_L_all, 1.0, color='red', alpha=0.10)

    # Tie line at T_slice
    res_slice = solve_liquid_solid_binodal(T_slice)
    if res_slice is not None:
        x_S_s, x_L_s = res_slice
        ax.plot([x_S_s, x_L_s], [T_slice, T_slice], 'g--', lw=2.4,
                label=f'Isothermal Tie-Line at $T = {T_slice:.0f}$ K\n($x_S = {x_S_s:.3f}$, $x_L = {x_L_s:.3f}$)')
        ax.plot([x_S_s], [T_slice], 'bo', markersize=8)
        ax.plot([x_L_s], [T_slice], 'rs', markersize=8)

        # Annotate lever rule
        ax.annotate(r'$\mathbf{Tie-Line\ (Lever\ Rule)}$' + '\n' +
                    r'$f_\alpha = \frac{x_L - x_0}{x_L - x_S},\ f_L = \frac{x_0 - x_S}{x_L - x_S}$',
                    xy=(0.5 * (x_S_s + x_L_s), T_slice), xytext=(0.5 * (x_S_s + x_L_s) - 0.22, T_slice + 45),
                    arrowprops=dict(facecolor='darkgreen', shrink=0.08, width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold', color='darkgreen',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="darkgreen", lw=1.2))

    # Region labels
    ax.text(0.12, 1250, r'$\mathbf{Solid\ (\alpha)}$' + '\n(Single Phase)', fontsize=12, color='darkblue', ha='center')
    ax.text(0.53, 1370, r'$\mathbf{BINODAL\ REGIME}$' + '\n' + r'$\mathbf{(\alpha + L\ Mixture)}$',
            fontsize=12, color='darkgreen', ha='center')
    ax.text(0.85, 1600, r'$\mathbf{Liquid\ (L)}$' + '\n(Single Phase)', fontsize=12, color='darkred', ha='center')

    # Melting points
    ax.plot([0], [TM_SI], 'b^', markersize=10)
    ax.text(0.02, TM_SI + 10, f'$T_m(Si) = {TM_SI:.0f}$ K\n({TM_SI - 273.15:.0f} °C)', fontsize=10, color='darkblue')

    ax.plot([1], [TM_GE], 'rv', markersize=10)
    ax.text(0.78, TM_GE - 30, f'$T_m(Ge) = {TM_GE:.1f}$ K\n({TM_GE - 273.15:.1f} °C)', fontsize=10, color='darkred')

    ax.set_xlim(0, 1)
    ax.set_ylim(1150, 1750)
    ax.set_xlabel(r'Mole Fraction of Germanium, $x_{Ge}$', fontweight='bold')
    ax.set_ylabel(r'Temperature, $T$ [K]', fontweight='bold')
    ax.set_title('Si-Ge Phase Diagram: Scientifically Determined Binodal Regime',
                 fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.92)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    return fig


# ==============================================================================
# PLOT 3: SUBCRITICAL SOLID-STATE MISCIBILITY GAP (BINODAL & SPINODAL)
# ==============================================================================
def plot_solid_miscibility_gap(save_path="si_ge_solid_miscibility_gap.png"):
    """
    At low temperatures, Omega_S = 3850 J/mol > 0 causes the solid solution to
    exhibit a subcritical miscibility gap below Tc = Omega_S / (2*R) = 231.5 K.

    Here:
    - Binodal (first derivative = 0 for symmetric regular solution):
        d(Delta G_m) / dx = 0  =>  R*T*ln(x / (1-x)) + Omega_S*(1 - 2*x) = 0
    - Spinodal (second derivative = 0):
        d^2(Delta G_m) / dx^2 = 0  =>  x*(1 - x) = R*T / (2*Omega_S)
    """
    Tc = OMEGA_S / (2.0 * R)  # ~231.52 K

    T_vals = np.linspace(50.0, Tc - 0.5, 120)
    x_bin_1, x_bin_2 = [], []
    x_spin_1, x_spin_2 = [], []

    for T in T_vals:
        # Spinodal: exact analytical solution from d2G/dx2 = 0
        rad = 1.0 - 4.0 * (R * T) / (2.0 * OMEGA_S)
        if rad >= 0:
            s1 = 0.5 * (1.0 - np.sqrt(rad))
            s2 = 0.5 * (1.0 + np.sqrt(rad))
            x_spin_1.append(s1)
            x_spin_2.append(s2)

        # Binodal: numerical root of d(Delta G)/dx = 0
        def f_bin(x):
            return R * T * np.log(x / (1.0 - x)) + OMEGA_S * (1.0 - 2.0 * x)

        try:
            sol = root_scalar(f_bin, bracket=[1e-15, 0.499999], method='brentq')
            b1 = sol.root
            b2 = 1.0 - b1
            x_bin_1.append(b1)
            x_bin_2.append(b2)
        except Exception:
            x_bin_1.append(np.nan)
            x_bin_2.append(np.nan)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Complete curve arrays with the critical point (0.5, Tc)
    x_binodal = np.array(x_bin_1 + [0.5] + x_bin_2[::-1])
    T_binodal = np.array(list(T_vals) + [Tc] + list(T_vals)[::-1])

    x_spinodal = np.array(x_spin_1 + [0.5] + x_spin_2[::-1])
    T_spinodal = np.array(list(T_vals) + [Tc] + list(T_vals)[::-1])

    ax.plot(x_binodal, T_binodal, 'b-', lw=2.5,
            label=r'Binodal Curve ($\frac{\partial \Delta G}{\partial x} = 0$, Coexistence)')
    ax.plot(x_spinodal, T_spinodal, 'r--', lw=2.2,
            label=r'Spinodal Curve ($\frac{\partial^2 \Delta G}{\partial x^2} = 0$, Limit of Stability)')

    # Shading regimes
    ax.fill_between(x_spinodal, T_spinodal, color='red', alpha=0.15,
                    label='Unstable Regime (Spinodal Decomposition)')
    ax.fill_between(x_binodal, T_binodal, color='orange', alpha=0.12,
                    label='Metastable Regime (Nucleation & Growth)')

    # Critical point marker
    ax.plot([0.5], [Tc], 'k*', markersize=12,
            label=f'Critical Point: $T_c = {Tc:.1f}$ K ($-41.6$ °C)')
    ax.annotate(f'Critical Temperature\n$T_c = \\frac{{\\Omega_S}}{{2R}} = {Tc:.1f}$ K',
                xy=(0.5, Tc), xytext=(0.52, Tc + 25),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#ffffcc", ec="black", lw=1.2))

    ax.set_xlim(0, 1)
    ax.set_ylim(40, 300)
    ax.set_xlabel(r'Mole Fraction of Germanium, $x_{Ge}$', fontweight='bold')
    ax.set_ylabel(r'Temperature, $T$ [K]', fontweight='bold')
    ax.set_title(r'Solid-State Miscibility Gap in Si-Ge ($\Omega_S = 3850$ J/mol)' + '\n' +
                 r'Binodal via First Derivative ($\frac{\partial G}{\partial x} = 0$) & Spinodal via Second Derivative',
                 fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower center', framealpha=0.92)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    return fig


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Scientifically accurate determination and plotting of Si-Ge Binodal Regime."
    )
    parser.add_argument(
        '-t', '--temp', type=float, default=1450.0,
        help="Temperature in Kelvin for G-x and dG/dx plot (default: 1450 K)"
    )
    parser.add_argument(
        '--show', action='store_true',
        help="Display matplotlib interactive window"
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" Scientifically Accurate Si-Ge Binodal Regime Determination")
    print(f" Analysis Temperature: {args.temp:.1f} K ({args.temp - 273.15:.1f} °C)")
    print("=" * 70)

    # 1. G-x and First Derivative dG/dx matching
    plot_binodal_first_derivative(T=args.temp, save_path="si_ge_binodal_derivative.png")

    # 2. Complete Phase Diagram highlighting the Binodal Regime
    plot_binodal_phase_diagram(T_slice=args.temp, save_path="si_ge_binodal_phase_diagram.png")

    # 3. Solid-state miscibility gap binodal & spinodal
    plot_solid_miscibility_gap(save_path="si_ge_solid_miscibility_gap.png")

    print("\n[OK] All binodal plots generated and saved successfully!")
    print("  1. si_ge_binodal_derivative.png    -> First derivative dG/dx matching & G-x")
    print("  2. si_ge_binodal_phase_diagram.png -> Complete T-x Binodal Regime & tie line")
    print("  3. si_ge_solid_miscibility_gap.png -> Subcritical miscibility gap (Binodal & Spinodal)")

    if args.show:
        plt.show()
