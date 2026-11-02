"""
Thermodynamic Modeling and G-x Plot Generation for the Si-Ge Binary System.

This script calculates and plots:
1. Gibbs free energy of mixing / molar Gibbs energy (G-x curves) for Liquid and Solid phases.
2. Common tangent construction to determine equilibrium solidus (x_S) and liquidus (x_L) compositions.
3. Multi-temperature G-x evolution across all characteristic temperature regimes:
   - T > Tm(Si): Complete liquid stability
   - Tm(Ge) < T < Tm(Si): Two-phase equilibrium (common tangent)
   - T < Tm(Ge): Complete solid stability
4. Coupled G-x curve and T-x phase diagram (lens-shaped isomorphous system).

Thermodynamic Data Sources:
- Pure elements: SGTE standard lattice stabilities (Dinsdale, 1991).
- Interaction parameters (Regular/Subregular solution):
  * Solid (Diamond, A4): Omega_S = 3850 J/mol (Mey / Mullis / Bergman et al.)
  * Liquid: Omega_L(T) = 6610 - 0.354 * T J/mol (Mullis assessment)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# Set matplotlib style for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3,
})

# ==============================================================================
# 1. PHYSICAL CONSTANTS & THERMODYNAMIC PARAMETERS
# ==============================================================================
R = 8.314462618  # Universal gas constant [J / (mol * K)]

# Pure Silicon (Si) properties (SGTE / Dinsdale 1991)
TM_SI = 1687.0    # Melting point of pure Si [K] (1414 °C)
DH_SI = 50210.0   # Enthalpy of fusion for Si [J / mol]
DS_SI = DH_SI / TM_SI  # Entropy of fusion [J / (mol * K)] ~ 29.76 J/(mol*K)

# Pure Germanium (Ge) properties (SGTE / Dinsdale 1991)
TM_GE = 1211.4    # Melting point of pure Ge [K] (938.25 °C)
DH_GE = 36940.0   # Enthalpy of fusion for Ge [J / mol]
DS_GE = DH_GE / TM_GE  # Entropy of fusion [J / (mol * K)] ~ 30.49 J/(mol*K)


class SiGeThermodynamics:
    """
    Thermodynamic model for binary Silicon - Germanium (Si-Ge) solutions.
    Components:
      - Component 1 (A): Silicon (Si)
      - Component 2 (B): Germanium (Ge), mole fraction denoted as x = x_Ge
    Reference state:
      - Pure solid Si and pure solid Ge at temperature T (G_Si^S = 0, G_Ge^S = 0).
    """

    def __init__(self, model='regular'):
        """
        Parameters:
            model: 'regular' (default literature parameters) or 'ideal' (Omega = 0)
        """
        self.model = model

    def get_interaction_parameters(self, T):
        """
        Returns regular solution interaction parameters Omega_L and Omega_S [J/mol].
        """
        if self.model == 'ideal':
            return 0.0, 0.0
        # Regular solution parameters:
        # Liquid: slight temperature-dependence (Mullis assessment / CALPHAD)
        omega_L = 6610.0 - 0.354 * T
        # Solid (Diamond cubic A4): positive interaction energy due to atomic size mismatch
        omega_S = 3850.0
        return omega_L, omega_S

    def delta_G_fusion(self, T):
        """
        Gibbs free energy of fusion for pure components:
        dG = G^Liquid - G^Solid = dH * (1 - T / Tm)
        Returns:
            dG_Si, dG_Ge [J/mol]
        """
        dG_Si = DH_SI * (1.0 - T / TM_SI)
        dG_Ge = DH_GE * (1.0 - T / TM_GE)
        return dG_Si, dG_Ge

    def G_solid(self, x, T):
        """
        Molar Gibbs energy of the Solid phase relative to pure solid elements.
        G^S(x, T) = R*T*[(1-x)*ln(1-x) + x*ln(x)] + Omega_S * x * (1-x)
        """
        eps = 1e-15
        x = np.clip(x, eps, 1.0 - eps)
        _, omega_S = self.get_interaction_parameters(T)
        G_ideal = R * T * ((1.0 - x) * np.log(1.0 - x) + x * np.log(x))
        G_ex = omega_S * x * (1.0 - x)
        return G_ideal + G_ex

    def G_liquid(self, x, T):
        """
        Molar Gibbs energy of the Liquid phase relative to pure solid elements.
        G^L(x, T) = (1-x)*dG_Si + x*dG_Ge + R*T*[(1-x)*ln(1-x) + x*ln(x)] + Omega_L * x * (1-x)
        """
        eps = 1e-15
        x = np.clip(x, eps, 1.0 - eps)
        dG_Si, dG_Ge = self.delta_G_fusion(T)
        omega_L, _ = self.get_interaction_parameters(T)
        G_ref = (1.0 - x) * dG_Si + x * dG_Ge
        G_ideal = R * T * ((1.0 - x) * np.log(1.0 - x) + x * np.log(x))
        G_ex = omega_L * x * (1.0 - x)
        return G_ref + G_ideal + G_ex

    def chemical_potentials_solid(self, x, T):
        """
        Partial molar Gibbs energies (chemical potentials) for Solid:
        mu_Si^S = G^S - x * (dG^S / dx) = R*T*ln(1-x) + Omega_S * x^2
        mu_Ge^S = G^S + (1-x) * (dG^S / dx) = R*T*ln(x) + Omega_S * (1-x)^2
        """
        eps = 1e-15
        x = np.clip(x, eps, 1.0 - eps)
        _, omega_S = self.get_interaction_parameters(T)
        mu_Si = R * T * np.log(1.0 - x) + omega_S * (x ** 2)
        mu_Ge = R * T * np.log(x) + omega_S * ((1.0 - x) ** 2)
        return mu_Si, mu_Ge

    def chemical_potentials_liquid(self, x, T):
        """
        Partial molar Gibbs energies (chemical potentials) for Liquid:
        mu_Si^L = dG_Si + R*T*ln(1-x) + Omega_L * x^2
        mu_Ge^L = dG_Ge + R*T*ln(x) + Omega_L * (1-x)^2
        """
        eps = 1e-15
        x = np.clip(x, eps, 1.0 - eps)
        dG_Si, dG_Ge = self.delta_G_fusion(T)
        omega_L, _ = self.get_interaction_parameters(T)
        mu_Si = dG_Si + R * T * np.log(1.0 - x) + omega_L * (x ** 2)
        mu_Ge = dG_Ge + R * T * np.log(x) + omega_L * ((1.0 - x) ** 2)
        return mu_Si, mu_Ge

    def calculate_common_tangent(self, T, x_init=None):
        """
        Solves for equilibrium solidus (x_S) and liquidus (x_L) compositions
        by equating the chemical potentials in Solid and Liquid:
            mu_Si^S(x_S, T) = mu_Si^L(x_L, T)
            mu_Ge^S(x_S, T) = mu_Ge^L(x_L, T)
        """
        if not (TM_GE < T < TM_SI):
            return None

        # Robust initial guesses based on approximate linear lever between melting points
        if x_init is None:
            tau = (TM_SI - T) / (TM_SI - TM_GE)
            x_S_guess = np.clip(tau * 0.7, 0.05, 0.85)
            x_L_guess = np.clip(tau * 1.2, 0.15, 0.95)
            x_init = [x_S_guess, x_L_guess]

        def residual(p):
            x_S, x_L = p
            if x_S <= 1e-6 or x_S >= 1 - 1e-6 or x_L <= 1e-6 or x_L >= 1 - 1e-6:
                return [1e6, 1e6]
            mu_Si_S, mu_Ge_S = self.chemical_potentials_solid(x_S, T)
            mu_Si_L, mu_Ge_L = self.chemical_potentials_liquid(x_L, T)
            return [mu_Si_S - mu_Si_L, mu_Ge_S - mu_Ge_L]

        sol = root(residual, x_init, method='hybr')
        if sol.success and (0 < sol.x[0] < sol.x[1] < 1):
            return sol.x[0], sol.x[1]

        # Fallback to Nelder-Mead or bounded solver if needed
        from scipy.optimize import minimize
        def obj(p):
            r = residual(p)
            return r[0]**2 + r[1]**2
        res = minimize(obj, x_init, bounds=[(1e-4, 0.999), (1e-4, 0.999)])
        if res.success and (0 < res.x[0] < res.x[1] < 1):
            return res.x[0], res.x[1]

        return None


# ==============================================================================
# 2. PLOTTING FUNCTIONS
# ==============================================================================

def plot_single_temperature_gx(T=1450.0, thermo=None, save_path="si_ge_gx_single.png"):
    """
    Plots the G-x curves at a single temperature T with the common tangent
    construction and labeled phase regions.
    """
    if thermo is None:
        thermo = SiGeThermodynamics(model='regular')

    x = np.linspace(0.001, 0.999, 1000)
    G_S = thermo.G_solid(x, T) / 1000.0  # Convert to kJ/mol
    G_L = thermo.G_liquid(x, T) / 1000.0  # Convert to kJ/mol

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Plot G-x curves
    ax.plot(x, G_S, label=r'Solid Solution (Diamond, $\alpha$)', color='#1f77b4', lw=2.5)
    ax.plot(x, G_L, label=r'Liquid Solution ($L$)', color='#d62728', lw=2.5)

    # Equilibrium common tangent
    eq = thermo.calculate_common_tangent(T)
    if eq is not None:
        x_S, x_L = eq
        G_S_eq = thermo.G_solid(x_S, T) / 1000.0
        G_L_eq = thermo.G_liquid(x_L, T) / 1000.0

        # Chemical potentials at equilibrium (intercepts at x=0 and x=1)
        mu_Si_eq, mu_Ge_eq = thermo.chemical_potentials_solid(x_S, T)
        mu_Si_eq /= 1000.0
        mu_Ge_eq /= 1000.0

        # Common tangent line: y(x) = mu_Si + x * (mu_Ge - mu_Si)
        tangent_y = mu_Si_eq + x * (mu_Ge_eq - mu_Si_eq)
        ax.plot(x, tangent_y, '--', color='#2ca02c', lw=2, label='Common Tangent')

        # Contact points
        ax.plot([x_S], [G_S_eq], 'o', color='#1f77b4', markersize=8, zorder=5)
        ax.plot([x_L], [G_L_eq], 's', color='#d62728', markersize=8, zorder=5)

        # Annotations for contact points
        ax.annotate(f'Solidus\n$x_S = {x_S:.3f}$',
                    xy=(x_S, G_S_eq), xytext=(x_S - 0.16, G_S_eq + 2.5),
                    arrowprops=dict(facecolor='#1f77b4', shrink=0.08, width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold', color='#1f77b4',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", lw=1))

        ax.annotate(f'Liquidus\n$x_L = {x_L:.3f}$',
                    xy=(x_L, G_L_eq), xytext=(x_L + 0.05, G_L_eq + 2.5),
                    arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.5, headwidth=6),
                    fontsize=10, fontweight='bold', color='#d62728',
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fde8e8", ec="#d62728", lw=1))

        # Vertical tie lines down to x-axis
        ymin, ymax = ax.get_ylim()
        ax.vlines(x_S, ymin - 10, G_S_eq, color='#1f77b4', linestyle=':', alpha=0.7)
        ax.vlines(x_L, ymin - 10, G_L_eq, color='#d62728', linestyle=':', alpha=0.7)

        # Chemical potential intercepts
        ax.plot([0], [mu_Si_eq], 'k*', markersize=10, label=r'$\mu_{Si}^{\circ}$ Intercept')
        ax.plot([1], [mu_Ge_eq], 'm*', markersize=10, label=r'$\mu_{Ge}^{\circ}$ Intercept')
        ax.text(0.02, mu_Si_eq - 0.8, r'$\mu_{Si} = ' + f'{mu_Si_eq:.2f}$ kJ/mol', fontsize=10, color='black')
        ax.text(0.82, mu_Ge_eq - 0.8, r'$\mu_{Ge} = ' + f'{mu_Ge_eq:.2f}$ kJ/mol', fontsize=10, color='purple')

        # Shading phase regions
        ax.axvspan(0.0, x_S, alpha=0.10, color='blue', label=r'Solid Stable ($\alpha$)')
        ax.axvspan(x_S, x_L, alpha=0.15, color='green', label=r'Two-Phase ($\alpha + L$)')
        ax.axvspan(x_L, 1.0, alpha=0.10, color='red', label=r'Liquid Stable ($L$)')

    ax.set_xlim(0, 1)
    ax.set_xlabel(r'Mole Fraction of Germanium, $x_{Ge}$ (Component B)', fontweight='bold')
    ax.set_ylabel(r'Molar Gibbs Free Energy, $G_m$ [kJ / mol]', fontweight='bold')
    ax.set_title(f'Si-Ge $G-x$ Plot with Common Tangent Construction at $T = {T:.0f}$ K ({T - 273.15:.1f} °C)',
                 pad=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower center', framealpha=0.92, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    return fig, ax


def plot_multitemperature_gx(temperatures=None, thermo=None, save_path="si_ge_gx_multitemp.png"):
    """
    Plots the G-x curves at 6 representative temperatures spanning:
    1. Above Tm(Si)
    2. At Tm(Si)
    3. Intermediate high T
    4. Intermediate low T
    5. At Tm(Ge)
    6. Below Tm(Ge)
    """
    if thermo is None:
        thermo = SiGeThermodynamics(model='regular')

    if temperatures is None:
        temperatures = [1750.0, 1687.0, 1550.0, 1400.0, 1211.4, 1100.0]

    x = np.linspace(0.001, 0.999, 500)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True)
    axes = axes.flatten()

    for idx, (T, ax) in enumerate(zip(temperatures, axes)):
        G_S = thermo.G_solid(x, T) / 1000.0
        G_L = thermo.G_liquid(x, T) / 1000.0

        ax.plot(x, G_S, color='#1f77b4', lw=2.2, label=r'Solid ($S$)')
        ax.plot(x, G_L, color='#d62728', lw=2.2, label=r'Liquid ($L$)')

        # Status text
        if T > TM_SI:
            status = r"$T > T_m(Si)$: Liquid Stable"
            ax.axvspan(0, 1, color='red', alpha=0.08)
        elif abs(T - TM_SI) < 1.0:
            status = r"$T = T_m(Si)$: Coexistence at pure Si"
        elif abs(T - TM_GE) < 1.0:
            status = r"$T = T_m(Ge)$: Coexistence at pure Ge"
        elif T < TM_GE:
            status = r"$T < T_m(Ge)$: Solid Stable"
            ax.axvspan(0, 1, color='blue', alpha=0.08)
        else:
            status = r"$T_m(Ge) < T < T_m(Si)$: $S + L$ Equilibrium"

        # Calculate common tangent if in two-phase regime
        if TM_GE < T < TM_SI:
            eq = thermo.calculate_common_tangent(T)
            if eq is not None:
                x_S, x_L = eq
                mu_Si_eq, mu_Ge_eq = thermo.chemical_potentials_solid(x_S, T)
                tangent_y = (mu_Si_eq + x * (mu_Ge_eq - mu_Si_eq)) / 1000.0
                ax.plot(x, tangent_y, '--', color='#2ca02c', lw=1.8, label='Common Tangent')
                ax.plot([x_S], [thermo.G_solid(x_S, T) / 1000.0], 'o', color='#1f77b4', markersize=6)
                ax.plot([x_L], [thermo.G_liquid(x_L, T) / 1000.0], 's', color='#d62728', markersize=6)
                ax.axvspan(x_S, x_L, color='green', alpha=0.15)
                ax.text(0.5, 0.08, f"$x_S={x_S:.2f}$, $x_L={x_L:.2f}$",
                        transform=ax.transAxes, ha='center', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="green", alpha=0.9))

        ax.set_title(f"$T = {T:.1f}$ K ({T - 273.15:.0f} °C)\n{status}", fontsize=11, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.4)
        if idx % 3 == 0:
            ax.set_ylabel(r"$G_m$ [kJ / mol]", fontweight='bold')
        if idx >= 3:
            ax.set_xlabel(r"$x_{Ge}$", fontweight='bold')
        if idx == 0:
            ax.legend(loc='lower center', fontsize=9)

    plt.suptitle("Evolution of Si-Ge $G-x$ Curves with Decreasing Temperature", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    return fig, axes


def plot_coupled_gx_and_phase_diagram(T_slice=1450.0, thermo=None, save_path="si_ge_coupled_diagram.png"):
    """
    Plots the G-x curve at T_slice side-by-side with the complete equilibrium
    Si-Ge T-x phase diagram calculated by mapping the common tangents.
    """
    if thermo is None:
        thermo = SiGeThermodynamics(model='regular')

    # Calculate phase diagram (Liquidus and Solidus curves)
    T_range = np.linspace(TM_GE + 2.0, TM_SI - 2.0, 120)
    x_S_list = []
    x_L_list = []
    T_valid = []

    last_sol = None
    for T in T_range:
        res = thermo.calculate_common_tangent(T, x_init=last_sol)
        if res is not None:
            x_S_list.append(res[0])
            x_L_list.append(res[1])
            T_valid.append(T)
            last_sol = [res[0], res[1]]

    # Add pure component endpoints
    x_S_all = [0.0] + x_S_list + [1.0]
    x_L_all = [0.0] + x_L_list + [1.0]
    T_all = [TM_SI] + T_valid + [TM_GE]

    fig, (ax_pd, ax_gx) = plt.subplots(1, 2, figsize=(15, 6.5))

    # --- LEFT PANEL: T-x Phase Diagram ---
    ax_pd.plot(x_S_all, T_all, 'b-', lw=2.5, label='Solidus ($x_S$)')
    ax_pd.plot(x_L_all, T_all, 'r-', lw=2.5, label='Liquidus ($x_L$)')

    # Fill regions
    ax_pd.fill_betweenx(T_all, 0, x_S_all, color='blue', alpha=0.10)
    ax_pd.fill_betweenx(T_all, x_S_all, x_L_all, color='green', alpha=0.15)
    ax_pd.fill_betweenx(T_all, x_L_all, 1, color='red', alpha=0.10)

    # Labels inside regions
    ax_pd.text(0.12, 1250, r'$\mathbf{Solid\ (\alpha)}$' + '\n(Diamond Cubic)', fontsize=12, color='darkblue', ha='center')
    ax_pd.text(0.52, 1370, r'$\mathbf{\alpha + L}$', fontsize=13, color='darkgreen', ha='center')
    ax_pd.text(0.85, 1600, r'$\mathbf{Liquid\ (L)}$', fontsize=12, color='darkred', ha='center')

    # Draw tie-line at T_slice
    eq_slice = thermo.calculate_common_tangent(T_slice)
    if eq_slice is not None:
        x_S_eq, x_L_eq = eq_slice
        ax_pd.plot([x_S_eq, x_L_eq], [T_slice, T_slice], 'g--', lw=2.2, label=f'Tie-line at {T_slice:.0f} K')
        ax_pd.plot([x_S_eq], [T_slice], 'bo', markersize=8)
        ax_pd.plot([x_L_eq], [T_slice], 'rs', markersize=8)

    ax_pd.set_xlim(0, 1)
    ax_pd.set_ylim(1150, 1750)
    ax_pd.set_xlabel(r'Mole Fraction of Germanium, $x_{Ge}$', fontweight='bold')
    ax_pd.set_ylabel(r'Temperature, $T$ [K]', fontweight='bold')
    ax_pd.set_title('Si-Ge Binary Phase Diagram (Lens Type)', fontweight='bold')
    ax_pd.grid(True, linestyle='--', alpha=0.5)
    ax_pd.legend(loc='upper right')

    # --- RIGHT PANEL: G-x at T_slice ---
    x_grid = np.linspace(0.001, 0.999, 500)
    G_S = thermo.G_solid(x_grid, T_slice) / 1000.0
    G_L = thermo.G_liquid(x_grid, T_slice) / 1000.0

    ax_gx.plot(x_grid, G_S, 'b-', lw=2.5, label=r'Solid ($S$)')
    ax_gx.plot(x_grid, G_L, 'r-', lw=2.5, label=r'Liquid ($L$)')

    if eq_slice is not None:
        x_S_eq, x_L_eq = eq_slice
        G_S_eq = thermo.G_solid(x_S_eq, T_slice) / 1000.0
        G_L_eq = thermo.G_liquid(x_L_eq, T_slice) / 1000.0
        mu_Si_eq, mu_Ge_eq = thermo.chemical_potentials_solid(x_S_eq, T_slice)
        tangent_y = (mu_Si_eq + x_grid * (mu_Ge_eq - mu_Si_eq)) / 1000.0

        ax_gx.plot(x_grid, tangent_y, 'g--', lw=2, label='Common Tangent')
        ax_gx.plot([x_S_eq], [G_S_eq], 'bo', markersize=8)
        ax_gx.plot([x_L_eq], [G_L_eq], 'rs', markersize=8)

        ax_gx.axvspan(0.0, x_S_eq, alpha=0.10, color='blue')
        ax_gx.axvspan(x_S_eq, x_L_eq, alpha=0.15, color='green')
        ax_gx.axvspan(x_L_eq, 1.0, alpha=0.10, color='red')

        ax_gx.vlines(x_S_eq, ax_gx.get_ylim()[0], G_S_eq, color='b', linestyle=':')
        ax_gx.vlines(x_L_eq, ax_gx.get_ylim()[0], G_L_eq, color='r', linestyle=':')

    ax_gx.set_xlim(0, 1)
    ax_gx.set_xlabel(r'Mole Fraction of Germanium, $x_{Ge}$', fontweight='bold')
    ax_gx.set_ylabel(r'Molar Gibbs Energy, $G_m$ [kJ / mol]', fontweight='bold')
    ax_gx.set_title(f'Corresponding $G-x$ Curve at $T = {T_slice:.0f}$ K', fontweight='bold')
    ax_gx.grid(True, linestyle='--', alpha=0.5)
    ax_gx.legend(loc='lower center')

    plt.suptitle('Coupled Thermodynamic Equilibrium: Phase Diagram and $G-x$ Construction',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")
    return fig, (ax_pd, ax_gx)


# ==============================================================================
# MAIN EXECUTION & CLI
# ==============================================================================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate and plot Si-Ge Gibbs free energy (G-x) curves and phase diagram."
    )
    parser.add_argument(
        '-t', '--temp', type=float, default=1450.0,
        help="Temperature in Kelvin for single G-x plot and coupled tie-line (default: 1450 K)"
    )
    parser.add_argument(
        '-m', '--model', choices=['regular', 'ideal'], default='regular',
        help="Thermodynamic solution model: 'regular' (default) or 'ideal'"
    )
    parser.add_argument(
        '--show', action='store_true',
        help="Display interactive matplotlib window (plt.show())"
    )
    args = parser.parse_args()

    print("=" * 65)
    print(f" Si-Ge Thermodynamic Modeling & G-x Plot Generator")
    print(f" Solution Model: {args.model.upper()}")
    print(f" Target Temperature: {args.temp:.1f} K ({args.temp - 273.15:.1f} °C)")
    print("=" * 65)

    thermo_model = SiGeThermodynamics(model=args.model)

    # 1. Single Temperature G-x plot with Common Tangent
    plot_single_temperature_gx(T=args.temp, thermo=thermo_model, save_path="si_ge_gx_single.png")

    # 2. Multi-Temperature G-x evolution (1750 K down to 1100 K)
    plot_multitemperature_gx(thermo=thermo_model, save_path="si_ge_gx_multitemp.png")

    # 3. Coupled Phase Diagram & G-x with Tie-line
    plot_coupled_gx_and_phase_diagram(T_slice=args.temp, thermo=thermo_model, save_path="si_ge_coupled_diagram.png")

    print("\n[OK] All plots generated and saved successfully!")
    print("  1. si_ge_gx_single.png       -> Single-temperature G-x with common tangent")
    print("  2. si_ge_gx_multitemp.png    -> 6-panel G-x evolution across temperature regimes")
    print("  3. si_ge_coupled_diagram.png  -> Side-by-side Phase Diagram & G-x tie line")

    if args.show:
        plt.show()
