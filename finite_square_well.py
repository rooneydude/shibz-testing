"""
Quantum Particle in a Finite Square Well
=========================================

Solves the 1D time-independent Schrodinger equation for a finite square well
potential using a tridiagonal Hamiltonian matrix built from finite differences.

The potential is defined as:
    V(x) = 0        for |x| < a   (inside the well)
    V(x) = V0       for |x| >= a  (outside the well)

The discretized Schrodinger equation:
    -hbar^2/(2m) * (psi_{i+1} - 2*psi_i + psi_{i-1}) / dx^2 + V_i * psi_i = E * psi_i

leads to a tridiagonal matrix eigenvalue problem: H * psi = E * psi
where:
    diagonal elements:     H_ii = 2*K + V_i
    off-diagonal elements: H_{i,i+1} = H_{i+1,i} = -K
    with K = hbar^2 / (2 * m * dx^2)
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Physical constants (natural units: hbar = 1, m = 1)
# ---------------------------------------------------------------------------
HBAR = 1.0
MASS = 1.0


def build_potential(x, well_half_width, V0):
    """Construct the finite square well potential V(x).

    Parameters
    ----------
    x : ndarray
        Spatial grid points.
    well_half_width : float
        Half-width *a* of the well (well extends from -a to +a).
    V0 : float
        Depth of the potential outside the well.

    Returns
    -------
    V : ndarray
        Potential energy at each grid point.
    """
    V = np.where(np.abs(x) <= well_half_width, 0.0, V0)
    return V


def build_hamiltonian_tridiagonal(N, dx, V):
    """Build the diagonal and off-diagonal arrays of the tridiagonal
    Hamiltonian matrix using a second-order finite-difference scheme.

    H_ii         =  2K + V_i        (main diagonal)
    H_{i, i+-1}  = -K               (super/sub diagonal)

    where K = hbar^2 / (2 * m * dx^2).

    Parameters
    ----------
    N : int
        Number of interior grid points.
    dx : float
        Grid spacing.
    V : ndarray, shape (N,)
        Potential energy at each interior grid point.

    Returns
    -------
    diag : ndarray, shape (N,)
        Main diagonal of H.
    off_diag : ndarray, shape (N-1,)
        Off-diagonal (sub and super) of H.
    """
    K = HBAR ** 2 / (2.0 * MASS * dx ** 2)
    diag = 2.0 * K + V
    off_diag = -K * np.ones(N - 1)
    return diag, off_diag


def solve_finite_square_well(
    well_half_width=1.0,
    V0=50.0,
    x_max=3.0,
    N=500,
    num_states=6,
):
    """Solve the finite square well problem.

    Parameters
    ----------
    well_half_width : float
        Half-width of the well.
    V0 : float
        Potential energy outside the well.
    x_max : float
        The simulation domain spans [-x_max, x_max]. Should be larger than
        *well_half_width* so the wavefunction decays sufficiently.
    N : int
        Number of interior grid points (excluding boundaries where psi=0).
    num_states : int
        Number of lowest-energy eigenstates to return.

    Returns
    -------
    x : ndarray
        Spatial grid (interior points).
    V : ndarray
        Potential at each grid point.
    energies : ndarray
        Eigenvalues (energies) of the first *num_states* states.
    wavefunctions : ndarray, shape (N, num_states)
        Corresponding normalised wavefunctions (columns).
    """
    # Spatial grid (interior points only; boundary psi=0 at +/- x_max)
    x = np.linspace(-x_max, x_max, N + 2)[1:-1]  # drop boundary points
    dx = x[1] - x[0]

    # Potential
    V = build_potential(x, well_half_width, V0)

    # Tridiagonal Hamiltonian
    diag, off_diag = build_hamiltonian_tridiagonal(N, dx, V)

    # Solve the symmetric tridiagonal eigenvalue problem
    energies, wavefunctions = eigh_tridiagonal(
        diag, off_diag, select="i", select_range=(0, num_states - 1)
    )

    # Normalise wavefunctions
    for i in range(num_states):
        norm = np.sqrt(np.trapezoid(wavefunctions[:, i] ** 2, x))
        wavefunctions[:, i] /= norm

    return x, V, energies, wavefunctions


def classify_bound_states(energies, V0):
    """Separate bound states (E < V0) from scattering states."""
    bound_mask = energies < V0
    return bound_mask


def plot_results(x, V, energies, wavefunctions, well_half_width, V0):
    """Produce two publication-quality figures:
    1. Energy levels overlaid on the potential with wavefunctions.
    2. Probability densities |psi(x)|^2.
    """
    num_states = len(energies)
    bound_mask = classify_bound_states(energies, V0)

    # --- Figure 1: Potential + wavefunctions offset by their energy --------
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    # Draw potential
    ax1.plot(x, V, "k-", linewidth=2, label="V(x)")
    ax1.axhline(V0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Shade the well region
    ax1.axvspan(-well_half_width, well_half_width, alpha=0.08, color="blue")

    scale = 0.4 * (energies[1] - energies[0]) if num_states > 1 else 1.0

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_states))
    for n in range(num_states):
        E_n = energies[n]
        psi_n = wavefunctions[:, n]
        state_type = "bound" if bound_mask[n] else "quasi-bound"
        label = f"n={n}  E={E_n:.3f} ({state_type})"

        ax1.axhline(E_n, color=colors[n], linestyle=":", linewidth=0.7, alpha=0.5)
        ax1.plot(x, E_n + scale * psi_n, color=colors[n], linewidth=1.4, label=label)

    ax1.set_xlabel("x", fontsize=13)
    ax1.set_ylabel("Energy / Wave amplitude (offset by E_n)", fontsize=13)
    ax1.set_title("Finite Square Well: Wavefunctions", fontsize=15)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(-5, V0 * 1.2)
    fig1.tight_layout()

    # --- Figure 2: Probability densities -----------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for n in range(num_states):
        prob = wavefunctions[:, n] ** 2
        ax2.plot(x, prob, color=colors[n], linewidth=1.5,
                 label=f"n={n}  E={energies[n]:.3f}")

    ax2.axvspan(-well_half_width, well_half_width, alpha=0.08, color="blue",
                label="Well region")
    ax2.set_xlabel("x", fontsize=13)
    ax2.set_ylabel("|ψ(x)|²", fontsize=13)
    ax2.set_title("Probability Densities", fontsize=15)
    ax2.legend(fontsize=9)
    fig2.tight_layout()

    return fig1, fig2


def print_summary(energies, V0):
    """Print a table of computed energy eigenvalues."""
    bound_mask = classify_bound_states(energies, V0)
    print("=" * 55)
    print(f"  Finite Square Well  (V0 = {V0})")
    print("=" * 55)
    print(f"  {'n':<5} {'Energy':>12}  {'Type':<12}")
    print("-" * 55)
    for n, (E, is_bound) in enumerate(zip(energies, bound_mask)):
        tag = "BOUND" if is_bound else "QUASI-BOUND"
        print(f"  {n:<5} {E:>12.6f}  {tag:<12}")
    print("=" * 55)
    n_bound = int(bound_mask.sum())
    print(f"  Bound states found: {n_bound} / {len(energies)}")
    print()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    # --- Experiment parameters ---------------------------------------------
    well_half_width = 1.0       # Half-width of the well (a)
    V0 = 50.0                   # Potential barrier height outside the well
    x_max = 4.0                 # Simulation domain: [-x_max, x_max]
    N = 1000                    # Number of interior grid points
    num_states = 6              # Number of lowest eigenstates to compute

    # --- Solve -------------------------------------------------------------
    x, V, energies, wavefunctions = solve_finite_square_well(
        well_half_width=well_half_width,
        V0=V0,
        x_max=x_max,
        N=N,
        num_states=num_states,
    )

    # --- Report ------------------------------------------------------------
    print_summary(energies, V0)

    # --- Verify wavefunction properties ------------------------------------
    dx = x[1] - x[0]
    print("Normalisation check (should all be ~1.0):")
    for n in range(num_states):
        norm_sq = np.trapezoid(wavefunctions[:, n] ** 2, x)
        print(f"  n={n}: integral |psi|^2 dx = {norm_sq:.8f}")
    print()

    # --- Tunnelling: fraction of probability outside the well ---------------
    outside = np.abs(x) > well_half_width
    print("Probability outside the well (tunnelling):")
    for n in range(num_states):
        p_outside = np.trapezoid(wavefunctions[:, n] ** 2 * outside, x)
        print(f"  n={n}: P(|x| > a) = {p_outside:.6f}")
    print()

    # --- Visualise ---------------------------------------------------------
    fig1, fig2 = plot_results(
        x, V, energies, wavefunctions, well_half_width, V0
    )
    fig1.savefig("wavefunctions.png", dpi=150)
    fig2.savefig("probability_densities.png", dpi=150)
    print("Figures saved: wavefunctions.png, probability_densities.png")
    plt.show()


if __name__ == "__main__":
    main()
