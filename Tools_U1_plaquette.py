###############################################################################################################
#                                                   IMPORTS                                                   # 
###############################################################################################################

import dataclasses
from functools import reduce
from typing import Optional, Callable

import numpy as np
from numpy import fft as npfft

import scipy.special
import scipy.sparse as sp
from scipy.sparse import linalg as spla
from scipy.integrate import simpson

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

###############################################################################################################
#                                     MATPLOTLIB CONFIGURATION MANAGEMENT                                     # 
###############################################################################################################

plt.rcParams.update({
    #"figure.figsize": (8, 5),
    "figure.dpi": 100,
    "font.size": 14,
    "font.family": "serif",
    "text.usetex": True,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "axes.grid": True,
    "grid.linestyle": "-",
    "grid.alpha": 0.7,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "legend.fontsize": 12,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "text.latex.preamble": r'\usepackage{amsmath, amssymb}'
})

###############################################################################################################
#                                                USEFUL FUNCTIONS                                             #
###############################################################################################################

"""
CONTENTS: 
1) kron_n(*matrices): Compute the nested Kronecker product of multiple sparse matrices (CSR format).
2) vector_sparse_ndim_1idx(Nsize, index): Create a sparse vector of size Nsize with ones at the specified indices.
3.1 ab¡nd 3.2) Physical Mathieu functions and energies mapping (physical ordering).
4) Mathieu basis construction for a single link and computation of the relevant operators in this basis, given the Hamiltonian parameters.
"""

# 1)
def kron_n(*matrices):
    """
    Compute the nested Kronecker product of multiple sparse matrices (CSR format).

    Parameters:
        *matrices: One or more sparse matrices.

    Returns:
        scipy.sparse.csr_matrix: The Kronecker product of all inputs.
    """
    if not matrices:
        raise ValueError("At least one matrix must be provided.")
    return reduce(lambda a, b: sp.kron(a, b, format="csr"), matrices)

# 2)
def vector_sparse_ndim_1idx(Nsize, index):
    row = np.array(index)
    col = np.zeros_like(row)
    data = np.ones_like(row, dtype=float)
    return sp.csr_matrix((data, (row, col)), shape=(Nsize, 1))

# 3.1)
def physical_mathieu_characteristic_value_single(q, n_index):
    """
    Return the physical Mathieu characteristic value for a single level n_index.
    Physical ordering:
    0: a0 → ce0 ; 1: b2 → se2 ; 2: a2 → ce2 ; 3: b4 → se4 ; 4: a4 → ce4 ; ...
    """
    if n_index % 2 == 0:
        # even physical index → a_n with n = n_index
        m = n_index
        a = scipy.special.mathieu_a(m, q)
    else:
        # odd physical index → b_{n+1}
        m = n_index + 1
        a = scipy.special.mathieu_b(m, q)
    return a

# 3.2)
def physical_mathieu_wavefunction_single(q, n_index, x):
    """
    Return the Mathieu wavefunction ψ_n(x) corresponding to
    the physical index n_index. Note x = θ/2.
    Physical ordering:
    n_index = 0 → ce0 ; n_index = 1 → se2 ; n_index = 2 → ce2 ; n_index = 3 → se4 ; ...
    """
    if n_index % 2 == 0:
        # even → ce_{2k}
        m = n_index
        ψ, _ = scipy.special.mathieu_cem(m, q, x)
    else:
        # odd → se_{2k+2}
        m = n_index + 1
        ψ, _ = scipy.special.mathieu_sem(m, q, x)
    return ψ

# 4)
def single_link_data(λ_ovr_m, λ_ovr_g, N, N_θ):
    """
    This function is designed for a specific treatment of the Hamiltonian constants,
    so that ig H_l = (2m+g) n^2 - λ cos(θ) 
            -> H_l = [ 2m/λ + g/λ ] n^2 - cos(θ) 
                   = [ 2/λ_ovr_m + 1/λ_ovr_g ] n^2 - cos(θ) 
    where λ_ovr_m = λ/m and λ_ovr_g = λ/g.
    With this regard, the Mathieu parameter results q = - 2 / [ 2/λ_ovr_m + 1/λ_ovr_g ]
    and the energy eigenvalues are 
    E_{ν_l} =  [2m/λ + g/λ] a_{ν_l}(q) / 4 
            = [2/λ_ovr_m + 1/λ_ovr_g] a_{ν_l}(q) / 4, 
    where a_{ν_l}(q) are even Mathieu characteristic values -> {a_0, b_2, a_2, b_4, a_4,...}.

    INPUTS:
    λ_ovr_m = λ/m
    λ_ovr_g = λ/g
    N = cutoff number of basis states for each link
    N_θ = number of θ grid sampling points of continuous variable for integration

    OUTPUTS:
    E = array of size N with the energy eigenvalues of the single-link Hamiltonian
    exp_plusiθ_op_matrix = matrix of size (N, N) with the exp(+iθ) operator in the Mathieu basis
    exp_minusiθ_op_matrix = matrix of size (N, N) with the exp(-iθ) operator in the Mathieu basis
    nθ_sqr_op_matrix = matrix of size (N, N) with the n^2 operator in the Mathieu basis
    """
    q = - 2.0 / (2.0/λ_ovr_m + 1.0/λ_ovr_g) # Works in the λ/m=np.inf limit as well, where q = - λ_ovr_g / 2.0

    space_θ = XSpace(N_θ, a=-np.pi, b=np.pi)
    x_deg_θ = (space_θ.x / 2.0) * (180.0/np.pi) # Convert θ → x in degrees for mathieu_cem

    ν_matrix = np.zeros((N, N_θ))
    E_ν_vector = np.zeros(N)

    for ν in range(N):
        a = physical_mathieu_characteristic_value_single(q, ν)
        ψ_θ = physical_mathieu_wavefunction_single(q, ν, x_deg_θ)
        norm_θ = np.sqrt(simpson(np.abs(ψ_θ)**2, space_θ.x))
        ψ_θ/= norm_θ
        ν_matrix[ν] = ψ_θ
        E_ν_vector[ν] = (2.0/λ_ovr_m + 1.0/λ_ovr_g) * a / 4.0

    # exp(±iθ) matrices
    exp_plusiθ_op_matrix = np.zeros((N, N), dtype=complex)
    exp_minusiθ_op_matrix = np.zeros((N, N), dtype=complex)
    nθ_op_matrix = np.zeros((N, N), dtype=complex)
    nθ_sqr_op_matrix = np.zeros((N, N), dtype=complex)

    exp_plusiθ_op = space_θ.make_potential(lambda θ: np.exp(1j*θ))
    exp_minusiθ_op = space_θ.make_potential(lambda θ: np.exp(-1j*θ))
    nθ_op = space_θ.make_n_power(order=1)
    nθ_sqr_op = space_θ.make_n_power(order=2)
    
    for i in range(N):
        for j in range(N):
            exp_plusiθ_op_matrix[i, j] = simpson(np.conj(ν_matrix[i]) * exp_plusiθ_op(ν_matrix[j]), space_θ.x)
            exp_minusiθ_op_matrix[i, j] = simpson(np.conj(ν_matrix[i]) * exp_minusiθ_op(ν_matrix[j]), space_θ.x)
            nθ_op_matrix[i, j] = simpson(np.conj(ν_matrix[i]) * nθ_op(ν_matrix[j]), space_θ.x)
            nθ_sqr_op_matrix[i, j] = simpson(np.conj(ν_matrix[i]) * nθ_sqr_op(ν_matrix[j]), space_θ.x)

    # First i have to impose hermiticity in the number operators as, by construction of the mathieu basis, 
    # they are not hermitian numerically (due to numerical errors in the integration)
    nθ_op_matrix = (nθ_op_matrix + nθ_op_matrix.conj().T) / 2.0
    nθ_sqr_op_matrix = (nθ_sqr_op_matrix + nθ_sqr_op_matrix.conj().T) / 2.0

    return E_ν_vector, exp_plusiθ_op_matrix, exp_minusiθ_op_matrix, nθ_op_matrix, nθ_sqr_op_matrix

###############################################################################################################
#                                                X-SPACE DATACLASS                                            # 
###############################################################################################################

Tensor = np.ndarray
Vector = np.ndarray
Matrix = np.ndarray
TensorFunction = Callable[[Tensor], Tensor]
TensorExponential = Callable[[complex, Tensor], Tensor]

@dataclasses.dataclass
class XSpace:
    """
    This creates an object to store the coordinates of a position
    space and construct its potential and differential operators.

    Parameters
    ----------
    N : int
        Number of points of the grid.
    a : float
        First value of the position space interval.
    b : float
        Last value of the position space interval.
    axis : Optional[int]
        Axis of the dimension represented in this Space for the
        multidimensional case. Defaults to 0.
    close : bool
        Closed interval if True, not closed if False. Defaults to False.

    Derived properties
    ------------------
    L : float
        Interval length
    dx : float
        Separation between points in the grid
    x : np.ndarray[float]
        Vector of coordinates in position space
    p : np.ndarray[float]
        Vector of coordinates in frequency space.
    """

    N: int
    a: float = 0.0
    b: float = 1.0
    axis: int = 0
    close: bool = False
    # Derived fields
    L: float = dataclasses.field(init=False)
    dx: float = dataclasses.field(init=False)
    x: np.ndarray = dataclasses.field(init=False)
    p: np.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        self.L = self.b - self.a
        self.dx = self.L / self.N if not self.close else self.L / (self.N - 1)
        n = np.arange(self.N)
        self.x = self.a + self.dx * n
        if self.close:
            k = np.linspace(-np.pi / self.dx, np.pi / self.dx, self.N + 1)[:-1]
            self.p = npfft.fftshift(k)
        else:
            k = 2 * np.pi * n / self.L
            self.p = k - (n >= (self.N / 2)) * 2 * np.pi / self.dx
            # This is equivalent to do self.p = npfft.fftfreq(N, d=dx) * 2 * π without using any k
   
        # FFT works in a specific way and it has to be ordered as it is now, starting from the 0 (or nearest 0)
        # This is different as the "intuitive" way of writing it as -M,...,0,...,M

    # Standard derivative operator (FFT-based)
    def make_derivative(
        self, order: int = 1, axis: Optional[int] = None
    ) -> TensorFunction:
        """Return a function that applies a the derivative operator (d/dx) a number
        of times given by `order`, assuming that the dependence on the `x` coordinate
        is encoded on the `axis` index of the tensor.
        """
        axis = self.axis if axis is None else axis

        def apply(f: Tensor) -> Tensor:
            f = npfft.fft(f, axis=axis, norm="ortho")
            d = f.ndim
            pn = self.p.reshape([1] * axis + [self.N] + [1] * (d - axis - 1))
            pn = (1j * pn) ** order
            return npfft.ifft(pn * f, axis=axis, norm="ortho")

        return apply

    # (-i d/dx)^order operator
    def make_n_power(
        self, order: int = 1, axis: Optional[int] = None
    ) -> TensorFunction:
        """Return a function that applies a the derivative operator (-i*d/dx) a number
        of times given by `order`, assuming that the dependence on the `x` coordinate
        is encoded on the `axis` index of the tensor.
        """
        axis = self.axis if axis is None else axis

        def apply(f: Tensor) -> Tensor:
            f = npfft.fft(f, axis=axis, norm="ortho")
            d = f.ndim
            pn = self.p.reshape([1] * axis + [self.N] + [1] * (d - axis - 1)) ** order
            return npfft.ifft(pn * f, axis=axis, norm="ortho")

        return apply

    # exp(z * (-i d/dx)^order)
    def make_n_power_exp(
        self, order: int = 1, axis: Optional[int] = None
    ) -> TensorExponential:
        """Return a function that applies a the derivative operator (-i*d/dx) a number
        of times given by `order`, assuming that the dependence on the `x` coordinate
        is encoded on the `axis` index of the tensor.
        """
        axis = self.axis if axis is None else axis

        def apply(z: complex, f: Tensor) -> Tensor:
            f = npfft.fft(f, axis=axis, norm="ortho")
            d = f.ndim
            pn = self.p.reshape([1] * axis + [self.N] + [1] * (d - axis - 1)) ** order
            return npfft.ifft(np.exp(z * pn) * f, axis=axis, norm="ortho")

        return apply
    
    # Potential multiplication operator V(x)
    def make_potential(self, V, axis: Optional[int] = None) -> TensorFunction:
        """Return a function that applies the potential `V(x)` defined over the `x`
        coordinate, assuming that this coordinate is encoded in the `axis` index of
        the input tensor."""
        axis = self.axis if axis is None else axis
        Vx = V(self.x)

        def apply(f: Tensor) -> Tensor:
            d = f.ndim
            theV = Vx.reshape([1] * axis + [self.N] + [1] * (d - axis - 1))
            return f * theV

        return apply

    # Exponential of potential operator exp(z * V(x))
    def make_potential_exp(self, V, axis: Optional[int] = None) -> TensorExponential:
        """Return a function that applies the potential `V(x)` defined over the `x`
        coordinate, assuming that this coordinate is encoded in the `axis` index of
        the input tensor."""
        axis = self.axis if axis is None else axis
        Vx = V(self.x)

        def apply(z: complex, f: Tensor) -> Tensor:
            d = f.ndim
            theV = Vx.reshape([1] * axis + [self.N] + [1] * (d - axis - 1))
            return f * np.exp(z * theV)

        return apply

###############################################################################################################
#                                                  HAMILTONIANS                                               #
###############################################################################################################

###############################   ------  CHARGE BASIS, α_1 = -1   ------   ###############################
# Admits N even/odd and with/without JJcapacitance. 
# Also notice the choice of order for the charge basis: [ θ12, θ24, θ13, θ34 ] 
# (this is important for the definition of the nϕi's and the Hamiltonian terms).

def inf_space_LGT_plaquette_charge_basis_Matrix_reduced4vars_alpha1minus1(
    EC_m1, EC_g1, EJ1,
    EC_m2, EC_g2, EJ2,
    EC_m3, EC_g3, EJ3,
    EC_m4, EC_g4, EJ4,
    Nsize,
    EC_mg1=0, EC_mm1=0,
    EC_mg2=0, EC_mm2=0,
    EC_mg3=0, EC_mm3=0,
    EC_mg4=0, EC_mm4=0,
):
    if Nsize % 2 == 0:
        M = int(Nsize / 2) 
        id_c = sp.eye(2 * M )
        n_c = sp.spdiags(np.arange(-M, M), [0], 2*M, 2*M)
        n2 = n_c @ n_c
        up_c = sp.spdiags(np.ones(2 * M), [1], 2 * M, 2 * M)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M - 1], [0])), shape=(2 * M, 2 * M))  # wrap-around term
    else:
        M = (Nsize-1) // 2 
        id_c = sp.eye(2 * M + 1)
        n_c = sp.spdiags(np.arange(-M, M + 1), [0], 2 * M + 1, 2 * M + 1)
        n2 = n_c @ n_c
        up_c = sp.spdiags(np.ones(2 * M + 1), [1], 2 * M + 1, 2 * M + 1)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M], [0])), shape=(2 * M + 1, 2 * M + 1))  # wrap-around term

    ###### CHOICE OF ORDER HERE: [ θ12, θ24, θ13, θ34 ]
    nθ12 = kron_n(n_c, id_c, id_c, id_c)
    nθ12_sqr = kron_n(n2, id_c, id_c, id_c)
    nθ24 = kron_n(id_c, n_c, id_c, id_c)
    nθ24_sqr = kron_n(id_c, n2, id_c, id_c)
    nθ13 = kron_n(id_c, id_c, n_c, id_c)
    nθ13_sqr = kron_n(id_c, id_c, n2, id_c)
    nθ34 = kron_n(id_c, id_c, id_c, n_c)
    nθ34_sqr = kron_n(id_c, id_c, id_c, n2)
    id_N = kron_n(id_c, id_c, id_c, id_c)

    α_1 = -1
    α_2 = 0
    α_3 = 0
    α_4 = 1

    nϕ1 = α_1 * id_N + nθ12 + nθ13
    nϕ1_sqr = nϕ1 @ nϕ1
    nϕ2 = α_2 * id_N + nθ24 - nθ12
    nϕ2_sqr = nϕ2 @ nϕ2
    nϕ3 = α_3 * id_N + nθ34 - nθ13
    nϕ3_sqr = nϕ3 @ nϕ3
    nϕ4 = α_4 * id_N - nθ24 - nθ34
    nϕ4_sqr = nϕ4 @ nϕ4

    cos_12 = 0.5 * ( 
        kron_n(up_c, id_c, id_c, id_c) 
        +  kron_n(up_c.T, id_c, id_c, id_c) 
    ) 
    cos_24 = 0.5 * ( 
        kron_n(id_c, up_c, id_c, id_c) 
        +  kron_n(id_c, up_c.T, id_c, id_c) 
    ) 
    cos_13 = 0.5 * ( 
        kron_n(id_c, id_c, up_c, id_c) 
        +  kron_n(id_c, id_c, up_c.T, id_c) 
    ) 
    cos_34 = 0.5 * ( 
        kron_n(id_c, id_c, id_c, up_c) 
        +  kron_n(id_c, id_c, id_c, up_c.T) 
    ) 

    H12 = (
          4 * EC_m1 * nϕ1_sqr 
        + 4 * EC_g1 * nθ12_sqr 
        + 4 * EC_mg1 * (nθ12 @  nϕ2 - nϕ1 @ nθ12) 
        + 4 * EC_mm1 * nϕ2 @ nϕ1 
        - EJ1 * cos_12 
    )  
    H24 = (
          4 * EC_m2 * nϕ2_sqr 
        + 4 * EC_g2 * nθ24_sqr 
        + 4 * EC_mg2 * (nθ24 @  nϕ4 - nϕ2 @ nθ24) 
        + 4 * EC_mm2 * nϕ4 @ nϕ2 
        - EJ2 * cos_24
    )
    H13 = (
          4 * EC_m3 * nϕ3_sqr 
        + 4 * EC_g3 * nθ13_sqr 
        + 4 * EC_mg3 * (nθ13 @  nϕ3 - nϕ1 @ nθ13) 
        + 4 * EC_mm3 * nϕ3 @ nϕ1 
        - EJ3 * cos_13
    )
    H34 = (
          4 * EC_m4 * nϕ4_sqr 
        + 4 * EC_g4 * nθ34_sqr 
        + 4 * EC_mg4 * (nθ34 @  nϕ4 - nϕ3 @ nθ34) 
        + 4 * EC_mm4 * nϕ4 @ nϕ3 
        - EJ4 *  cos_34
    )

    return H12 + H24 + H13 + H34 


###############################   ------  CHARGE BASIS, α's = 0   ------   ###############################
def inf_space_LGT_plaquette_charge_basis_Matrix_reduced4vars_alphas0(
    EC_g1, EJ1, 
    EC_g2, EJ2, 
    EC_g3, EJ3, 
    EC_g4, EJ4, 
    Nsize,
    EC_m1=0, EC_m2=0, EC_m3=0, EC_m4=0
):
    
    if Nsize % 2 == 0:
        M = int(Nsize / 2)  
        id_c = sp.eye(2 * M )
        n_c = sp.spdiags(np.arange(-M, M), [0], 2*M, 2*M)
        up_c = sp.spdiags(np.ones(2 * M), [1], 2 * M, 2 * M)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M - 1], [0])), shape=(2 * M, 2 * M))  # wrap-around term
        n2 = n_c @ n_c
    
    else:
        M = (Nsize-1) // 2  
        id_c = sp.eye(2 * M + 1)
        n_c = sp.spdiags(np.arange(-M, M + 1), [0], 2 * M + 1, 2 * M + 1)
        up_c = sp.spdiags(np.ones(2 * M + 1), [1], 2 * M + 1, 2 * M + 1)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M], [0])), shape=(2 * M + 1, 2 * M + 1))  # wrap-around term
        n2 = n_c @ n_c

        ###### CHOICE OF ORDER HERE: [ θ12, θ24, θ13, θ34 ]
        nθ12 = kron_n(n_c, id_c, id_c, id_c)
        nθ12_sqr = kron_n(n2, id_c, id_c, id_c)
        nθ24 = kron_n(id_c, n_c, id_c, id_c)
        nθ24_sqr = kron_n(id_c, n2, id_c, id_c)
        nθ13 = kron_n(id_c, id_c, n_c, id_c)
        nθ13_sqr = kron_n(id_c, id_c, n2, id_c)
        nθ34 = kron_n(id_c, id_c, id_c, n_c)
        nθ34_sqr = kron_n(id_c, id_c, id_c, n2)
        nϕ1 = nθ12 + nθ13
        nϕ1_sqr = nϕ1 @ nϕ1
        nϕ2 = nθ24 - nθ12
        nϕ2_sqr = nϕ2 @ nϕ2
        nϕ3 = nθ34 - nθ13
        nϕ3_sqr = nϕ3 @ nϕ3
        nϕ4 = - nθ24 - nθ34
        nϕ4_sqr = nϕ4 @ nϕ4

        cos_12 = 0.5 * ( 
            kron_n(up_c, id_c, id_c, id_c) 
            +  kron_n(up_c.T, id_c, id_c, id_c) 
        ) 
        cos_24 = 0.5 * ( 
            kron_n(id_c, up_c, id_c, id_c) 
            +  kron_n(id_c, up_c.T, id_c, id_c) 
        ) 
        cos_13 = 0.5 * ( 
            kron_n(id_c, id_c, up_c, id_c) 
            +  kron_n(id_c, id_c, up_c.T, id_c) 
        ) 
        cos_34 = 0.5 * ( 
            kron_n(id_c, id_c, id_c, up_c) 
            +  kron_n(id_c, id_c, id_c, up_c.T) 
        ) 

        H12 = (
              4 * EC_m1 * nϕ1_sqr 
            + 4 * EC_g1 * nθ12_sqr 
            - EJ1 * cos_12 
        )     
        H24 = (
              4 * EC_m2 * nϕ2_sqr 
            + 4 * EC_g2 * nθ24_sqr 
            - EJ2 * cos_24
        )
        H13 = (
              4 * EC_m3 * nϕ3_sqr 
            + 4 * EC_g3 * nθ13_sqr 
            - EJ3 * cos_13 
        )
        H34 = (
              4 * EC_m4 * nϕ4_sqr 
            + 4 * EC_g4 * nθ34_sqr 
            - EJ4 * cos_34 
        )

        return H12 + H24 + H13 + H34 
    

#####################   ------  OPERATORS FROM PREVIOUS HAMILTONIAN, CHARGE BASIS, α's = 0   ------   #####################
def make_plaquette_operator_charge_basis_Matrix_reduced4vars_alphas0(Nsize):
    
    if Nsize % 2 == 0:
        M = int(Nsize / 2)  
        up_c = sp.spdiags(np.ones(2 * M), [1], 2 * M, 2 * M)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M - 1], [0])), shape=(2 * M, 2 * M))  # wrap-around term
    
    else:
        M = (Nsize-1) // 2  
        up_c = sp.spdiags(np.ones(2 * M + 1), [1], 2 * M + 1, 2 * M + 1)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M], [0])), shape=(2 * M + 1, 2 * M + 1))  # wrap-around term

    cos_1234 = 0.5 * ( 
        kron_n(up_c, up_c, up_c.T, up_c.T) 
        +  kron_n(up_c.T, up_c.T, up_c, up_c) 
    ) 
    
    return cos_1234

def make_n12_operator_charge_basis_Matrix_reduced4vars_alphas0(Nsize, order):
    # At least for the moment, only ready for n and n^2

    if Nsize % 2 == 0:
        M = int(Nsize / 2) 
        id_c = sp.eye(2 * M )
        n_c = sp.spdiags(np.arange(-M, M), [0], 2*M, 2*M)
        n2 = n_c @ n_c
    
    else:
        M = (Nsize-1) // 2 
        id_c = sp.eye(2 * M + 1)
        n_c = sp.spdiags(np.arange(-M, M + 1), [0], 2 * M + 1, 2 * M + 1)
        n2 = n_c @ n_c

    if order == 2:
        nθ12 =  kron_n(n2, id_c, id_c, id_c)
    elif order == 1:
        nθ12 =  kron_n(n_c, id_c, id_c, id_c)
    else:
        raise ValueError("Order must be 1 or 2.")

    return nθ12

def make_n13_operator_charge_basis_Matrix_reduced4vars_alphas0(Nsize, order):
    # At least for the moment, only ready for n and n^2
    if Nsize % 2 == 0:
        M = int(Nsize / 2) 
        id_c = sp.eye(2 * M )
        n_c = sp.spdiags(np.arange(-M, M), [0], 2*M, 2*M)
        n2 = n_c @ n_c
    
    else:
        M = (Nsize-1) // 2 
        id_c = sp.eye(2 * M + 1)
        n_c = sp.spdiags(np.arange(-M, M + 1), [0], 2 * M + 1, 2 * M + 1)
        n2 = n_c @ n_c

    if order == 2:
        nθ13 =  kron_n(id_c, id_c, n2, id_c)
    elif order == 1:
        nθ13 =  kron_n(id_c, id_c, n_c, id_c)
    else:
        raise ValueError("Order must be 1 or 2.")

    return nθ13

def make_n24_operator_charge_basis_Matrix_reduced4vars_alphas0(Nsize, order):   
    # At least for the moment, only ready for n and n^2
    if Nsize % 2 == 0:
        M = int(Nsize / 2) 
        id_c = sp.eye(2 * M )
        n_c = sp.spdiags(np.arange(-M, M), [0], 2*M, 2*M)
        n2 = n_c @ n_c
    
    else:
        M = (Nsize-1) // 2 
        id_c = sp.eye(2 * M + 1)
        n_c = sp.spdiags(np.arange(-M, M + 1), [0], 2 * M + 1, 2 * M + 1)
        n2 = n_c @ n_c

    if order == 2:
        nθ24 =  kron_n(id_c, n2, id_c, id_c)
    elif order == 1:
        nθ24 =  kron_n(id_c, n_c, id_c, id_c)
    else:
        raise ValueError("Order must be 1 or 2.")

    return nθ24

def make_n34_operator_charge_basis_Matrix_reduced4vars_alphas0(Nsize, order):   
    # At least for the moment, only ready for n and n^2
    if Nsize % 2 == 0:
        M = int(Nsize / 2) 
        id_c = sp.eye(2 * M )
        n_c = sp.spdiags(np.arange(-M, M), [0], 2*M, 2*M)
        n2 = n_c @ n_c
    
    else:
        M = (Nsize-1) // 2 
        id_c = sp.eye(2 * M + 1)
        n_c = sp.spdiags(np.arange(-M, M + 1), [0], 2 * M + 1, 2 * M + 1)
        n2 = n_c @ n_c

    if order == 2:
        nθ34 =  kron_n(id_c, id_c, id_c, n2)
    elif order == 1:
        nθ34 =  kron_n(id_c, id_c, id_c, n_c)
    else:
        raise ValueError("Order must be 1 or 2.")

    return nθ34


##########################   ------  EFFECTIVE HAMILTONIAN, CHARGE BASIS, α's = 0   ------   ##########################
def inf_space_LGT_plaquette_charge_basis_Matrix_reduced4vars_alphas0_effective_staticm(
    EC_g1, EC_g2, EC_g3, EC_g4,
    EJ, EC_m,
    Nsize
):
    
    if Nsize % 2 == 0:
        M = int(Nsize / 2) 
        n_c = sp.spdiags(np.arange(-M, M), [0], 2*M, 2*M)
        n2 = n_c @ n_c
        up_c = sp.spdiags(np.ones(2 * M), [1], 2 * M, 2 * M)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M - 1], [0])), shape=(2 * M, 2 * M))
    else:
        M = (Nsize-1) // 2 
        n_c = sp.spdiags(np.arange(-M, M + 1), [0], 2 * M + 1, 2 * M + 1)
        n2 = n_c @ n_c
        up_c = sp.spdiags(np.ones(2 * M + 1), [1], 2 * M + 1, 2 * M + 1)
        up_c = up_c + sp.csr_matrix(([1.0], ([2 * M], [0])), shape=(2 * M + 1, 2 * M + 1))

    ###### CHOICE OF ORDER HERE: [ θ12, θ24, θ13, θ34 ]
    nθ12 = n_c
    nθ12_sqr = n2
    nθ24 = nθ12
    nθ24_sqr = n2
    nθ13 = -nθ12
    nθ13_sqr = n2
    nθ34 = -nθ12
    nθ34_sqr = n2

    cos_1234= 0.5 * ( 
        up_c + up_c.T
    ) 

    H = (
        (4 * EC_g1) * nθ12_sqr
        + (4 * EC_g2) * nθ24_sqr
        + (4 * EC_g3) * nθ13_sqr
        + (4 * EC_g4) * nθ34_sqr
        - (5*EJ**4 / (16*(4*EC_m)**3)) *  cos_1234    
    )

    return H
