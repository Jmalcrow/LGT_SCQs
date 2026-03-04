#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 14:49:51 2026

Using DMRG to find low-energy properties and perhaps dynamics of the
plaquette.

I still need to be careful about the coefficient 1/4 in the vortex operator.
# !!!

@author: zhengshi
"""

import h5py  # For saving tenpy objects, if necessary.
import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
# import qutip as qt
# import scipy.constants as spconst
import scqubits as scq
import tenpy


# %% For use in Tenpy model construction.
class TransmonSite(tenpy.networks.Site):
    r"""
    """

    def __init__(self, EJ, EC, ng, ncut, truncated_dim):
        # if not conserve:
        conserve = 'None'
        # if conserve not in ['dipole', 'N', 'parity', 'None']:
        #     raise ValueError('invalid `conserve`: ' + repr(conserve))
        self.EJ = EJ
        self.EC = EC
        self.ng = ng
        self.ncut = ncut
        self.truncated_dim = truncated_dim
        transmon = scq.Transmon(EJ, EC, ng, ncut, truncated_dim)
        self.energy_esys = transmon.eigensys(truncated_dim)
        list_E = self.energy_esys[0]
        diag_elem_NN = (np.arange(-self.ncut, self.ncut + 1, 1))**2
        op_NN = transmon.process_op(
            native_op=np.diag(diag_elem_NN), energy_esys=self.energy_esys)
        exp_i_phi = transmon.exp_i_phi_operator(energy_esys=self.energy_esys)
        ops = dict(
            E=np.diag(list_E),
            N=transmon.n_operator(energy_esys=self.energy_esys),
            NN=op_NN,
            exp_i_phi=exp_i_phi,
            exp_mi_phi=(np.conj(exp_i_phi)).T,
            cos_phi=transmon.cos_phi_operator(energy_esys=self.energy_esys),
            sin_phi=transmon.sin_phi_operator(energy_esys=self.energy_esys),
            )
        leg = tenpy.linalg.np_conserved.LegCharge.from_trivial(truncated_dim)
        self.conserve = conserve
        states = [str(n) for n in range(0, truncated_dim)]
        tenpy.networks.Site.__init__(
            self, leg, states, sort_charge=True, **ops)
        self.state_labels['vac'] = self.state_labels['0']  # alias

    # def __repr__(self):
    #     """Debug representation of self."""
    #     return f'TransmonSite({self.truncated_dim:d}, {self.EC:f}, {self.EJ:f})'


class CoupledTransmonModel(tenpy.models.model.CouplingMPOModel):
    default_lattice = tenpy.models.lattice.TrivialLattice
    force_default_lattice = True

    def init_sites(self, model_params):
        list_EJ = np.asarray(model_params.get('EJ', 0, 'real_or_array'))
        ECmat = np.asarray(model_params.get('ECmat', 0, 'real_or_array'))
        list_truncated_dim = np.asarray(model_params.get(
            'truncated_dim', 6, 'real_or_array'))
        list_ncut = np.asarray(model_params.get(
            'ncut', 80, 'real_or_array'))
        return [TransmonSite(
            list_EJ[n], ECmat[n, n], 0, list_ncut[n], list_truncated_dim[n])
            for n in range(len(list_truncated_dim))]

    def init_lattice(self, model_params):
        sites = self.init_sites(model_params)
        basis = np.array(([len(sites), 0], [0, 1]))
        pos = np.array([[i, 0] for i in range(len(sites))])
        lat = tenpy.models.lattice.Lattice(
            [1, 1], sites, basis=basis, positions=pos,
        )
        return lat

    def init_terms(self, model_params):
        ECmat = np.asarray(model_params.get('ECmat', 0, 'real_or_array'))
        const_shift = model_params.get('constant_shift', 0, 'real_or_array')
        assert const_shift < 0
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(1, u, 'E')
            self.add_onsite(const_shift, u, 'Id')
        for u1 in range(len(self.lat.unit_cell)):
            for u2 in range(u1+1, len(self.lat.unit_cell)):
                self.add_coupling(
                    8*ECmat[u1, u2], u1, 'N', u2, 'N', (0, 0))


class HamFactory(CoupledTransmonModel):
    def init_terms(self, model_params):
        ECmat = np.asarray(model_params.get('ECmat', 0, 'real_or_array'))
        # list_EJ = np.asarray(model_params.get('EJ', 0, 'real_or_array'))
        ham_type = model_params.get('ham_type', 'ham_total')
        ham_idx = model_params.get('ham_idx', -1)
        if ham_type == 'ham_total':
            for u in range(len(self.lat.unit_cell)):
                self.add_onsite(1, u, 'E')
                # self.add_onsite(4*ECmat[u, u], u, 'NN')
                # self.add_onsite(-list_EJ[u], u, 'cos_phi')
            for u1 in range(len(self.lat.unit_cell)):
                for u2 in range(u1+1, len(self.lat.unit_cell)):
                    self.add_coupling(
                        8*ECmat[u1, u2], u1, 'N', u2, 'N', (0, 0))
        elif ham_type in ['cos_phi', 'sin_phi', 'NN']:
            assert ham_idx in np.arange(len(self.lat.unit_cell))
            self.add_onsite(1, ham_idx, ham_type)
        elif ham_type == 'cos_phi_plaquette':
            # This is hard-wired! # !!!
            # Needs proper generalization to multiple plaquettes.
            self.add_multi_coupling_term(
                0.5, [0, 1, 2, 3],
                ['exp_i_phi', 'exp_i_phi', 'exp_mi_phi', 'exp_mi_phi'],
                ['Id']*3,
                plus_hc=True
                )
        elif ham_type == 'exp_i_theta_n_plaquette':
            # This is hard-wired! # !!!
            # Needs proper generalization to multiple plaquettes.
            theta = model_params.get('op_theta', np.nan)
            list_theta = [theta, theta, -theta, -theta]
            for n_site, site in enumerate(self.lat.mps_sites()):
                transmon = scq.Transmon(
                    site.EJ, site.EC, site.ng, site.ncut, site.truncated_dim)
                diag_elem_N = np.arange(-site.ncut, site.ncut + 1, 1)
                op_exp_pm_i_theta_n = transmon.process_op(
                    native_op=np.diag(
                        np.exp(1j*list_theta[n_site]*diag_elem_N/4)),
                    energy_esys=site.energy_esys)
                site.add_op('exp_pm_i_theta_n', op_exp_pm_i_theta_n)
            self.add_multi_coupling_term(
                1, [0, 1, 2, 3],
                ['exp_pm_i_theta_n']*4,
                ['Id']*3,
                )
        else:
            raise ValueError('Unexpected Hamiltonian type.')


def calc_mat_elem(operator_mpo, list_states, mpo_options):
    # Calculate matrix elements.
    num_states = len(list_states)
    operator_mat = np.zeros((num_states, num_states), dtype=np.complex128)
    for col in range(num_states):
        col_state = list_states[col].copy()
        operator_mpo.apply(col_state, mpo_options)  # In place.
        for row in range(col, num_states):
            row_state = list_states[row].copy()
            operator_mat[row, col] = row_state.overlap(col_state)
            operator_mat[col, row] = np.conj(operator_mat[row, col])
        # if col % 10 == 0:
        #     print(col)
    return operator_mat


# %% Model parameters.


list_ncut = [100]*4
list_trunc_dims = [20]*4


dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 100,
        'svd_min': 1.e-10,
    },
    # 'verbose': True,
    'combine': True,
}
mpo_options = {
    'compression_method': 'SVD',
    'trunc_params': dmrg_params['trunc_params']}

# %% The run. The code allows to sweep lambda and find multiple low-energy
# states at the same time, but we don't have to.

num_states = 1  # How many low-energy states we are interested in.
init_indices = [0, 0, 0, 0]
# Initial state index; can be tweaked.

list_lambda = np.exp(np.linspace(np.log(0.1), np.log(1000), 41))
# list_lambda = [1.4822688982138956]
# list_lambda = [50]

# list_m_mass = np.exp(np.linspace(np.log(5), np.log(500), 5))
list_m_mass_ratio = [0.1, 1000]#[1, 10, 100]

list_cos_phi_plaquette, list_nn_link = [], []

# for n_lambda, lambda_JJ in enumerate(list_lambda):
for n_sweep, sweep_params in enumerate(itertools.product(
        list_lambda, list_m_mass_ratio)):
    lambda_JJ = sweep_params[0]
    m_mass = lambda_JJ/sweep_params[1]
    g_gauge = 1
    kin_mat = np.eye(4)*(g_gauge+2*m_mass)
    for row, col in [(0, 1), (2, 3)]:
        kin_mat[row, col] = -m_mass
        kin_mat[col, row] = -m_mass
    for row, col in [(0, 2), (1, 3)]:
        kin_mat[row, col] = m_mass
        kin_mat[col, row] = m_mass
    # print(kin_mat)
    list_EJ = array([lambda_JJ]*4)
    model_params = {
        'EJ': list_EJ, 'ECmat': kin_mat/4,
        'truncated_dim': list_trunc_dims, 'ncut': list_ncut,
        'constant_shift': -10,  # Used for finding the excited states.
        # See https://github.com/tenpy/tenpy/issues/329 .
        }
    # %
    model = CoupledTransmonModel(model_params)  # Hamiltonian construction.
    # # Sanity check for Hamiltonian construction:
    # print(
    #     model.all_coupling_terms().to_TermList()
    #     + model.all_onsite_terms().to_TermList())
    if n_sweep == 0:
        psi_dmrg_temp = tenpy.networks.mps.MPS.from_product_state(
            model.lat.mps_sites(), init_indices, unit_cell_width=1)
        # if n_sweep != 0 we reuse the previous state as the initial guess.

    # Finding the low-lying states one at a time.
    list_E, list_psi_dmrg = [], []
    for m_run in range(num_states):
        dmrg_eng = tenpy.algorithms.dmrg.TwoSiteDMRGEngine(
            psi_dmrg_temp, model, dmrg_params, orthogonal_to=list_psi_dmrg)
        # Excited states are found using the orthogonal_to option.
        E_temp, psi_dmrg_temp = dmrg_eng.run()
        list_E.append(E_temp)
        list_psi_dmrg.append(psi_dmrg_temp.copy())
        print(sweep_params, m_run)

    # ham_mpo = HamFactory(model_params).H_MPO
    # ham_mat = calc_mat_elem(ham_mpo, list_psi_dmrg, mpo_options)
    # % Sanity check: Hamiltonian in eigenstate basis
    # print(np.max(np.abs(ham_mat-np.diag(list_E))))
    # print(np.max(np.abs(np.diag(ham_mat)-list_E)))

    # Finding <cos(theta_12+theta_24-theta_13-theta_34)> and <n^2> in the
    # ground state.
    model_params_op = model_params.copy()
    del model_params_op['constant_shift']
    model_params_op['ham_type'] = 'cos_phi_plaquette'
    cos_phi_plaquette_mpo = HamFactory(model_params_op).H_MPO
    list_cos_phi_plaquette.append(
        cos_phi_plaquette_mpo.expectation_value(list_psi_dmrg[0]))
    model_params_op['ham_type'] = 'NN'
    model_params_op['ham_idx'] = 0
    nn_link_mpo = HamFactory(model_params_op).H_MPO
    list_nn_link.append(
        nn_link_mpo.expectation_value(list_psi_dmrg[0]))
    print(n_sweep, sweep_params)
list_cos_phi_plaquette = array(list_cos_phi_plaquette)
list_nn_link = array(list_nn_link)
# %%

list_cos_phi_plaquette_full = array([
    [3.42787149e-06, 5.21932673e-06, 7.66802600e-06, 1.09131227e-05,
     1.51116662e-05, 2.04540665e-05, 2.71697029e-05, 3.55143350e-05,
     4.60389994e-05, 5.89373055e-05, 7.50598634e-05, 9.55759123e-05,
     1.20294608e-04, 1.52482260e-04, 1.92370154e-04, 2.42082984e-04,
     3.03656924e-04, 3.84179289e-04, 4.80897577e-04, 6.07333700e-04,
     7.65171539e-04, 9.62990896e-04, 1.21598256e-03, 1.52880161e-03,
     1.92792854e-03, 2.42871176e-03, 3.05902931e-03, 3.85729050e-03,
     4.85878066e-03, 6.11284009e-03, 7.70733627e-03, 9.70128369e-03,
     1.22165685e-02, 1.53945588e-02, 1.93868663e-02, 2.44066421e-02,
     3.04793047e-02, 3.77298463e-02, 4.64292293e-02, 5.61672676e-02,
     6.66350143e-02],  # lambda/m=0.1
    [5.63771156e-05, 1.23257833e-04, 2.61465388e-04, 5.35654920e-04,
     1.05512541e-03, 1.99075112e-03, 3.58770237e-03, 6.16715062e-03,
     1.01126962e-02, 1.58439319e-02, 2.37870660e-02, 3.43558853e-02,
     4.79528019e-02, 6.49909088e-02, 8.59281704e-02, 1.11297749e-01,
     1.41714415e-01, 1.77833879e-01, 2.20240076e-01, 2.69236355e-01,
     3.24544688e-01, 3.84979471e-01, 4.48275019e-01, 5.11306252e-01,
     5.70800146e-01, 6.24243639e-01, 6.70484969e-01, 7.09689143e-01,
     7.42836036e-01, 7.71149748e-01, 7.95713830e-01, 8.17337062e-01,
     8.36564938e-01, 8.53758268e-01, 8.69168829e-01, 8.82989244e-01,
     8.95381633e-01, 9.06488477e-01, 9.16440836e-01, 9.25348786e-01,
     9.33328966e-01],  # lambda/m=1
    [9.11333902e-05, 2.21268296e-04, 5.29780432e-04, 1.24347288e-03,
     2.83820021e-03, 6.23439903e-03, 1.30160172e-02, 2.54871019e-02,
     4.62573763e-02, 7.72293230e-02, 1.18479516e-01, 1.67954122e-01,
     2.22319639e-01, 2.78269987e-01, 3.33385104e-01, 3.86282083e-01,
     4.36348752e-01, 4.83406164e-01, 5.27466947e-01, 5.68608114e-01,
     6.06920782e-01, 6.42497839e-01, 6.75435209e-01, 7.05835332e-01,
     7.33809033e-01, 7.59475387e-01, 7.82960221e-01, 8.04394005e-01,
     8.23909600e-01, 8.41640123e-01, 8.57717045e-01, 8.72268523e-01,
     8.85418075e-01, 8.97283542e-01, 9.07976358e-01, 9.17601177e-01,
     9.26255632e-01, 9.34030092e-01, 9.41008183e-01, 9.47266293e-01,
     9.52874043e-01],  # lambda/m=10
    [9.60270766e-05, 2.36099797e-04, 5.73835691e-04, 1.37050485e-03,
     3.18901934e-03, 7.14571708e-03, 1.51933314e-02, 3.01536592e-02,
     5.50231226e-02, 9.14339223e-02, 1.38268679e-01, 1.91903077e-01,
     2.48045327e-01, 3.03498971e-01, 3.56673064e-01, 4.07117930e-01,
     4.54870399e-01, 5.00039817e-01, 5.42665884e-01, 5.82726584e-01,
     6.20186674e-01, 6.55034181e-01, 6.87294836e-01, 7.17031565e-01,
     7.44337582e-01, 7.69328146e-01, 7.92133052e-01, 8.12890429e-01,
     8.31741850e-01, 8.48828600e-01, 8.64288914e-01, 8.78256012e-01,
     8.90856767e-01, 9.02210873e-01, 9.12430393e-01, 9.21619591e-01,
     9.29874968e-01, 9.37285449e-01, 9.43932662e-01, 9.49891278e-01,
     9.55229338e-01],  # lambda/m=100
    [9.65353451e-05, 2.37654146e-04, 5.78501748e-04, 1.38412256e-03,
     3.22711637e-03, 7.24592609e-03, 1.54351099e-02, 3.06738926e-02,
     5.59949643e-02, 9.29810918e-02, 1.40359060e-01, 1.94330061e-01,
     2.50532320e-01, 3.05832875e-01, 3.58767124e-01, 4.08993005e-01,
     4.56599715e-01, 5.01699567e-01, 5.44309604e-01, 5.84381195e-01,
     6.21859600e-01, 6.56722051e-01, 6.88989451e-01, 7.18722880e-01,
     7.46014944e-01, 7.70980813e-01, 7.93750515e-01, 8.14462701e-01,
     8.33259751e-01, 8.50284024e-01, 8.65675049e-01, 8.79567500e-01,
     8.92089790e-01, 9.03363158e-01, 9.13501145e-01, 9.22609362e-01,
     9.30785484e-01, 9.38119408e-01, 9.44693533e-01, 9.50583113e-01,
     9.55856671e-01],  # lambda/m=1000
    [9.65920398e-05, 2.37827679e-04, 5.79023269e-04, 1.38564652e-03,
     3.23138565e-03, 7.25717060e-03, 1.54622679e-02, 3.07323504e-02,
     5.61040938e-02, 9.31544671e-02, 1.40592484e-01, 1.94599802e-01,
     2.50807307e-01, 3.06089828e-01, 3.58997352e-01, 4.09199898e-01,
     4.56792248e-01, 5.01886667e-01, 5.44497385e-01, 5.84572721e-01,
     6.22055858e-01, 6.56922996e-01, 6.89194647e-01, 7.18931783e-01,
     7.46226984e-01, 7.71195409e-01, 7.93967072e-01, 8.14680603e-01,
     8.33478352e-01, 8.50502639e-01, 8.65892953e-01, 8.79783923e-01,
     8.92303919e-01, 9.03574146e-01, 9.13708120e-01, 9.22811449e-01,
     9.30981826e-01, 9.38309193e-01, 9.44876017e-01, 9.50757647e-01,
     9.56022722e-01],  # lambda/m=infty
    ])
array([
       ])
list_nn_link_full = array([
    [5.55150338e-04, 6.39907523e-04, 7.21911194e-04, 7.98870189e-04,
     8.69165012e-04, 9.31871476e-04, 9.86681744e-04, 1.03376944e-03,
     1.07364020e-03, 1.10699562e-03, 1.13462454e-03, 1.15732487e-03,
     1.17585305e-03, 1.19089602e-03, 1.20305801e-03, 1.21285901e-03,
     1.22073916e-03, 1.22706839e-03, 1.23215113e-03, 1.23624656e-03,
     1.23956766e-03, 1.24229967e-03, 1.24461537e-03, 1.24666882e-03,
     1.24865104e-03, 1.25076993e-03, 1.25331694e-03, 1.25672739e-03,
     1.26160857e-03, 1.26889881e-03, 1.28024358e-03, 1.29785981e-03,
     1.32563338e-03, 1.36967817e-03, 1.43925557e-03, 1.54932201e-03,
     1.71616677e-03, 1.96376290e-03, 2.33045465e-03, 2.83120471e-03,
     3.47611877e-03],  # lambda/m=0.1
    [3.44172899e-03, 4.99360938e-03, 7.11388032e-03, 9.92621274e-03,
     1.35360011e-02, 1.80088726e-02, 2.33526044e-02, 2.95091514e-02,
     3.63604816e-02, 4.37468949e-02, 5.14927999e-02, 5.94347063e-02,
     6.74487384e-02, 7.54779395e-02, 8.35613943e-02, 9.18676582e-02,
     1.00734387e-01, 1.10713464e-01, 1.22613873e-01, 1.37518998e-01,
     1.56733334e-01, 1.81598039e-01, 2.13148720e-01, 2.51708875e-01,
     2.96671275e-01, 3.46701928e-01, 4.00350333e-01, 4.56694227e-01,
     5.15680867e-01, 5.78084503e-01, 6.45209378e-01, 7.18587350e-01,
     7.99752073e-01, 8.90165408e-01, 9.91245340e-01, 1.10443033e+00,
     1.23126146e+00, 1.37342816e+00, 1.53286020e+00, 1.71155972e+00,
     1.91215147e+00],  # lambda/m=1
    [4.74611831e-03, 7.39421041e-03, 1.14471053e-02, 1.75660724e-02,
     2.66307240e-02, 3.97157838e-02, 5.79681711e-02, 8.23494636e-02,
     1.13283916e-01, 1.50373404e-01, 1.92392151e-01, 2.37631933e-01,
     2.84416032e-01, 3.31495846e-01, 3.78180285e-01, 4.24247511e-01,
     4.69775103e-01, 5.14994287e-01, 5.60207594e-01, 6.05762615e-01,
     6.52057612e-01, 6.99557013e-01, 7.48803932e-01, 8.00425482e-01,
     8.55131927e-01, 9.13712864e-01, 9.77033615e-01, 1.04603416e+00,
     1.12173189e+00, 1.20522845e+00, 1.29772058e+00, 1.40051420e+00,
     1.51504141e+00, 1.64287985e+00, 1.78577424e+00, 1.94566130e+00,
     2.12469689e+00, 2.32528013e+00, 2.55009445e+00, 2.80212088e+00,
     3.08466552e+00],  # lambda/m=10
    [4.91593662e-03, 7.72536954e-03, 1.20863854e-02, 1.87821150e-02,
     2.88955325e-02, 4.38109958e-02, 6.50842519e-02, 9.41041721e-02,
     1.31574904e-01, 1.77072102e-01, 2.29066029e-01, 2.85552264e-01,
     3.44919796e-01, 4.06504682e-01, 4.70608320e-01, 5.38152834e-01,
     6.10258152e-01, 6.87925671e-01, 7.71879455e-01, 8.62530791e-01,
     9.60005076e-01, 1.06418421e+00, 1.17474662e+00, 1.29120893e+00,
     1.41298038e+00, 1.53943727e+00, 1.67001664e+00, 1.80432064e+00,
     1.94221776e+00, 2.08392636e+00, 2.23006917e+00, 2.38169414e+00,
     2.54026486e+00, 2.70763005e+00, 2.88598393e+00, 3.07782891e+00,
     3.28594866e+00, 3.51339616e+00, 3.76349769e+00, 4.03987132e+00,
     4.34645491e+00],  # lambda/m=100
    [4.93341780e-03, 7.75970297e-03, 1.21532399e-02, 1.89106074e-02,
     2.91377384e-02, 4.42549105e-02, 6.58668988e-02, 9.54163160e-02,
     1.33647030e-01, 1.80143270e-01, 2.33361802e-01, 2.91307967e-01,
     3.52460225e-01, 4.16357089e-01, 4.83618271e-01, 5.55589402e-01,
     6.33914164e-01, 7.20230080e-01, 8.16040745e-01, 9.22722106e-01,
     1.04158684e+00, 1.17394525e+00, 1.32113516e+00, 1.48451940e+00,
     1.66545856e+00, 1.86526498e+00, 2.08514061e+00, 2.32610028e+00,
     2.58888417e+00, 2.87386687e+00, 3.18097540e+00, 3.50963354e+00,
     3.85875329e+00, 4.22679344e+00, 4.61189968e+00, 5.01212799e+00,
     5.42573668e+00, 5.85151449e+00, 6.28909971e+00, 6.73924350e+00,
     7.20398150e+00],  # lambda/m=1000
    [4.93536591e-03, 7.76353191e-03, 1.21607024e-02, 1.89249656e-02,
     2.91648380e-02, 4.43046510e-02, 6.59547345e-02, 9.55638263e-02,
     1.33880392e-01, 1.80489838e-01, 2.33847813e-01, 2.91961607e-01,
     3.53321457e-01, 4.17491824e-01, 4.85133814e-01, 5.57650253e-01,
     6.36759864e-01, 7.24198840e-01, 8.21603850e-01, 9.30534585e-01,
     1.05255997e+00, 1.18934724e+00, 1.34272825e+00, 1.51474448e+00,
     1.70768155e+00, 1.92410161e+00, 2.16687789e+00, 2.43923298e+00,
     2.74478169e+00, 3.08757911e+00, 3.47217445e+00, 3.90367142e+00,
     4.38779603e+00, 4.93097254e+00, 5.54040875e+00, 6.22419164e+00,
     6.99139472e+00, 7.85219837e+00, 8.81802502e+00, 9.90169065e+00,
     1.11175749e+01],  # lambda/m=infty
    ])
# %%
fig, ax = plt.subplots(2, sharex=True)
list_lambda_m_ratio = [0.1, 1, 10, 100, 1000, r'$\infty$']
list_colors = [
    (0, 0, 1, 0.2), (0, 0, 1, 0.4), (0, 0, 1, 0.6), (0, 0, 1, 0.8),
    (0, 0, 1, 1), (0, 0, 1, 0.1)]

for idx_mass in range(len(list_lambda_m_ratio)-1):
    ax[0].plot(
        list_lambda, list_cos_phi_plaquette_full[idx_mass],
        c=list_colors[idx_mass])
    ax[1].plot(
        list_lambda, list_nn_link_full[idx_mass],
        label=r'$\lambda/m=$'+str(list_lambda_m_ratio[idx_mass]),
        c=list_colors[idx_mass])

ax[0].plot(
    list_lambda, list_cos_phi_plaquette_full[-1],
    c='k', ls='--')
ax[1].plot(
    list_lambda, list_nn_link_full[-1],
    label=r'$\lambda/m=$'+str(list_lambda_m_ratio[-1]),
    c='k', ls='--')

ax[0].set_xscale('log')
ax[0].set_xlim(0.1, 1000)
ax[0].set_ylim(-0.01, 1.01)
ax[0].set_yticks([0, 0.5, 1])
ax[0].set_ylabel(r'$\langle \hat{\blacksquare} \rangle$')
ax[0].grid()
ax[1].set_ylim(-0.01, 10.01)
ax[1].set_yticks([0, 5, 10])
ax[1].grid()
ax[1].legend()
ax[1].set_ylabel(r'$\langle \hat{n}_{12}^2 \rangle$')
ax[1].set_xlabel(r'$\lambda/g$')
# %% Preparing the vortex state as an MPS.
model_params_op = model_params.copy()
del model_params_op['constant_shift']
model_params_op['ham_type'] = 'exp_i_theta_n_plaquette'
model_params_op['op_theta'] = 0.2
exp_i_theta_n_plaquette_mpo = HamFactory(model_params_op).H_MPO

exp_i_theta_n_plaquette_mat = calc_mat_elem(
    exp_i_theta_n_plaquette_mpo, list_psi_dmrg, mpo_options)

ground_state = list_psi_dmrg[0].copy()
exp_i_theta_n_plaquette_mpo.apply(ground_state, mpo_options)
norm_exp_i_theta_n_state = ground_state.norm
# %%
print('Weight of vortex state in low-lying subspace: ',
      np.sum(np.abs(exp_i_theta_n_plaquette_mat[0])**2))
print('Normalization of vortex state as an MPS: ', norm_exp_i_theta_n_state**2)


# %%

# %% Saving the MPS and the metadata.
# h5py is recommended by tenpy. Since the Site objects are also saved, we
# should not build weird objects (e.g. scq.Transmon) into them.
savedata = {
    "list_E": list_E,
    "list_psi_dmrg": list_psi_dmrg,
    "model": model,
    "dmrg_params": dmrg_params,
    "model_params": model_params}
filename = 'something.h5'
with h5py.File(filename, 'w') as file:
    tenpy.tools.hdf5_io.save_to_hdf5(file, savedata)


# %% Loading the MPS and the metadata.
filename = 'something.h5'
# filename = "tenpy_100_(30, 30, 30, 30)_60_pi_2.h5"
with h5py.File(filename, 'r') as file:
    loaddata = tenpy.tools.hdf5_io.load_from_hdf5(file)
    list_psi_dmrg = loaddata['list_psi_dmrg']
    list_E = loaddata['list_E']
    model = loaddata['model']
    dmrg_params = loaddata['dmrg_params']
    model_params = loaddata['model_params']
# %%




# %% OBSOLETE
# % Storage: list_cos_phi_plaquette and list_nn_link
list_lambda = np.exp(np.linspace(np.log(0.1), np.log(800), 51))
list_cos_phi_plaquette_full, list_nn_link_full = [], []

# m_mass, g_gauge, list_ncut, list_trunc_dims = 0, 1, [100]*4, [20]*4
# In the m_mass=0 limit, list_trunc_dims does not matter
list_cos_phi_plaquette_full.append(
    array([9.65920398e-05, 1.95326187e-04, 3.92543117e-04, 7.82071242e-04,
       1.53950909e-03, 2.98108168e-03, 5.64636359e-03, 1.03892129e-02,
       1.84254648e-02, 3.12437063e-02, 5.02881204e-02, 7.64405206e-02,
       1.09540537e-01, 1.48283526e-01, 1.90626188e-01, 2.34456396e-01,
       2.78130391e-01, 3.20663598e-01, 3.61629481e-01, 4.00939677e-01,
       4.38644211e-01, 4.74808321e-01, 5.09462838e-01, 5.42601298e-01,
       5.74196633e-01, 6.04219631e-01, 6.32651431e-01, 6.59489117e-01,
       6.84746444e-01, 7.08452061e-01, 7.30646850e-01, 7.51381222e-01,
       7.70712666e-01, 7.88703646e-01, 8.05419833e-01, 8.20928647e-01,
       8.35298062e-01, 8.48595650e-01, 8.60887824e-01, 8.72239250e-01,
       8.82712413e-01, 8.92367292e-01, 9.01261140e-01, 9.09448351e-01,
       9.16980383e-01, 9.23905744e-01, 9.30270017e-01, 9.36115912e-01,
       9.41483352e-01, 9.46409572e-01, 9.50929233e-01])
    )
list_nn_link_full.append(
    array([4.93536591e-03, 7.03118713e-03, 9.99365867e-03, 1.41579282e-02,
       1.99664624e-02, 2.79829868e-02, 3.88895233e-02, 5.34503952e-02,
       7.24263681e-02, 9.64338944e-02, 1.25775014e-01, 1.60304145e-01,
       1.99415841e-01, 2.42196667e-01, 2.87695305e-01, 3.35195875e-01,
       3.84388148e-01, 4.35397933e-01, 4.88708956e-01, 5.45035627e-01,
       6.05198767e-01, 6.70034888e-01, 7.40348741e-01, 8.16904045e-01,
       9.00439425e-01, 9.91695328e-01, 1.09144117e+00, 1.20049739e+00,
       1.31975158e+00, 1.45017050e+00, 1.59280989e+00, 1.74882372e+00,
       1.91947383e+00, 2.10614013e+00, 2.31033189e+00, 2.53369989e+00,
       2.77804985e+00, 3.04535698e+00, 3.33778200e+00, 3.65768861e+00,
       4.00766256e+00, 4.39053263e+00, 4.80939343e+00, 5.26763045e+00,
       5.76894741e+00, 6.31739624e+00, 6.91740975e+00, 7.57383756e+00,
       8.29198520e+00, 9.07765706e+00, 9.93720330e+00])
    )

# m_mass, g_gauge, list_ncut, list_trunc_dims = 0.1, 1, [100]*4, [20]*4
list_cos_phi_plaquette_full.append(
    array([5.63771145e-05, 1.14493199e-04, 2.31484814e-04, 4.65091700e-04,
       9.26266621e-04, 1.82236600e-03, 3.52598403e-03, 6.67061723e-03,
       1.22528086e-02, 2.16769390e-02, 3.66293989e-02, 5.86783284e-02,
       8.86489658e-02, 1.26085197e-01, 1.69207629e-01, 2.15483255e-01,
       2.62457418e-01, 3.08356251e-01, 3.52240563e-01, 3.93823451e-01,
       4.33179128e-01, 4.70502705e-01, 5.05973523e-01, 5.39707261e-01,
       5.71760157e-01, 6.02152736e-01, 6.30893305e-01, 6.57993454e-01,
       6.83475303e-01, 7.07373153e-01, 7.29732325e-01, 7.50607007e-01,
       7.70057989e-01, 7.88150634e-01, 8.04953147e-01, 8.20535157e-01,
       8.34966553e-01, 8.48316562e-01, 8.60653024e-01, 8.72041830e-01,
       8.82546513e-01, 8.92227950e-01, 9.01144159e-01, 9.09350183e-01,
       9.16898034e-01, 9.23836690e-01, 9.30212129e-01, 9.36067399e-01,
       9.41442706e-01, 9.46375526e-01, 9.50900721e-01])
    )
list_nn_link_full.append(
    array([3.44172899e-03, 4.91197839e-03, 6.99904824e-03, 9.95031579e-03,
       1.41010843e-02, 1.98950091e-02, 2.78990154e-02, 3.88016767e-02,
       5.33784817e-02, 7.24059842e-02, 9.65180042e-02, 1.26028449e-01,
       1.60789403e-01, 2.00175014e-01, 2.43239840e-01, 2.89003135e-01,
       3.36734251e-01, 3.86125406e-01, 4.37316161e-01, 4.90807059e-01,
       5.47326605e-01, 6.07704875e-01, 6.72783270e-01, 7.43368659e-01,
       8.20225715e-01, 9.04094006e-01, 9.95715486e-01, 1.09586179e+00,
       1.20535621e+00, 1.32508974e+00, 1.45603296e+00, 1.59924581e+00,
       1.75588691e+00, 1.92722315e+00, 2.11464002e+00, 2.31965284e+00,
       2.54391903e+00, 2.78925156e+00, 3.05763359e+00, 3.35123453e+00,
       3.67242757e+00, 4.02380889e+00, 4.40821862e+00, 4.82876381e+00,
       5.28884358e+00, 5.79217655e+00, 6.34283092e+00, 6.94525735e+00,
       7.60432493e+00, 8.32536057e+00, 9.11419198e+00])
    )

# m_mass, g_gauge, list_ncut, list_trunc_dims = 1, 1, [100]*4, [20]*4
list_cos_phi_plaquette_full.append(
    array([3.42791536e-06, 7.02334926e-06, 1.43794814e-05, 2.94094621e-05,
       6.00596964e-05, 1.22392256e-04, 2.48659409e-04, 5.03009902e-04,
       1.01129904e-03, 2.01561828e-03, 3.96852797e-03, 7.68157533e-03,
       1.45238317e-02, 2.66033999e-02, 4.67377099e-02, 7.78858826e-02,
       1.21832771e-01, 1.77602315e-01, 2.40957795e-01, 3.05948369e-01,
       3.67471689e-01, 4.22814989e-01, 4.71508543e-01, 5.14345931e-01,
       5.52491925e-01, 5.86991941e-01, 6.18610203e-01, 6.47842845e-01,
       6.74992993e-01, 7.00247976e-01, 7.23735846e-01, 7.45558755e-01,
       7.65809210e-01, 7.84576407e-01, 8.01947905e-01, 8.18009567e-01,
       8.32845047e-01, 8.46535262e-01, 8.59157977e-01, 8.70787505e-01,
       8.81494515e-01, 8.91345923e-01, 9.00404858e-01, 9.08730681e-01,
       9.16379047e-01, 9.23402008e-01, 9.29848134e-01, 9.35762654e-01,
       9.41187612e-01, 9.46162028e-01, 9.50722063e-01])
    )
list_nn_link_full.append(
    array([5.55150338e-04, 7.95054672e-04, 1.13847601e-03, 1.62991667e-03,
       2.33283667e-03, 3.33754464e-03, 4.77217458e-03, 6.81773680e-03,
       9.72829420e-03, 1.38570300e-02, 1.96878314e-02, 2.78690259e-02,
       3.92394007e-02, 5.48244941e-02, 7.57630851e-02, 1.03109374e-01,
       1.37481468e-01, 1.78652477e-01, 2.25382746e-01, 2.75792406e-01,
       3.28139282e-01, 3.81454175e-01, 4.35680004e-01, 4.91433833e-01,
       5.49677897e-01, 6.11469005e-01, 6.77820470e-01, 7.49652567e-01,
       8.27798344e-01, 9.13036942e-01, 1.00613428e+00, 1.10787901e+00,
       1.21910927e+00, 1.34073092e+00, 1.47373029e+00, 1.61918431e+00,
       1.77827001e+00, 1.95227426e+00, 2.14260429e+00, 2.35079912e+00,
       2.57854201e+00, 2.82767412e+00, 3.10020940e+00, 3.39835088e+00,
       3.72450851e+00, 4.08131862e+00, 4.47166525e+00, 4.89870349e+00,
       5.36588493e+00, 5.87698561e+00, 6.43613654e+00])
    )

# m_mass, g_gauge, list_ncut, list_trunc_dims = 10, 1, [100]*4, [20]*4
list_cos_phi_plaquette_full.append(
    array([7.59669309e-09, 1.55963141e-08, 3.17242743e-08, 6.49045605e-08,
       1.34699761e-07, 2.76499272e-07, 5.67594739e-07, 1.16432824e-06,
       2.38041110e-06, 4.89415427e-06, 9.97568616e-06, 2.03125225e-05,
       4.09682091e-05, 8.53591331e-05, 1.76302141e-04, 3.60674316e-04,
       7.35082251e-04, 1.49294716e-03, 3.01906121e-03, 6.06538494e-03,
       1.20719987e-02, 2.37013107e-02, 4.56138773e-02, 8.51633397e-02,
       1.51625342e-01, 2.50465768e-01, 3.71509863e-01, 4.87245712e-01,
       5.76313790e-01, 6.38168070e-01, 6.81807612e-01, 7.14992210e-01,
       7.42209237e-01, 7.65713039e-01, 7.86593731e-01, 8.05399856e-01,
       8.22443157e-01, 8.37933347e-01, 8.52032964e-01, 8.64878607e-01,
       8.76589335e-01, 8.87270638e-01, 8.97016726e-01, 9.05912189e-01,
       9.14033244e-01, 9.21448772e-01, 9.28221159e-01, 9.34406986e-01,
       9.40057668e-01, 9.45219955e-01, 9.49936406e-01])
    )
list_nn_link_full.append(
    array([1.13379522e-05, 1.62427519e-05, 2.32694007e-05, 3.33358586e-05,
       4.77572515e-05, 6.84177468e-05, 9.80168492e-05, 1.40422380e-04,
       2.01176433e-04, 2.88220794e-04, 4.12937326e-04, 5.91640944e-04,
       8.47723389e-04, 1.21473526e-03, 1.74082375e-03, 2.49514085e-03,
       3.57714502e-03, 5.13024292e-03, 7.36225424e-03, 1.05776306e-02,
       1.52331486e-02, 2.20484476e-02, 3.22556924e-02, 4.81746289e-02,
       7.42870450e-02, 1.17867929e-01, 1.84984330e-01, 2.70844013e-01,
       3.60687516e-01, 4.44354609e-01, 5.21176926e-01, 5.94800059e-01,
       6.69102058e-01, 7.46993974e-01, 8.30459727e-01, 9.20875301e-01,
       1.01929840e+00, 1.12666429e+00, 1.24389796e+00, 1.37197165e+00,
       1.51193297e+00, 1.66492105e+00, 1.83217793e+00, 2.01505978e+00,
       2.21504795e+00, 2.43376140e+00, 2.67296996e+00, 2.93460831e+00,
       3.22079216e+00, 3.53383505e+00, 3.87626713e+00])
    )

# m_mass, g_gauge, list_ncut, list_trunc_dims = 100, 1, [100]*4, [20]*4
list_cos_phi_plaquette_full.append(
    array([3.53757029e-13, 7.83739981e-13, 1.48279621e-12, 3.30680279e-12,
       6.72467819e-12, 1.39443029e-11, 2.87252181e-11, 5.88273225e-11,
       1.21057065e-10, 2.88163504e-09, 3.12547810e-09, 1.83972297e-08,
       3.28715231e-08, 5.89413859e-08, 1.80862928e-07, 3.21002801e-07,
       7.56806620e-07, 1.40622629e-06, 2.27660391e-06, 3.58024386e-06,
       6.88654609e-06, 2.10470485e-05, 4.53409095e-05, 1.07413915e-04,
       2.34554468e-04, 4.83724309e-04, 1.00857058e-03, 2.06801551e-03,
       4.22667931e-03, 8.61064089e-03, 1.74594731e-02, 3.51891867e-02,
       7.01865137e-02, 1.37145002e-01, 2.55461210e-01, 4.26595834e-01,
       5.96637036e-01, 7.10345146e-01, 7.76040208e-01, 8.16686152e-01,
       8.44250735e-01, 8.64354570e-01, 8.79986431e-01, 8.92793800e-01,
       9.03681763e-01, 9.13155812e-01, 9.21515225e-01, 9.28952030e-01,
       9.35602632e-01, 9.41570187e-01, 9.46939133e-01])
    )
list_nn_link_full.append(
    array([1.23759330e-07, 1.77297076e-07, 2.53995025e-07, 3.63872187e-07,
       5.21281768e-07, 7.46786129e-07, 1.06984287e-06, 1.53265292e-06,
       2.19567335e-06, 3.14551517e-06, 4.50625778e-06, 6.45565867e-06,
       9.24837626e-06, 1.32492412e-05, 1.89809230e-05, 2.71922317e-05,
       3.89559720e-05, 5.58091850e-05, 7.99540901e-05, 1.14546251e-04,
       1.64107332e-04, 2.35117960e-04, 3.36867313e-04, 4.82676080e-04,
       6.91657236e-04, 9.91267772e-04, 1.42109398e-03, 2.03863092e-03,
       2.92920421e-03, 4.22635392e-03, 6.16551814e-03, 9.25652176e-03,
       1.48838333e-02, 2.73333613e-02, 5.92203703e-02, 1.35965870e-01,
       2.69363816e-01, 4.26418898e-01, 5.80431067e-01, 7.29597779e-01,
       8.75342701e-01, 1.01903427e+00, 1.16364549e+00, 1.31298340e+00,
       1.47064269e+00, 1.63961382e+00, 1.82235435e+00, 2.02095701e+00,
       2.23736735e+00, 2.47347535e+00, 2.73128535e+00])
    )
# %% Plotting <cos(theta_12+theta_24-theta_13-theta_34)> and <n^2> in the
# ground state (see next cell).
fig, ax = plt.subplots(2, sharex=True)
list_m_mass = [0, 0.1, 1, 10, 100]

for idx_mass in range(len(list_m_mass)):
    ax[0].plot(
        list_lambda, list_cos_phi_plaquette_full[idx_mass],
        label='m='+str(list_m_mass[idx_mass]))
    ax[1].plot(list_lambda, list_nn_link_full[idx_mass])


ax[0].set_xscale('log')
ax[0].set_xlim(0.1, 800)
ax[0].set_ylim(-0.01, 1.01)
ax[0].set_ylabel(r'$\langle \hat{\blacksquare} \rangle$')
ax[0].grid()
ax[1].set_ylim(-0.01, 13)
ax[1].set_yticks([0, 3, 6, 9, 12])
ax[1].grid()
ax[0].legend()
ax[1].set_ylabel(r'$\langle n_{12}^2 \rangle$')
ax[1].set_xlabel(r'$\lambda$')
fig.suptitle('g=1')









