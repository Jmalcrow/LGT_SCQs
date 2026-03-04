#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 14:49:51 2026

I'm stupidly using a trivial lattice with a large unit cell. In this case it's
probably excusable since we don't go to too large lattices.

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


# %% Figuring out the mass couplings.
num_row, num_col = 2, 2
counter = 0
list_site = []
dict_site = {}
for row in range(num_row+1):
    for col in range(num_col):
        loc_tup = (2*col+1, 2*row)
        list_site.append(loc_tup)
        dict_site[loc_tup] = counter
        counter += 1
    if row != num_row:
        for col in range(num_col+1):
            loc_tup = (2*col, 2*row+1)
            list_site.append(loc_tup)
            dict_site[loc_tup] = counter
            counter += 1
num_site = len(list_site)
m_coeff_mat = np.zeros((num_site, num_site), dtype=int)
for n_site, site in enumerate(list_site):
    list_neighbor = [
        ((site[0]-1, site[1]+1), 1),
        ((site[0]-1, site[1]-1), -1),
        ((site[0]+1, site[1]+1), -1),
        ((site[0]+1, site[1]-1), 1),
        ]
    if site[0] % 2 == 0:  # Vertical links.
        list_neighbor += [
            ((site[0], site[1]-2), -1), ((site[0], site[1]+2), -1)]
    else:  # Horizontal links.
        list_neighbor += [
            ((site[0]-2, site[1]), -1), ((site[0]+2, site[1]), -1)]
    for neighbor_loc, m_sign in list_neighbor:
        temp_idx = dict_site.get(neighbor_loc, -1)
        if temp_idx != -1:
            m_coeff_mat[n_site, temp_idx] = m_sign
            # print(list_sites[n_site], list_sites[temp_idx], m_sign)

# # The following gives a picture.
fig, ax = plt.subplots()
ax.scatter(array(list_site)[:, 0], array(list_site)[:, 1])

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
            assert (ham_idx[0] < num_col) and (ham_idx[1] < num_row)
            # ham_idx is x, y rather than row, col.
            plaquette_idx = [dict_site.get(loc_tup, -1) for loc_tup in [
                (2*ham_idx[0]+1, 2*ham_idx[1]),
                (2*ham_idx[0], 2*ham_idx[1]+1),
                (2*ham_idx[0]+2, 2*ham_idx[1]+1),
                (2*ham_idx[0]+1, 2*ham_idx[1]+2)]]
            assert -1 not in plaquette_idx
            # This is semi-hard-wired! # !!!
            # Needs proper generalization to multiple plaquettes.
            self.add_multi_coupling_term(
                1.,
                # [0, 2, 3, 5], # [1, 3, 4, 6],
                plaquette_idx,
                ['exp_i_phi', 'exp_mi_phi', 'exp_i_phi', 'exp_mi_phi'],
                'Id',
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



# import logging.config
# conf = {
#     'version': 1,
#     'disable_existing_loggers': False,
#     'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
#     'handlers': {'to_file': {'class': 'logging.FileHandler',
#                              'filename': 'output_filename.log',
#                              'formatter': 'custom',
#                              'level': 'INFO',
#                              'mode': 'a'},
#                 'to_stdout': {'class': 'logging.StreamHandler',
#                               'formatter': 'custom',
#                               'level': 'INFO',
#                               'stream': 'ext://sys.stdout'}},
#     'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
# }
# logging.config.dictConfig(conf)
# %% Model parameters.


list_ncut = [100]*num_site
list_trunc_dims = [5]*num_site


dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-8,
    'trunc_params': {
        'chi_max': 500,
        'svd_min': 1.e-8,
    },
    # 'verbose': True,
    'combine': True,
}
mpo_options = {
    'compression_method': 'SVD',
    'trunc_params': dmrg_params['trunc_params']}

# % The run. The code allows to sweep lambda and find multiple low-energy
# states at the same time, but we don't have to.

num_states = 1  # How many low-energy states we are interested in.
init_indices = [0]*num_site
# Initial state index; can be tweaked.

# list_lambda = np.exp(np.linspace(np.log(1), np.log(1e3), 16))
list_lambda = np.exp(np.linspace(np.log(0.1), np.log(1e3), 21))
# list_lambda = [6.309573444801933]
# list_lambda = [31.622776601683793]

# list_m_mass = np.exp(np.linspace(np.log(5), np.log(500), 5))
list_m_mass = [1]

# list_E_full, list_psi_dmrg_full = [], []
list_cos_phi_plaquette, list_nn_link = [], []

# for n_lambda, lambda_JJ in enumerate(list_lambda):
for n_sweep, sweep_params in enumerate(itertools.product(
        list_lambda, list_m_mass)):
    lambda_JJ = sweep_params[0]
    # m_mass = sweep_params[1]
    m_mass = lambda_JJ/100  # !!!
    g_gauge = 1
    kin_mat = np.eye(num_site)*(g_gauge+2*m_mass)+m_coeff_mat*m_mass
    # for row, col in [(0, 1), (0, 3), (1, 4), (2, 5), (3, 6), (5, 6)]:
    #     kin_mat[row, col] = -m_mass
    #     kin_mat[col, row] = -m_mass
    # for row, col in [(0, 2), (1, 3), (3, 5), (4, 6)]:
    #     kin_mat[row, col] = m_mass
    #     kin_mat[col, row] = m_mass
    # print(kin_mat)
    list_EJ = array([lambda_JJ]*num_site)
    model_params = {
        'EJ': list_EJ, 'ECmat': kin_mat/4,
        'truncated_dim': list_trunc_dims, 'ncut': list_ncut,
        'constant_shift': -50,  # Used for finding the excited states.
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
    # list_E_full.append(list_E)
    # list_psi_dmrg_full.append(list_psi_dmrg)
    # ham_mpo = HamFactory(model_params).H_MPO
    # ham_mat = calc_mat_elem(ham_mpo, list_psi_dmrg, mpo_options)
    # % Sanity check: Hamiltonian in eigenstate basis
    # print(np.max(np.abs(ham_mat-np.diag(list_E))))
    # print(np.max(np.abs(np.diag(ham_mat)-list_E)))
    # Finding <cos plaquette> and <n^2> in the ground state.
    model_params_op = model_params.copy()
    del model_params_op['constant_shift']
    model_params_op['ham_type'] = 'cos_phi_plaquette'
    for ham_idx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        model_params_op['ham_idx'] = ham_idx
        cos_phi_plaquette_mpo = HamFactory(model_params_op).H_MPO
        list_cos_phi_plaquette.append(
            cos_phi_plaquette_mpo.expectation_value(list_psi_dmrg[0]))
    model_params_op['ham_type'] = 'NN'
    for ham_idx in range(num_site):
        model_params_op['ham_idx'] = ham_idx
        nn_link_mpo = HamFactory(model_params_op).H_MPO
        list_nn_link.append(
            nn_link_mpo.expectation_value(list_psi_dmrg[0]))
    print(n_sweep, sweep_params)
list_cos_phi_plaquette = array(list_cos_phi_plaquette)
list_nn_link = array(list_nn_link)
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

# %% Storage: list_cos_phi_plaquette and list_nn_link for 2x2 plaquettes
list_lambda = np.exp(np.linspace(np.log(0.1), np.log(1e3), 21))
list_lambda_over_m = [1, 10, 100, r'$\infty$']
list_cos_phi_plaquette_full, list_nn_link_full = [], []

# g_gauge, list_ncut, list_trunc_dims = 1, [100]*num_site, [5]*num_site
# lambda_JJ/m_mass = 1
list_cos_phi_plaquette_full.append(array([
    [5.63951973e-05, 5.63952280e-05, 5.63953352e-05, 5.63952028e-05],  # 1e-1
    [2.61838719e-04, 2.61840210e-04, 2.61840504e-04, 2.61839972e-04],
    [1.06063864e-03, 1.06063897e-03, 1.06063892e-03, 1.06063861e-03],
    [3.64401377e-03, 3.64401372e-03, 3.64401366e-03, 3.64401381e-03],
    [1.05023877e-02, 1.05023877e-02, 1.05023877e-02, 1.05023877e-02],
    [0.02565201, 0.02565201, 0.02565201, 0.02565201],  # 1e0
    [0.0544571 , 0.0544571 , 0.0544571 , 0.0544571 ],
    [0.10353708, 0.10353708, 0.10353708, 0.10353708],
    [0.1806827 , 0.1806827 , 0.1806827 , 0.1806827 ],
    [0.29156303, 0.29156303, 0.29156303, 0.29156304],
    [0.42615111, 0.42615111, 0.42615109, 0.4261511 ],  # 1e1
    [0.54882006, 0.54882002, 0.54881994, 0.54881995],
    [0.63236782, 0.63236775, 0.63236773, 0.63236788],
    [0.68116425, 0.68116398, 0.68116403, 0.6811642 ],
    [0.70861283, 0.70861286, 0.70861287, 0.70861328],
    [0.72409239, 0.72409228, 0.72409262, 0.72409303],  # 1e2
    [0.73293795, 0.73293774, 0.7329378 , 0.73293819],
    [0.73808289, 0.73808244, 0.73808242, 0.73808287],
    [0.74113273, 0.74113219, 0.74113215, 0.74113273],
    [0.74297225, 0.74297191, 0.74297189, 0.74297229],
    [0.74409727, 0.74409673, 0.74409667, 0.74409734],  # 1e3
    ]))
list_nn_link_full.append(array([
    [0.00344214, 0.00344214, 0.00344214, 0.00344296, 0.00344214,
     0.00344296, 0.00344296, 0.00344214, 0.00344296, 0.00344214,
     0.00344214, 0.00344214],  # 1e-1
    [0.00711753, 0.00711753, 0.00711753, 0.0071248 , 0.00711753,
     0.0071248 , 0.0071248 , 0.00711753, 0.0071248 , 0.00711753,
     0.00711753, 0.00711753],
    [0.01356148, 0.01356148, 0.01356148, 0.01361202, 0.01356148,
     0.01361202, 0.01361202, 0.01356148, 0.01361202, 0.01356148,
     0.01356148, 0.01356148],
    [0.02348697, 0.02348697, 0.02348697, 0.02375031, 0.02348697,
     0.02375031, 0.02375031, 0.02348697, 0.02375031, 0.02348697,
     0.02348697, 0.02348697],
    [0.03689311, 0.03689311, 0.03689311, 0.03791124, 0.03689311,
     0.03791124, 0.03791124, 0.03689311, 0.03791124, 0.03689311,
     0.03689311, 0.03689311],
    [0.05313309, 0.05313309, 0.05313309, 0.05613379, 0.05313309,
     0.05613379, 0.05613379, 0.05313309, 0.05613379, 0.05313309,
     0.05313309, 0.05313309],  # 1e0
    [0.07160017, 0.07160017, 0.07160017, 0.07870578, 0.07160017,
     0.07870578, 0.07870578, 0.07160017, 0.07870578, 0.07160017,
     0.07160017, 0.07160017],
    [0.09277028, 0.09277028, 0.09277028, 0.10719376, 0.09277028,
     0.10719376, 0.10719376, 0.09277028, 0.10719376, 0.09277028,
     0.09277028, 0.09277028],
    [0.11981592, 0.11981593, 0.11981593, 0.14661509, 0.11981593,
     0.14661509, 0.14661509, 0.11981593, 0.14661509, 0.11981593,
     0.11981593, 0.11981593],
    [0.16094871, 0.16094871, 0.16094871, 0.20841768, 0.16094871,
     0.20841769, 0.20841769, 0.16094871, 0.20841769, 0.16094871,
     0.16094871, 0.16094871],
    [0.22781775, 0.22781776, 0.22781775, 0.30598668, 0.22781776,
     0.30598669, 0.30598666, 0.22781776, 0.30598668, 0.22781775,
     0.22781776, 0.22781775],  # 1e1
    [0.31956751, 0.3195675 , 0.31956751, 0.43153567, 0.3195675 ,
     0.43153577, 0.43153556, 0.31956752, 0.43153565, 0.31956747,
     0.31956752, 0.31956747],
    [0.41517076, 0.41517071, 0.41517076, 0.5541107 , 0.41517071,
     0.55411055, 0.5541107 , 0.41517079, 0.55411079, 0.41517077,
     0.41517079, 0.41517078],
    [0.49741274, 0.49741265, 0.49741274, 0.65452226, 0.49741265,
     0.65452234, 0.65452239, 0.49741254, 0.65452224, 0.49741256,
     0.49741254, 0.49741256],
    [0.5619306 , 0.56193063, 0.56193061, 0.73019449, 0.56193064,
     0.73019124, 0.73019326, 0.56193153, 0.73019468, 0.56193123,
     0.56193152, 0.56193122],
    [0.60993165, 0.60993147, 0.60993168, 0.78428032, 0.60993148,
     0.78427203, 0.78427735, 0.60993265, 0.78428038, 0.60993193,
     0.6099326 , 0.60993191],  # 1e2
    [0.64415669, 0.64415658, 0.64415671, 0.8213824 , 0.64415659,
     0.82137838, 0.82138052, 0.64415743, 0.82138238, 0.64415689,
     0.64415741, 0.64415687],
    [0.66770667, 0.66770677, 0.66770666, 0.84608461, 0.66770677,
     0.84608479, 0.84608477, 0.6677067 , 0.84608492, 0.66770639,
     0.6677067 , 0.66770639],
    [0.68347102, 0.6834712 , 0.68347101, 0.86220642, 0.68347121,
     0.86220646, 0.86220643, 0.68347125, 0.86220681, 0.6834708 ,
     0.68347125, 0.6834708 ],
    [0.69381606, 0.69381629, 0.69381606, 0.87259541, 0.69381629,
     0.87259485, 0.87259497, 0.69381656, 0.87259534, 0.69381627,
     0.69381655, 0.69381627],
    [0.7005119 , 0.70051235, 0.7005119 , 0.87923697, 0.70051236,
     0.87923608, 0.87923627, 0.70051291, 0.879237  , 0.70051227,
     0.7005129 , 0.70051227],  # 1e3
    ])
    )

# g_gauge, list_ncut, list_trunc_dims = 1, [100]*num_site, [5]*num_site
# lambda_JJ/m_mass = 10
list_cos_phi_plaquette_full.append(array([
    [9.11328283e-05, 9.11328281e-05, 9.11328286e-05, 9.11328284e-05],  # 1e-1
    [5.29798269e-04, 5.29798302e-04, 5.29798302e-04, 5.29798269e-04],
    [2.83872434e-03, 2.83872423e-03, 2.83872422e-03, 2.83872434e-03],
    [1.30263946e-02, 1.30263947e-02, 1.30263946e-02, 1.30263949e-02],
    [4.63738495e-02, 4.63738497e-02, 4.63738498e-02, 4.63738495e-02],
    [0.11909675, 0.11909675, 0.11909675, 0.11909675],  # 1e0
    [0.22386075, 0.22386075, 0.22386075, 0.22386075],
    [0.33552356, 0.33552356, 0.33552356, 0.33552356],
    [0.43833394, 0.43833394, 0.43833394, 0.43833394],
    [0.52868255, 0.52868255, 0.52868255, 0.52868255],
    [0.60645267, 0.60645267, 0.60645267, 0.60645267],  # 1e1
    [0.67107532, 0.67107532, 0.67107532, 0.67107532],
    [0.72145963, 0.72145963, 0.72145963, 0.72145963],
    [0.75741728, 0.75741728, 0.75741728, 0.75741728],
    [0.78081189, 0.78081189, 0.78081189, 0.78081189],
    [0.79498649, 0.79498649, 0.79498649, 0.79498649],  # 1e2
    [0.80325703, 0.80325703, 0.80325703, 0.80325703],
    [0.80804284, 0.80804284, 0.80804284, 0.80804284],
    [0.81083849, 0.81083849, 0.81083849, 0.81083849],
    [0.81249846, 0.81249846, 0.81249846, 0.81249846],
    [0.81350046, 0.81350046, 0.81350046, 0.81350046],  # 1e3
    ])
    )
list_nn_link_full.append(array([
    [0.00474613, 0.00474613, 0.00474613, 0.00474615, 0.00474613,
     0.00474615, 0.00474615, 0.00474613, 0.00474615, 0.00474613,
     0.00474613, 0.00474613],  # 1e-1
    [0.01144725, 0.01144725, 0.01144725, 0.01144755, 0.01144725,
     0.01144755, 0.01144755, 0.01144725, 0.01144755, 0.01144725,
     0.01144725, 0.01144725],
    [0.02663257, 0.02663257, 0.02663257, 0.02663627, 0.02663257,
     0.02663627, 0.02663627, 0.02663257, 0.02663627, 0.02663257,
     0.02663257, 0.02663257],
    [0.05798677, 0.05798677, 0.05798677, 0.05802372, 0.05798677,
     0.05802372, 0.05802372, 0.05798677, 0.05802372, 0.05798677,
     0.05798677, 0.05798677],
    [0.11341617, 0.11341617, 0.11341617, 0.11367594, 0.11341617,
     0.11367594, 0.11367594, 0.11341617, 0.11367594, 0.11341617,
     0.11341617, 0.11341617],
    [0.19299848, 0.19299848, 0.19299848, 0.19416252, 0.19299848,
     0.19416252, 0.19416252, 0.19299848, 0.19416252, 0.19299848,
     0.19299848, 0.19299848],  # 1e0
    [0.28626331, 0.28626331, 0.28626331, 0.28970143, 0.28626331,
     0.28970143, 0.28970143, 0.28626331, 0.28970143, 0.28626331,
     0.28626331, 0.28626331],
    [0.38244336, 0.38244336, 0.38244336, 0.39016805, 0.38244336,
     0.39016805, 0.39016805, 0.38244336, 0.39016805, 0.38244336,
     0.38244336, 0.38244336],
    [0.47817698, 0.47817698, 0.47817698, 0.49326173, 0.47817698,
     0.49326173, 0.49326173, 0.47817698, 0.49326173, 0.47817698,
     0.47817698, 0.47817698],
    [0.57508803, 0.57508803, 0.57508803, 0.60219811, 0.57508803,
     0.60219811, 0.60219811, 0.57508803, 0.60219811, 0.57508803,
     0.57508803, 0.57508803],
    [0.67572836, 0.67572836, 0.67572836, 0.72073244, 0.67572836,
     0.72073244, 0.72073244, 0.67572836, 0.72073244, 0.67572836,
     0.67572836, 0.67572836],  # 1e1
    [0.78153259, 0.78153259, 0.78153259, 0.84973059, 0.78153259,
     0.84973059, 0.84973059, 0.78153259, 0.84973059, 0.78153259,
     0.78153259, 0.78153259],
    [0.8909922 , 0.8909922 , 0.8909922 , 0.98445545, 0.8909922 ,
     0.98445545, 0.98445545, 0.8909922 , 0.98445545, 0.8909922 ,
     0.8909922 , 0.8909922 ],
    [0.99812795, 0.99812795, 0.99812795, 1.11411128, 0.99812795,
     1.11411128, 1.11411128, 0.99812795, 1.11411128, 0.99812795,
     0.99812795, 0.99812795],
    [1.09425695, 1.09425695, 1.09425695, 1.22657831, 1.09425695,
     1.22657831, 1.22657831, 1.09425695, 1.22657831, 1.09425695,
     1.09425695, 1.09425695],
    [1.17275146, 1.17275146, 1.17275146, 1.31492491, 1.17275146,
     1.31492491, 1.31492491, 1.17275146, 1.31492491, 1.17275146,
     1.17275146, 1.17275146],  # 1e2
    [1.23186547, 1.23186547, 1.23186547, 1.37915749, 1.23186547,
     1.37915749, 1.37915749, 1.23186547, 1.37915749, 1.23186547,
     1.23186547, 1.23186547],
    [1.27378296, 1.27378296, 1.27378296, 1.42344653, 1.27378296,
     1.42344653, 1.42344653, 1.27378296, 1.42344653, 1.27378296,
     1.27378296, 1.27378296],
    [1.30230235, 1.30230235, 1.30230235, 1.45296782, 1.30230235,
     1.45296782, 1.45296782, 1.30230235, 1.45296782, 1.30230235,
     1.30230235, 1.30230235],
    [1.32118464, 1.32118464, 1.32118464, 1.47223727, 1.32118464,
     1.47223727, 1.47223727, 1.32118464, 1.47223727, 1.32118464,
     1.32118464, 1.32118464],
    [1.33346858, 1.33346858, 1.33346858, 1.48465406, 1.33346857,
     1.48465406, 1.48465406, 1.33346857, 1.48465406, 1.33346858,
     1.33346857, 1.33346858],  # 1e3
    ])
    )

# g_gauge, list_ncut, list_trunc_dims = 1, [100]*num_site, [5]*num_site
# lambda_JJ/m_mass = 10
list_cos_phi_plaquette_full.append(array([
    [9.60268348e-05, 9.60268348e-05, 9.60268348e-05, 9.60268348e-05],
    [5.73835575e-04, 5.73835587e-04, 5.73835587e-04, 5.73835574e-04],
    [3.18902383e-03, 3.18902382e-03, 3.18902382e-03, 3.18902383e-03],
    [1.51934812e-02, 1.51934811e-02, 1.51934811e-02, 1.51934812e-02],
    [5.50249018e-02, 5.50249018e-02, 5.50249018e-02, 5.50249018e-02],
    [1.38277757e-01, 1.38277757e-01, 1.38277757e-01, 1.38277756e-01],
    [2.48065070e-01, 2.48065070e-01, 2.48065070e-01, 2.48065070e-01],
    [3.56696045e-01, 3.56696045e-01, 3.56696045e-01, 3.56696045e-01],
    [4.54890556e-01, 4.54890556e-01, 4.54890556e-01, 4.54890556e-01],
    [5.42686463e-01, 5.42686463e-01, 5.42686463e-01, 5.42686463e-01],
    [6.20216060e-01, 6.20216060e-01, 6.20216060e-01, 6.20216060e-01],
    [6.87338266e-01, 6.87338266e-01, 6.87338266e-01, 6.87338266e-01],
    [7.44392077e-01, 7.44392077e-01, 7.44392077e-01, 7.44392077e-01],
    [7.92182472e-01, 7.92182472e-01, 7.92182472e-01, 7.92182472e-01],
    [8.31728383e-01, 8.31728383e-01, 8.31728383e-01, 8.31728383e-01],
    [8.64027546e-01, 8.64027546e-01, 8.64027546e-01, 8.64027546e-01],
    [8.89853204e-01, 8.89853204e-01, 8.89853204e-01, 8.89853204e-01],
    [9.09670451e-01, 9.09670451e-01, 9.09670451e-01, 9.09670451e-01],
    [9.23866355e-01, 9.23866355e-01, 9.23866355e-01, 9.23866355e-01],
    [9.33190664e-01, 9.33190663e-01, 9.33190663e-01, 9.33190663e-01],
    [9.38848612e-01, 9.38848611e-01, 9.38848611e-01, 9.38848612e-01]])
    )
list_nn_link_full.append(array([
    [0.00491594, 0.00491594, 0.00491594, 0.00491594, 0.00491594,
     0.00491594, 0.00491594, 0.00491594, 0.00491594, 0.00491594,
     0.00491594, 0.00491594],
    [0.01208639, 0.01208639, 0.01208639, 0.01208639, 0.01208639,
     0.01208639, 0.01208639, 0.01208639, 0.01208639, 0.01208639,
     0.01208639, 0.01208639],
    [0.02889556, 0.02889556, 0.02889556, 0.0288956 , 0.02889556,
     0.0288956 , 0.0288956 , 0.02889556, 0.0288956 , 0.02889556,
     0.02889556, 0.02889556],
    [0.06508452, 0.06508452, 0.06508452, 0.06508504, 0.06508452,
     0.06508504, 0.06508504, 0.06508452, 0.06508504, 0.06508452,
     0.06508452, 0.06508452],
    [0.131577  , 0.131577  , 0.131577  , 0.13158118, 0.131577  ,
     0.13158118, 0.13158118, 0.131577  , 0.13158118, 0.131577  ,
     0.131577  , 0.131577  ],
    [0.22907669, 0.22907669, 0.22907669, 0.22909789, 0.22907669,
     0.22909789, 0.22909789, 0.22907669, 0.22909789, 0.22907669,
     0.22907669, 0.22907669],
    [0.34495703, 0.34495703, 0.34495703, 0.34503072, 0.34495703,
     0.34503072, 0.34503072, 0.34495703, 0.34503072, 0.34495703,
     0.34495703, 0.34495703],
    [0.47071774, 0.47071774, 0.47071774, 0.47093295, 0.47071774,
     0.47093295, 0.47093295, 0.47071774, 0.47093295, 0.47071774,
     0.47071774, 0.47071774],
    [0.61056878, 0.61056878, 0.61056878, 0.61117486, 0.61056878,
     0.61117486, 0.61117486, 0.61056878, 0.61117486, 0.61056878,
     0.61056878, 0.61056878],
    [0.77274447, 0.77274447, 0.77274447, 0.77441231, 0.77274447,
     0.77441231, 0.77441231, 0.77274447, 0.77441231, 0.77274447,
     0.77274447, 0.77274447],
    [0.96230451, 0.96230451, 0.96230451, 0.96666333, 0.96230451,
     0.96666333, 0.96666333, 0.96230451, 0.96666333, 0.96230451,
     0.96230451, 0.96230451],
    [1.18042995, 1.18042995, 1.18042995, 1.19100765, 1.18042995,
     1.19100765, 1.19100765, 1.18042995, 1.19100765, 1.18042995,
     1.18042995, 1.18042995],
    [1.42581661, 1.42581661, 1.42581661, 1.44935228, 1.42581661,
     1.44935228, 1.44935228, 1.42581661, 1.44935228, 1.42581661,
     1.42581661, 1.42581661],
    [1.69623477, 1.69623477, 1.69623477, 1.74400144, 1.69623477,
     1.74400144, 1.74400144, 1.69623477, 1.74400144, 1.69623477,
     1.69623477, 1.69623477],
    [1.990372  , 1.990372  , 1.990372  , 2.07880473, 1.990372  ,
     2.07880473, 2.07880473, 1.990372  , 2.07880473, 1.990372  ,
     1.990372  , 1.990372  ],
    [2.30935543, 2.30935543, 2.30935543, 2.45902778, 2.30935543,
     2.45902778, 2.45902778, 2.30935543, 2.45902778, 2.30935543,
     2.30935543, 2.30935543],
    [2.65597521, 2.65597521, 2.65597521, 2.88755717, 2.65597521,
     2.88755717, 2.88755717, 2.65597521, 2.88755717, 2.65597521,
     2.65597521, 2.65597521],
    [3.02888255, 3.02888255, 3.02888255, 3.35521022, 3.02888255,
     3.35521022, 3.35521022, 3.02888255, 3.35521022, 3.02888255,
     3.02888255, 3.02888255],
    [3.4129803 , 3.41298029, 3.4129803 , 3.83073566, 3.41298029,
     3.83073566, 3.83073565, 3.41298029, 3.83073565, 3.41298029,
     3.41298029, 3.41298029],
    [3.77754147, 3.77754146, 3.77754147, 4.26709366, 3.77754146,
     4.26709367, 4.26709366, 3.77754146, 4.26709365, 3.77754146,
     3.77754146, 3.77754146],
    [4.09066252, 4.09066252, 4.09066252, 4.6261903 , 4.09066252,
     4.62619031, 4.62619031, 4.09066252, 4.62619031, 4.09066252,
     4.09066252, 4.09066252]])
    )

# g_gauge, list_ncut, list_trunc_dims = 1, [100]*num_site, [5]*num_site
# lambda_JJ/m_mass = infty
list_cos_phi_plaquette_0 = array([[
    9.65920398e-05, 5.79023269e-04, 3.23138565e-03, 1.54622679e-02,
    5.61040938e-02,
    0.14059248, 0.25080731, 0.35899735, 0.45679225, 0.54449738,
    0.62205586, 0.68919465, 0.74622698, 0.79396707, 0.83347835,
    0.86589295, 0.89230392, 0.91370812, 0.93098183, 0.94487602,
    0.95602272
    ]])
list_nn_link_0 = array([[
    0.00493537, 0.0121607 , 0.02916484, 0.06595473, 0.13388039,
    0.23384781,  0.35332146,  0.48513381,  0.63675986,  0.82160385,
    1.05255997,  1.34272825,  1.70768155,  2.16687789,  2.74478169,
    3.47217445,  4.38779603,  5.54040875,  6.99139472,  8.81802502,
    11.11757487
    ]])

outer_idx = array([0, 1, 2, 4, 7, 9, 10, 11])
inner_idx = array([3, 5, 6, 8])
list_cos_phi_plaquette_avg = np.average(list_cos_phi_plaquette_full, axis=2)
list_nn_link_avg_outer = np.average(
    array(list_nn_link_full)[:, :, outer_idx], axis=2)
list_nn_link_avg_inner = np.average(
    array(list_nn_link_full)[:, :, inner_idx], axis=2)

list_cos_phi_plaquette_avg = np.concatenate((
    list_cos_phi_plaquette_avg, list_cos_phi_plaquette_0))
list_nn_link_avg_outer = np.concatenate((
    list_nn_link_avg_outer, list_nn_link_0))
list_nn_link_avg_inner = np.concatenate((
    list_nn_link_avg_inner, list_nn_link_0))

# %%
list_colors = ['0.8', '0.5', '0.3', '0']
fig, ax = plt.subplots(2, sharex=True)
for m_idx in range(len(list_lambda_over_m)-1):
    ax[0].plot(
        list_lambda, list_cos_phi_plaquette_avg[m_idx],
        c=list_colors[m_idx],
        label=r'$\lambda/m=$'+str(list_lambda_over_m[m_idx]),
        )
    ax[1].plot(
        list_lambda, list_nn_link_avg_outer[m_idx],
        ls='-', c=list_colors[m_idx],
        label=r'$\lambda/m=$'+str(list_lambda_over_m[m_idx])+', outer',
        )
    ax[1].plot(
        list_lambda, list_nn_link_avg_inner[m_idx],
        ls='--', c=list_colors[m_idx],
        label=r'$\lambda/m=$'+str(list_lambda_over_m[m_idx])+', inner',
        )
m_idx = len(list_lambda_over_m)-1
ax[0].plot(
    list_lambda, list_cos_phi_plaquette_avg[m_idx],
    ls=':', c=list_colors[m_idx],
    label=r'$\lambda/m=$'+str(list_lambda_over_m[m_idx]),
    )
ax[1].plot(
    list_lambda, list_nn_link_avg_outer[m_idx],
    ls=':', c=list_colors[m_idx],
    label=r'$\lambda/m=$'+str(list_lambda_over_m[m_idx]),
    )
ax[0].set_xscale('log')
ax[0].set_xlim(0.1, 1000)
ax[0].set_ylim(-0.01, 1.01)
ax[0].set_ylabel(r'$\langle \hat{\blacksquare} \rangle$')
ax[0].legend()
# ax[0].grid()
ax[1].set_ylim(-0.01, 13)
ax[1].set_yticks([0, 3, 6, 9, 12])
# ax[1].grid()
# ax[1].legend()
ax[1].set_ylabel(r'$\langle n^2 \rangle$')
ax[1].set_xlabel(r'$\lambda$')
fig.suptitle('2x2-plaquette lattice, g=1')


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













