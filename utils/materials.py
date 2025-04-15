# utils/materials

# imports
from utils import energy
from physics import bilayer_modulus_from_effective

# Reference. Bulk material properties
GOLD_E_BULK = 80e9

def get_membrane_settings(memb_id):
    # membrane
    memb_mat = 'ELASTOSIL'
    memb_t0 = 20e-6
    memb_E = 1.2e6
    memb_G = memb_E / 3
    memb_nu = 0.499
    memb_J = 0.5
    # metal
    met_nu = 0.44
    met_ps = 1.0001  # 1.000001

    if memb_id == 'C9-0pT-20nmAu':
        # 1. apply pre-stretch
        pre_stretch_nominal = 1.0
        pre_stretch_measured = 1.01  # Estimated from bulge test residual stress
        # 2. metal deposition
        met_mat = 'MPTMS+Au'
        met_t = 20e-9
        # 3. bulge test
        comp_E = 3.0e6
        comp_sigma0 = 30e3
        # --- Lower and upper bounds
        pre_stretch_lower_bound = 1.0
        pre_stretch_upper_bound = 1.02
        comp_E_lower_bound = 2.5e6
        comp_E_upper_bound = 3.0e6
        comp_sigma0_lower_bound = 27e3
        comp_sigma0_upper_bound = 31e3
    elif memb_id == 'C7-20pT-20nmAu':
        # 1. apply pre-stretch
        pre_stretch_nominal = 1.2
        pre_stretch_measured = 1.2          # Estimated
        # 2. metal deposition
        met_mat = 'MPTMS+Au'
        met_t = 20e-9
        # 3. bulge test
        comp_E = 3.25e6                     # Estimated
        comp_sigma0 = 300e3
        # --- Lower and upper bounds
        pre_stretch_lower_bound = 1.2
        pre_stretch_upper_bound = 1.26
        comp_E_lower_bound = 2.5e6
        comp_E_upper_bound = 3.7e6
        comp_sigma0_lower_bound = 300e3
        comp_sigma0_upper_bound = 310e3
    elif memb_id == 'C15-15pT-25nmAu':
        # 1. apply pre-stretch
        pre_stretch_nominal = 1.15
        pre_stretch_measured = 1.146
        # 2. metal deposition
        met_mat = 'MPTMS+Au'
        met_t = 25e-9
        # 3. bulge test
        comp_E = 6.0e6
        comp_sigma0 = 300e3
        # --- Lower and upper bounds
        pre_stretch_lower_bound = 1.12
        pre_stretch_upper_bound = 1.15
        comp_E_lower_bound = 5.75e6
        comp_E_upper_bound = 6.5e6
        comp_sigma0_lower_bound = 300e3
        comp_sigma0_upper_bound = 305e3
    elif memb_id == 'C17-20pT-25nmAu':
        # 1. apply pre-stretch
        pre_stretch_nominal = 1.20
        pre_stretch_measured = 1.25
        # 2. metal deposition
        met_mat = 'MPTMS+Au'
        met_t = 25e-9
        # 3. bulge test
        comp_E = 4.0e6
        comp_sigma0 = 520e3
        # --- Lower and upper bounds
        pre_stretch_lower_bound = 1.2
        pre_stretch_upper_bound = 1.25
        comp_E_lower_bound = 3.25e6
        comp_E_upper_bound = 4.5e6
        comp_sigma0_lower_bound = 510e3
        comp_sigma0_upper_bound = 540e3
    elif memb_id == 'C19-30pT-20+10nmAu':
        # 1. apply pre-stretch
        pre_stretch_nominal = 1.3
        pre_stretch_measured = 1.131
        # 2. metal deposition
        met_mat = 'MPTMS+Au'
        met_t = 30e-9
        # 3. bulge test
        comp_E = 12.5e6
        comp_sigma0 = 330e3
        # --- Lower and upper bounds
        pre_stretch_lower_bound = 1.10
        pre_stretch_upper_bound = 1.15
        comp_E_lower_bound = 7.0e6
        comp_E_upper_bound = 13.0e6
        comp_sigma0_lower_bound = 300e3
        comp_sigma0_upper_bound = 340e3
    else:
        raise ValueError('Unknown membrane ID.')

    dict_memb_settings = {
        'memb_id': memb_id,
        'memb_mat': memb_mat,
        'memb_t0': memb_t0,
        'memb_E': memb_E,
        'memb_G': memb_G,
        'memb_nu': memb_nu,
        'memb_J': memb_J,
        'memb_pre_stretch_nominal': pre_stretch_nominal,
        'memb_pre_stretch_measured': pre_stretch_measured,
        'met_mat': met_mat,
        'met_t': met_t,
        'met_nu': met_nu,
        'met_ps': met_ps,
        'comp_E': comp_E,
        'comp_sigma0': comp_sigma0,
    }

    dict_memb_settings = compute_derived_memb_settings(dict_memb_settings)

    return dict_memb_settings

def compute_derived_memb_settings(dict_memb_settings):
    # 2. apply pre-stretch
    memb_t_post_measured_pre_stretch = energy.calculate_stretched_thickness(
        original_thickness=dict_memb_settings['memb_t0'],
        stretch_factor=dict_memb_settings['memb_pre_stretch_measured'],
    )
    gent_sigma0_from_measured_pre_stretch = energy.gent_stress_from_pre_stretch(
        pre_stretch=dict_memb_settings['memb_pre_stretch_measured'],
        mu=dict_memb_settings['memb_G'],
        Jm=dict_memb_settings['memb_J'],
    )
    neo_hookean_sigma0_from_measured_pre_stretch = energy.neo_hookean_stress_from_pre_stretch(
        pre_stretch=dict_memb_settings['memb_pre_stretch_measured'],
        shear_modulus=dict_memb_settings['memb_G'],
    )
    # 3. metal deposition
    theoretical_comp_E_from_material_stack = bilayer_modulus_from_effective.effective_youngs_modulus_bilayer(
        t_silicone=memb_t_post_measured_pre_stretch,
        E_silicone=dict_memb_settings['memb_E'],
        t_gold=dict_memb_settings['met_t'],
        E_gold=GOLD_E_BULK,
    )
    # 4. bulge test
    met_E_from_measured_comp_E = bilayer_modulus_from_effective.estimate_gold_modulus_from_effective(
        E_eff_measured=dict_memb_settings['comp_E'],
        t_elastomer=memb_t_post_measured_pre_stretch,
        E_elastomer=dict_memb_settings['memb_E'],
        t_gold=dict_memb_settings['met_t'],
    )
    gent_pre_stretch_from_measured_comp_sigma0 = energy.compute_gent_stretch_from_stress(
        sigma=dict_memb_settings['comp_sigma0'],
        mu=dict_memb_settings['memb_G'],
        Jm=dict_memb_settings['memb_J'],
        initial_guess=1.15,
    )
    neo_hookean_pre_stretch_from_measured_comp_sigma0 = energy.compute_neo_hookean_stretch_from_stress(
        sigma=dict_memb_settings['comp_sigma0'],
        mu=dict_memb_settings['memb_G'],
        initial_guess=1.15,
    )
    dict_derived = {
        'memb_t_post_measured_pre_stretch': memb_t_post_measured_pre_stretch,
        'gent_sigma0_from_measured_pre_stretch': gent_sigma0_from_measured_pre_stretch,
        'neo_hookean_sigma0_from_measured_pre_stretch': neo_hookean_sigma0_from_measured_pre_stretch,
        'theoretical_comp_E_from_material_stack': theoretical_comp_E_from_material_stack,
        'met_E_from_measured_comp_E': met_E_from_measured_comp_E,
        'gent_pre_stretch_from_measured_comp_sigma0': gent_pre_stretch_from_measured_comp_sigma0,
        'neo_hookean_pre_stretch_from_measured_comp_sigma0': neo_hookean_pre_stretch_from_measured_comp_sigma0,
    }
    dict_memb_settings.update(dict_derived)
    return dict_memb_settings



if __name__ == '__main__':
    # 0. Bulk material properties
    gold_E_bulk = 80e9

    # 1. start with elastomer substrate
    memb_id = 1
    memb_mat = 'ELASTOSIL'
    memb_t0 = 20e-6
    memb_E = 1.2e6
    memb_G = memb_E / 3
    memb_nu = 0.499
    memb_j = 0.5
    # 2. apply pre-stretch
    pre_stretch_nominal = 1.3
    pre_stretch_measured = 1.131
    # 3. metal deposition
    met_mat = 'MPTMS+Au'
    met_t = 30e-9
    met_E = 6.0e9
    met_nu = 0.44
    # 4. bulge test
    comp_E = 1e6
    comp_sigma0 = 250e3


    # ---- DERIVED METRICS
    # 2. apply pre-stretch
    memb_t_post_measured_pre_stretch = energy.calculate_stretched_thickness(memb_t0, pre_stretch_measured)
    gent_sigma0_from_measured_pre_stretch = energy.gent_stress_from_pre_stretch(pre_stretch_measured, memb_G, memb_j)
    neo_hookean_sigma0_from_measured_pre_stretch = energy.neo_hookean_stress_from_pre_stretch(pre_stretch_measured, memb_G)

    # 3. metal deposition
    theoretical_comp_E_from_material_stack = bilayer_modulus_from_effective.effective_youngs_modulus_bilayer(
        t_silicone=memb_t_post_measured_pre_stretch,
        E_silicone=memb_E,
        t_gold=met_t,
        E_gold=gold_E_bulk,
    )

    # 4. bulge test
    met_E_from_measured_bulge = bilayer_modulus_from_effective.estimate_gold_modulus_from_effective(
        E_eff_measured=comp_E,
        t_elastomer=memb_t_post_measured_pre_stretch,
        E_elastomer=memb_E,
        t_gold=met_t,
    )
    gent_pre_stretch_from_measured_bulge = energy.compute_gent_stretch_from_stress(
        sigma=comp_sigma0,
        mu=memb_G,
        Jm=memb_j,
        initial_guess=1.15,
    )
    neo_hookean_pre_stretch_from_measured_bulge = energy.compute_neo_hookean_stretch_from_stress(
        sigma=comp_sigma0,
        mu=memb_G,
        initial_guess=1.15,
    )


