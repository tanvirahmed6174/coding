import numpy as np
import scqubits as scq
import qutip as q

def fast_flux_control(tlist, circ_params, args, goal="dynamics", c0=1, c1=0):
    ECJ = circ_params["ECJ"]
    EJ  = circ_params["EJ"]
    EL  = circ_params["EL"]

    fluxonium_yaml = f"""
                  branches:
                  - [JJ, 1,2, EJ={EJ}, ECJ={ECJ}]
                  - [L,  1,2, EL={EL}]
                    """
    fluxonium = scq.Circuit(fluxonium_yaml, from_file=False, use_dynamic_flux_grouping=True)
    fluxonium.set_discretized_phi_range(var_indices=(1,), phi_range=(-4*np.pi, 4*np.pi))
    fluxonium.cutoff_ext_1 = 110
    fluxonium.Φ1 = 0.5
    fluxonium.configure(system_hierarchy=[[1]], subsystem_trunc_dims=[6])

    def ϕ1_control(t, args):
        φm = args["φm"]
        φp = args["φp"]
        t_ramp_m = args["t_ramp_m"]
        t_wait = args["t_wait"]
        t_ramp_p = args["t_ramp_p"]
        t_ramp_up_start = 2*t_ramp_m + t_wait
        if (t < t_ramp_m):
            ϕ1_ext = 0.5 - t * φm / t_ramp_m
        elif t >= t_ramp_m and t < 2*t_ramp_m:
            ϕ1_ext = (0.5 - φm) + (t-t_ramp_m)*φm/t_ramp_m
        elif t >= 2*t_ramp_m and t < t_ramp_up_start:
            ϕ1_ext = 0.5
        elif t >= t_ramp_up_start and t < t_ramp_up_start+t_ramp_p:
            ϕ1_ext = 0.5 + (t-t_ramp_up_start)*φp/t_ramp_p
        elif t >= t_ramp_up_start+t_ramp_p and t < t_ramp_up_start+2*t_ramp_p:
            ϕ1_ext = (0.5 + φp) - (t-t_ramp_up_start-t_ramp_p)*φp/t_ramp_p
        else:
            ϕ1_ext = 0.5
        return ϕ1_ext

    t_ramp_m = args["tp"]/2
    t_ramp_p = args["tp"]/2
    t_wait   = args["tz"]
    φm = args["φ_amp"]
    φp = args["φ_amp"]
    args1 = {"φm": φm, "φp": φp, "t_ramp_m": t_ramp_m, "t_wait": t_wait, "t_ramp_p": t_ramp_p}

    # Extract the lowest two eigenstates of the Hamiltonian
    eigvals, eigvecs = fluxonium.eigensys(evals_count=2)
    zero_05 = q.Qobj(eigvecs[:,0])
    one_05  = q.Qobj(eigvecs[:,1])

    psi0 = c0 * zero_05 + c1 * one_05
    rho0 = psi0 * psi0.dag()

    plus    = (zero_05 + one_05)/np.sqrt(2)
    minus   = (zero_05 - one_05)/np.sqrt(2)
    plus_y  = (zero_05 + 1j * one_05)/np.sqrt(2)
    minus_y = (zero_05 - 1j * one_05)/np.sqrt(2)

    x_op    = plus * plus.dag() - minus * minus.dag()
    y_op    = plus_y * plus_y.dag() - minus_y * minus_y.dag()
    z_op    = zero_05 * zero_05.dag() - one_05 * one_05.dag()

    H_mesolve, *H_sym_ref = fluxonium.hamiltonian_for_qutip_dynamics(
        free_var_func_dict={"Φ1": ϕ1_control},
        prefactor=2*np.pi
    )

    result = q.mesolve(H_mesolve, rho0, tlist, e_ops=[x_op, y_op, z_op], args=args1)
    return result.expect[0], result.expect[1], result.expect[2]

def worker(φ_amp, tz, tp, t_ramp_m, t_ramp_p, circ_params, c0, c1):
    args = {"φ_amp": φ_amp, "tz": tz, "tp": tp}
    t_wait = tz
    tlist = np.linspace(0, 2*t_ramp_m + t_wait + 2*t_ramp_p, 1000)
    exp_x, exp_y, exp_z = fast_flux_control(tlist, circ_params, args, goal="dynamics", c0=c0, c1=c1)
    return (φ_amp, tz, exp_x[-1], exp_y[-1], exp_z[-1])