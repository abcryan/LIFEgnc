import numpy as np
from scipy.integrate import solve_ivp

from life_control.utils.other import Omega_omega
from life_control.utils.coordinate_trafos import quat_to_CTM
from life_control.plant_model.thrusters import (
    THRUSTER_POSITIONS,
    THRUSTER_NORMALS,
    B_F,
    B_TAU,
    N_THRUSTERS,
    clamp_thrust,
    force_torque_body,
)

from life_control.utils.physics import J_ring_unit, J_cylinder_unit

import life_control.utils.constants as const


class Plant:
    """
    Truth model dynamics for an N-spacecraft formation.

      - Spacecraft 0 = LEADER:    absolute (r, v) in SSB-centred ICRF.
      - Spacecraft i > 0 = FOLLOWER: relative (δr, δv) w.r.t. leader in ICRF,
        plus ABSOLUTE attitude.  Relative form avoids the ~mm-level numerical
        floor incurred by differencing two ~1.5e8 km SSB-relative vectors.

    Per-spacecraft physical model:
        outer ring   (variable mass — fuel)
        inner cylinder (constant mass)
      Inertia tensor is computed as J(m) = J_cyl + J_ring(m_ring(m)) and
      tracks m through the burn.  J̇ comes solely from the ring term.

    Inter-spacecraft gravity:
      Each spacecraft attracts every other via Newtonian gravity.
      Pairwise vectors are computed from δr (relative state) so the
      mm-floor benefit of the relative-state design is preserved.

    Environment snapshot (NEW):
      Ephemeris body positions are fetched ONCE per RHS call via
      `_body_snapshot` and threaded through every consumer (N-body gravity
      for the leader and every follower, the gravity-gradient torque, and
      the SRP Sun direction).  Previously `body_position` was queried once
      per body *per spacecraft*; now it is once per body *per step*.  The
      two new attitude torques therefore add no extra ephemeris lookups.

    PhysicsFlags:
      A frozen dataclass of bool flags is held in `self.flags`. Each
      acc_*/attitude_tau_* helper checks its own flag at the top and
      short-circuits to np.zeros(3) when disabled. The architecture
      (snapshot construction, dispatch, signatures) is identical in
      every run — only the physics is gated — so timing comparisons
      between flag configurations reflect physics cost only.
    """

    def __init__(self, env, param, flags, n_sc, dim_x_sc, dim_u_sc):
        self.env       = env
        self.param     = param
        self.flags     = flags          # PhysicsFlags instance
        self.n_sc      = n_sc
        self.dim_x_sc  = dim_x_sc
        self.dim_u_sc  = dim_u_sc

    # ── Single-step propagator ──────────────────────────────────────────────

    def step(self, x, u, t, dt,
             rtol = 1e-12, atol = 1e-12, renormalize = True):

        sol = solve_ivp(self.x_dot, (0.0, dt), x,
                        method  = "DOP853",
                        args    = (t, u),
                        rtol    = rtol,
                        atol    = atol,
                        t_eval  = [dt])

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        x_next = sol.y[:, -1]

        if renormalize:
            x_next = self._renormalize_quat(x_next, self.n_sc, self.dim_x_sc)

        return x_next

    # ── ODE RHS ─────────────────────────────────────────────────────────────

    def x_dot(self, tau, x, et0, u):

        xdot = np.zeros_like(x)
        et   = et0 + tau

        # ── Fleet snapshot ──────────────────────────────────────────────
        # Build (δr_all, m_total_all) for every spacecraft ONCE per RHS call.
        # δr_all[0] = 0 by definition (leader is the reference); all pairwise
        # separations among spacecraft are then δr_j − δr_i — never
        # reconstructed from SSB-absolute positions, preserving the mm-floor.
        delta_r_all, m_total_all = self._fleet_snapshot(x)

        # ── Body (ephemeris) snapshot ───────────────────────────────────
        # Fetch every gravitating body's position ONCE per RHS call. Shared
        # by N-body gravity (leader + followers), the gravity-gradient torque,
        # and the SRP Sun direction. This is what keeps the two new torque
        # terms (essentially) free: they reuse positions already needed for
        # the translational dynamics instead of re-querying the ephemeris.
        bodies = self._body_snapshot(et)

        # Leader's absolute position — only needed for terms with absolute
        # dependence on Sun/planets (gravity_abs, srp_abs).
        r_L = x[self._slice_x(0, self.dim_x_sc)][0:3]

        # ── Leader Dynamics (i=0) ──────
        #              /\
        #             /  \
        #            |LEAD|
        #           [|====|]
        #            /|  |\
        #           o-+--+-o
        # ────────────────────────────────
        x_L         = x[self._slice_x(0, self.dim_x_sc)]
        T_cmd_L     = u[self._slice_u(0, self.dim_u_sc)]

        v_L         = x_L[3:6]
        q_L         = x_L[6:10]
        w_L         = x_L[10:13]
        m_prop_L    = x_L[13]

        # Saturate thrust to physical range [0, T_max] N per thruster.
        # The integrator may probe the RHS at small sub-steps; saturating here
        # keeps the dynamics consistent with what the actuator can deliver.
        T_L         = clamp_thrust(T_cmd_L, self.param.T_MAX)

        # Body-frame net force [N] and torque [N·m] from the allocation.
        # If the propellant tank is dry, no actuator output is possible.
        if m_prop_L <= 0.0:
            T_L     = np.zeros_like(T_L)
        F_B_L, tau_ctrl_L = force_torque_body(T_L)

        # Rotate body-frame force to inertial frame and convert to acceleration.
        # quat_to_CTM(q) returns C_I^B (inertial -> body); its transpose maps
        # body -> inertial:   v^I = C_I^B.T @ v^B.
        m_total_L   = m_total_all[0]
        F_I_L       = quat_to_CTM(q_L).T @ F_B_L                       # [N]
        # a_ctrl in [km/s^2]:  (F[N] / m[kg]) * (1e-3 km/m)
        a_ctrl_L    = (F_I_L / m_total_L) * const.KM_PER_M

        # Leader's own absolute SRP acceleration [km/s^2]. Computed ONCE here
        # and reused for (a) the translational dynamics and (b) the SRP torque.
        a_srp_I_L   = self.acc_srp_abs(r_L, m_total_L, 'L', bodies)

        # Absolute Translational Dynamics — leader uses absolute ISC sum
        r_dot_L     = v_L
        v_dot_L     = self.acc_rhs_abs(r_L, a_ctrl_L, a_srp_I_L,
                                       0, delta_r_all, m_total_all, bodies)

        # Mass Derivative (Tsiolkovsky, per-thruster sum)
        m_dot_L     = self.mass_rhs(m_prop_L, T_L, 'L')

        # Rotational Kinematics + Dynamics
        q_dot_L     = self.attitude_kin_rhs(q_L, w_L)
        w_dot_L     = self.attitude_dyn_rhs(w_L, m_prop_L, m_dot_L, m_total_L,
                                            tau_ctrl_L, 'L',
                                            q_L, r_L, a_srp_I_L, bodies)

        xdot[self._slice_x(0, self.dim_x_sc)] = np.concatenate(
            [r_dot_L, v_dot_L, q_dot_L, w_dot_L, [m_dot_L]]
        )

        # ── Follower Dynamics (i>0) ────
        #             ____
        #            |FLWR|
        #           [|o  o|]
        #            |____|
        #            /    \
        #           o      o
        # ────────────────────────────────
        for i in range(1, self.n_sc):

            x_i         = x[self._slice_x(i, self.dim_x_sc)]
            T_cmd_i     = u[self._slice_u(i, self.dim_u_sc)]

            delta_r     = x_i[0:3]
            delta_v     = x_i[3:6]
            q_i         = x_i[6:10]
            w_i         = x_i[10:13]
            m_prop_i    = x_i[13]

            T_i         = clamp_thrust(T_cmd_i, self.param.T_MAX)
            if m_prop_i <= 0.0:
                T_i     = np.zeros_like(T_i)
            F_B_i, tau_ctrl_i = force_torque_body(T_i)

            m_total_i   = m_total_all[i]
            F_I_i       = quat_to_CTM(q_i).T @ F_B_i                   # [N]
            a_ctrl_i    = (F_I_i / m_total_i) * const.KM_PER_M

            # Follower's absolute inertial position. Reconstructed at SSB scale
            # ONLY for terms that genuinely live at that scale (SRP 1/r², and
            # the line of sight to distant bodies for the gravity-gradient
            # torque) — both use directions/magnitudes that are well
            # conditioned at SSB scale, unlike the pairwise ISC separations.
            r_F         = r_L + delta_r

            # Follower's own absolute SRP acceleration — computed ONCE and
            # reused for the differential translational term AND the torque.
            a_srp_I_F   = self.acc_srp_abs(r_F, m_total_i, 'F', bodies)
            # Differential SRP (follower − leader), identical to the previous
            # acc_srp_rel result, now assembled from the two cached accels.
            a_srp_diff  = a_srp_I_F - a_srp_I_L

            # Relative Translational Dynamics — ISC enters as a_F − a_L
            r_dot_i     = delta_v
            v_dot_i     = self.acc_rhs_rel(r_L, delta_r, a_srp_diff,
                                           a_ctrl_L, a_ctrl_i,
                                           i, delta_r_all, m_total_all, bodies)

            # Mass Derivative
            m_dot_i     = self.mass_rhs(m_prop_i, T_i, 'F')

            # Rotational Kinematics + Dynamics (attitude is absolute)
            q_dot_i     = self.attitude_kin_rhs(q_i, w_i)
            w_dot_i     = self.attitude_dyn_rhs(w_i, m_prop_i, m_dot_i, m_total_i,
                                                tau_ctrl_i, 'F',
                                                q_i, r_F, a_srp_I_F, bodies)

            xdot[self._slice_x(i, self.dim_x_sc)] = np.concatenate(
                [r_dot_i, v_dot_i, q_dot_i, w_dot_i, [m_dot_i]]
            )

        return xdot

    # ── Fleet snapshot ──────────────────────────────────────────────────────

    def _fleet_snapshot(self, x):
        """
        Extract (δr, m_total) for every spacecraft in the fleet, all at once.

        Returns
        -------
        delta_r_all : (n_sc, 3) ndarray
            Position of each spacecraft RELATIVE to the leader.
            delta_r_all[0]  = [0, 0, 0]  (leader is the reference)
            delta_r_all[i]  = δr_i = state[6:9] for follower i
        m_total_all : (n_sc,) ndarray
            Total mass (cyl + ring_dry + prop) for each spacecraft.

        Why δr is the right pairwise scale:
            r_k − r_j = (r_L + δr_k) − (r_L + δr_j) = δr_k − δr_j
            so we never need to know r_L (≈ 1.5e8 km) to compute pairwise
            separations (~ 1e-4 km). Machine precision preserved.
        """
        delta_r_all = np.zeros((self.n_sc, 3))
        m_total_all = np.zeros(self.n_sc)

        # Leader: δr = 0 by definition
        x_L              = x[self._slice_x(0, self.dim_x_sc)]
        m_total_all[0]   = self._m_total(x_L[13], 'L')

        # Followers
        for i in range(1, self.n_sc):
            x_i              = x[self._slice_x(i, self.dim_x_sc)]
            delta_r_all[i]   = x_i[0:3]
            m_total_all[i]   = self._m_total(x_i[13], 'F')

        return delta_r_all, m_total_all

    # ── Body (ephemeris) snapshot ────────────────────────────────────────────

    def _body_snapshot(self, et):
        """
        Ephemeris positions of every gravitating body at epoch `et`, fetched
        ONCE per RHS call and reused by all consumers that need a body
        position:

          - N-body gravity acceleration  (acc_grav_abs / acc_grav_rel),
          - the gravity-gradient torque  (attitude_tau_grav),
          - the SRP Sun direction        (acc_srp_abs, via the 'SUN' entry).

        Returns a dict {body_name: r_body [km]} keyed exactly like env.GM.
        Empty dict when nothing needs body positions (every consumer is
        flag-gated and short-circuits before indexing this dict, so an empty
        snapshot never raises).

        Cost note: previously body_position() ran once per body *per
        spacecraft* (leader in acc_grav_abs + each follower in acc_grav_rel +
        every spacecraft's SRP Sun lookup). It now runs once per body *per
        step*, so adding the gravity-gradient torque does not add a single
        ephemeris query.
        """
        if not (self.flags.grav_nbody or self.flags.tau_grav or self.flags.srp):
            return {}
        return {body: self.env.body_position(body, et) for body in self.env.GM}

    # ── Translational dynamics ──────────────────────────────────────────────

    def acc_rhs_abs(self, r_I, a_ctrl, a_srp_I,
                    i, delta_r_all, m_total_all, bodies):
        """Total absolute acceleration on a spacecraft (used by leader).
        `i` is the spacecraft index in the fleet, needed for the ISC sum.
        `a_srp_I` is the spacecraft's own SRP acceleration, computed once by
        the caller and passed in (also reused by the SRP torque). `bodies` is
        the per-step ephemeris snapshot. Each contribution is flag-gated
        inside its own routine — passing all args unconditionally keeps the
        dispatch cost flat across runs.
        """
        a_tot = (self.acc_grav_abs(r_I, bodies)
                 + a_srp_I
                 + self.acc_grav_isc_abs(i, delta_r_all, m_total_all)
                 + a_ctrl)
        return a_tot

    def acc_rhs_rel(self, r_L, delta_r, a_srp_diff,
                    a_ctrl_L, a_ctrl_F,
                    i, delta_r_all, m_total_all, bodies):
        """
        Total DIFFERENTIAL acceleration on a follower w.r.t. the leader.
        Each per-body subtraction is evaluated at well-conditioned scale
        to avoid catastrophic cancellation when differencing SSB-scale
        vectors. `i` is the follower's index in the fleet (>= 1).
        `a_srp_diff` is the precomputed differential SRP (follower − leader).
        """

        da_tot = (self.acc_grav_rel(r_L, delta_r, bodies)
                  + a_srp_diff
                  + self.acc_grav_isc_rel(i, delta_r_all, m_total_all)
                  + (a_ctrl_F - a_ctrl_L))
        return da_tot

    def acc_grav_abs(self, r_I, bodies):
        """N-body Solar System gravity at absolute r_I [km]. → [km/s^2].
        Reads body positions from the per-step snapshot `bodies`."""
        if not self.flags.grav_nbody:
            return np.zeros(3)
        v_dot = np.zeros(3)
        for body, mu in self.env.GM.items():
            dr    = r_I - bodies[body]
            v_dot -= mu * dr / np.linalg.norm(dr) ** 3
        return v_dot

    def acc_grav_rel(self, r_L, delta_r, bodies):
        """
        Differential N-body gravity on follower at r_L + δr. Evaluated as
            Σ_b [ a_b(r_L + δr) − a_b(r_L) ]
        with each term kept at body-relative scale (d2 = d1 + δr, never
        d2 = r_F − r_b reconstructed from SSB-based absolutes).
        Reads body positions from the per-step snapshot `bodies`.
        """
        if not self.flags.grav_nbody:
            return np.zeros(3)
        da = np.zeros(3)
        for body, mu in self.env.GM.items():
            r_b = bodies[body]
            d1  = r_L - r_b                       # body → leader
            d2  = d1 + delta_r                    # body → follower
            da -= mu * (d2 / np.linalg.norm(d2) ** 3
                      - d1 / np.linalg.norm(d1) ** 3)
        return da

    def acc_srp_abs(self, r_I, m_total, role, bodies):
        """
        Cannonball SRP at absolute position r_I; body of TOTAL mass `m_total`,
        projected area A(role), reflectivity c_reflect.  → [km/s^2].
        The Sun position is taken from the per-step snapshot `bodies` (no
        extra ephemeris query).
        """
        if not self.flags.srp:
            return np.zeros(3)

        if role == 'L':
            A   = self.param.SRP_area_L
        elif role == 'F':
            A   = self.param.SRP_area_F
        else:
            raise ValueError(f"Invalid role {role!r}: expected 'L' or 'F'")

        c_R = self.param.c_reflect

        r_sun  = bodies["SUN"]
        d      = r_I - r_sun
        d_norm = np.linalg.norm(d)
        u_hat  = d / d_norm                                            # sun → s/c (outward)
        P_at_r = const.P_SUN * (const.R_SUN_AU / d_norm) ** 2
        a_mag  = c_R * (A / m_total) * P_at_r * const.KM_PER_M
        return a_mag * u_hat

    # NOTE: acc_srp_rel has been removed. The differential SRP is now built in
    # x_dot from the two cached absolute SRP accelerations (a_srp_I_F −
    # a_srp_I_L), which is numerically identical to the old acc_srp_rel result
    # but avoids recomputing the follower SRP twice (once for translation,
    # once for torque).

    # ── Inter-spacecraft gravity ────────────────────────────────────────────

    def acc_grav_isc_abs(self, i, delta_r_all, m_total_all):
        """
        Newtonian gravity on spacecraft `i` from every other spacecraft:
            a_i^isc = Σ_{j ≠ i}  G m_j (r_j − r_i) / |r_j − r_i|³

        Pairwise vectors r_j − r_i are computed at RELATIVE scale via the
        δr snapshot (= δr_j − δr_i), never reconstructed from SSB-absolute
        positions. This keeps the small separation (~1e-4 km) accurate to
        machine precision regardless of where the formation sits in space.

        Units: G in km³/kg/s² (= const.G_KM), m in kg, r in km → km/s².
        Magnitude reference: at 100 m baseline between ~4000 kg combiner and
        ~3000 kg collector, |a| ≈ 3×10⁻¹⁴ km/s² ≈ 30 pm/s².
        """
        if not self.flags.grav_isc:
            return np.zeros(3)
        a = np.zeros(3)
        delta_r_i = delta_r_all[i]
        for j in range(self.n_sc):
            if j == i:
                continue
            # Pairwise vector at relative scale — leader's r_L cancels
            # algebraically, so we never form it.
            r_ji   = delta_r_all[j] - delta_r_i   # r_j − r_i, in km
            d_norm = np.linalg.norm(r_ji)
            a     += const.G_KM * m_total_all[j] * r_ji / d_norm ** 3
        return a

    def acc_grav_isc_rel(self, i, delta_r_all, m_total_all):
        """
        Differential ISC gravity: a_i^isc − a_L^isc, for follower i (≥ 1).

        Computed as a literal difference of two ISC sums. Safe because both
        quantities are themselves O(pm/s²) — no large-number cancellation
        risk — and the differential is the correct thing for the follower's
        relative-velocity equation.
        """
        if not self.flags.grav_isc:
            return np.zeros(3)
        a_i = self.acc_grav_isc_abs(i, delta_r_all, m_total_all)
        a_L = self.acc_grav_isc_abs(0, delta_r_all, m_total_all)
        return a_i - a_L


    # ── Rotational dynamics ─────────────────────────────────────────────────

    def attitude_kin_rhs(self, q, omega):
        """Quaternion kinematics: q̇ = ½ Ω(ω) q. NOT flag-gated — this is
        kinematics, not a perturbation."""
        return 0.5 * Omega_omega(omega) @ q

    def attitude_dyn_rhs(self, omega, m_prop, m_dot, m_total,
                         tau_ctrl, role,
                         q, r_I, a_srp_I, bodies):
        """
        Euler's rotational equation with a TIME-VARYING inertia:

            J(m_prop) ω̇ = τ_ctrl + τ_SRP + τ_grav + τ_p
                           − ω × (J ω) − J̇ ω

        J(m_prop) = J_cyl(role) + (m_ring_dry(role) + m_prop) · K_ring
        J̇         = ṁ_prop · K_ring

        J is built ONCE here and shared with attitude_tau_grav (which needs it
        for the gravity-gradient couple), so the new torque term costs no
        extra inertia build. The environmental torques receive the body-frame
        attitude `q`, the inertial position `r_I`, the cached SRP acceleration
        `a_srp_I`, and the per-step body snapshot `bodies`.

        The gyroscopic coupling (−ω × Jω) and the J̇ω term are flag-gated
        because they are *structural* parts of Euler's equation that the
        user may wish to study in isolation (e.g. "what does ignoring
        gyroscopic coupling do to my controller?").
        """
        J     = self._inertia_now(m_prop, role)
        J_dot = self._inertia_dot_now(m_dot)
        J_inv = np.linalg.inv(J)

        tau_tot = (tau_ctrl
                   + self.attitude_tau_srp(q, a_srp_I, m_total, role)
                   + self.attitude_tau_grav(q, r_I, J, bodies))

        if self.flags.gyro_coupling:
            tau_tot = tau_tot - np.cross(omega, J @ omega)

        if self.flags.j_dot_term:
            tau_tot = tau_tot - J_dot @ omega

        return J_inv @ tau_tot

    def attitude_tau_srp(self, q, a_srp_I, m_total, role):
        """
        SRP torque about the COM (design-doc eq. 12):

            τ_SRP^B = r_CP^B × f_SRP^B ,   f_SRP^B = m C_I^B(q) a_SRP^I

        Reuses the cannonball SRP acceleration `a_srp_I` already computed for
        the translational dynamics — no new Sun lookup or 1/r² evaluation.
        `r_CP^B` is the body-frame vector from the COM (body-frame origin) to
        the centre of pressure, supplied per role via self.param.

        Dependency: this term scales with the SRP force, so it is non-zero
        only when the translational SRP flag (`self.flags.srp`) is also on —
        if SRP is disabled, a_srp_I is zero and so is the torque, by design.

        Units: a_SRP^I [km/s²] → [m/s²] via / KM_PER_M, × m [kg] = f [N];
        r_CP^B [m] × f [N] = τ [N·m].
        """
        if not self.flags.tau_srp:
            return np.zeros(3)

        # Rotate SRP accel inertial → body, scale to a body-frame force [N].
        a_srp_B = quat_to_CTM(q) @ a_srp_I                  # km/s², body frame
        f_srp_B = (m_total * a_srp_B) / const.KM_PER_M      # N,     body frame

        r_cp = self.param.r_CP_L if role == 'L' else self.param.r_CP_F
        return np.cross(r_cp, f_srp_B)

    def attitude_tau_grav(self, q, r_I, J, bodies):
        """
        Gravity-gradient torque from every gravitating body (design-doc
        eq. 13):

            τ_grav^B = Σ_b (3 μ_b / |d_b|³) · û_b^B × (J û_b^B)

        with  d_b   = r_I − r_b   (inertial line of sight to body b)
              û_b^B = C_I^B(q) · d_b / |d_b|   (that line of sight in body axes).

        Body positions r_b are read from the per-step snapshot `bodies` (the
        same ones used by acc_grav_*), so this term adds no extra ephemeris
        lookups. J is the inertia about the COM, passed in from
        attitude_dyn_rhs (built once there).

        Units: μ_b [km³/s²], |d_b| [km] ⇒ 3μ_b/|d_b|³ [1/s²]; J [kg·m²] ⇒
        τ [kg·m²/s²] = [N·m]. The km length scale cancels in μ/|d|³, so no
        length conversion is required.

        Scale note: forming d_b = r_I − r_b directly at SSB scale is correct
        here — the line of sight to a distant body genuinely IS an SSB-scale
        vector, and only its *direction* is used (after normalisation). This
        is unlike the pairwise ISC separations, where the small difference
        itself is the quantity of interest and must be kept at relative scale.
        """
        if not self.flags.tau_grav:
            return np.zeros(3)

        tau  = np.zeros(3)
        C_IB = quat_to_CTM(q)                       # inertial → body
        for body, mu in self.env.GM.items():
            d      = r_I - bodies[body]             # r_sc − r_b  [km], inertial
            d_norm = np.linalg.norm(d)
            u_B    = C_IB @ (d / d_norm)            # line of sight, body frame
            tau   += (3.0 * mu / d_norm ** 3) * np.cross(u_B, J @ u_B)
        return tau


    # ── Mass dynamics ───────────────────────────────────────────────────────

    def mass_rhs(self, m_prop, T_cmd, role):
        """
        Per-thruster Tsiolkovsky mass flow, matching design-doc eq. (25):

            ṁ_prop = - (Σ_l T_l) / (Isp · g0)

        with `T_cmd` in newtons (already clamped to [0, T_max] by the caller).
        Each thruster's individual magnitude contributes additively — this
        correctly captures the propellant cost of opposing thruster firings
        used for torque generation.

        NOT flag-gated: mass change IS the rocket equation, not a perturbation.
        If you want to disable propellant burn for a test, set Isp = ∞.
        """
        if not self.flags.mass_change:
            return 0.0

        # Σ |T_l| in newtons → kN
        T_sum_kN = float(np.sum(T_cmd)) / const.N_PER_KN
        return - T_sum_kN / (self.param.ISP * const.G0)

    # ── Mass / inertia helpers ──────────────────────────────────────────────

    def _m_total(self, m_prop, role):
        """Total spacecraft mass = m_cyl + m_ring_dry + m_prop."""
        if role == 'L':
            return self.param.m_cylinder_L + self.param.m_ring_dry_L + max(m_prop, 0.0)
        elif role == 'F':
            return self.param.m_cylinder_F + self.param.m_ring_dry_F + max(m_prop, 0.0)
        else:
            raise ValueError(f"Invalid role {role!r}: expected 'L' or 'F'")

    def _inertia_now(self, m_prop, role):
        """
        Current J_B(m_prop) for the given role.
        J = (m_ring_dry + m_prop) · J_ring_unit + J_cyl
        """
        if role == 'L':
            m_ring_dry, J_cyl = self.param.m_ring_dry_L, self.param.J_cylinder_L
        elif role == 'F':
            m_ring_dry, J_cyl = self.param.m_ring_dry_F, self.param.J_cylinder_F
        else:
            raise ValueError(f"Invalid role {role!r}: expected 'L' or 'F'")

        m_ring = m_ring_dry + max(m_prop, 0.0)
        return m_ring * J_ring_unit(self.param.r_in, self.param.r_out,
                                     self.param.h_ring) + J_cyl

    def _inertia_dot_now(self, m_dot):
        """
        dJ/dt. Cylinder mass and dry-ring mass are constant ⇒ only the
        propellant contributes. J_ring is linear in (m_ring_dry + m_prop),
        so dJ/dt = ṁ_prop · K_ring. Geometry (r_in, r_out, h_ring) is
        shared between leader and follower, so no role dispatch is needed.
        """
        return m_dot * J_ring_unit(self.param.r_in, self.param.r_out,
                                    self.param.h_ring)

    # ── Slicing / quaternion utilities ──────────────────────────────────────

    @staticmethod
    def _renormalize_quat(x_next, n_sc, dim_x_sc):
        for i in range(n_sc):
            q_slice = slice(dim_x_sc * i + 6, dim_x_sc * i + 10)
            x_next[q_slice] = x_next[q_slice] / np.linalg.norm(x_next[q_slice])
        return x_next

    @staticmethod
    def _slice_x(i, dim_x_sc):
        """State slice for spacecraft i (length dim_x_sc)."""
        return slice(dim_x_sc * i, dim_x_sc * (i + 1))

    @staticmethod
    def _slice_u(i, dim_u_sc):
        """Control slice for spacecraft i (length dim_u_sc)."""
        return slice(dim_u_sc * i, dim_u_sc * (i + 1))