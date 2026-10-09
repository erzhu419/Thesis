import RESAC
import SoftBellman
import «RESAC-Counterproof»
import BAPR
import «BAPR-Counterproof»
import ActionCompensation
import CSBAPR
import «CSBAPR-Counterproof»
import ResidualMDP
import DRO
import DROBellman

#print axioms RESAC.RE_SAC_contraction
#print axioms SoftBellman.soft_bellman_contraction
#print axioms Wasserstein.dro_upper_bound
#print axioms DROBellman.robustBellmanOp_contraction
#print axioms DROBellman.mixture_robust_bellman_contraction
#print axioms DROBellman.wasserstein_robust_bellman_lipBound
#print axioms BAPR.bapr_contraction
#print axioms BAPR.bapr_joint_contraction
#print axioms BAPR.bapr_contraction_after_update
#print axioms BAPR.bapr_contraction_via_marginal
#print axioms BAPR.beta_eff_monotone_conservatism
#print axioms BAPR.Counter.T_bad_exact_factor
#print axioms BAPR.Counter.detection_delay_sufficient
#print axioms BAPR.ActionCompensation.exact_compensation_iff_feasible
#print axioms BAPR.ActionCompensation.polarity_compensates
#print axioms BAPR.ActionCompensation.polarity_compensated_inBox
#print axioms BAPR.ActionCompensation.arbitrary_polarity_switching
#print axioms BAPR.ActionCompensation.causalCommand_prefix
#print axioms BAPR.ActionCompensation.indistinguishable_binary_accuracy
#print axioms BAPR.ActionCompensation.finite_conditional_kl_chain_step
#print axioms BAPR.ActionCompensation.action_weighted_mode_error
#print axioms BAPR.ActionCompensation.modeError_l2Norm
#print axioms BAPR.ActionCompensation.pathwise_return_difference
#print axioms BAPR.ActionCompensation.pathwise_return_bound
#print axioms BAPR.ActionCompensation.expected_return_difference
#print axioms BAPR.ActionCompensation.expected_return_bound
#print axioms BAPR.ActionCompensation.polarity_expected_return_bound
#print axioms CSBAPR.error_growth_linear
#print axioms CSBAPR.scbapr_ood_stability_bound
#print axioms CSBAPR.scbapr_ood_linear_nD
#print axioms CSBAPR.scbapr_ood_quadratic_path
#print axioms CSBAPR.scbapr_ood_quadratic_nD
#print axioms CSBAPR.nau_derivative_is_constant
#print axioms CSBAPR.nmu_has_bounded_arithmetic_bias
#print axioms CSBAPR.relu_derivative_not_lipschitz
#print axioms CSBAPR.composed_deriv_lipschitz_simple
#print axioms CSBAPR.sindy_implies_jacobian_consistency_nD
#print axioms CSBAPR.sindy_to_ood_bound_nD
#print axioms CSBAPR.irm_tightens_jacobian_consistency
#print axioms CSBAPR.irm_vs_single_env_gap
#print axioms CSBAPR.jc_generalization_to_ood
#print axioms CSBAPR.csbapr_contraction
#print axioms MCWM_ResidualMDP.bellman_residual_contraction
#print axioms MCWM_ResidualMDP.residual_simulation_lemma
#print axioms MCWM_ResidualMDP.residual_fixed_point_exists
#print axioms MCWM_ResidualMDP.qdelta_icrl_rank_equivalence_uniform
#print axioms MCWM_ResidualMDP.qdelta_icrl_rank_equivalence_bounded

namespace ThesisConfidence

-- The thesis extension uses a fixed, linear propagation relation.
theorem normalized_fixed_point_scaling {X : Type*} [Fintype X]
    (K : X -> X -> Real) (c Q : X -> Real) (gamma : Real)
    (fixed : forall x, Q x = c x + gamma * Finset.univ.sum (fun y => K x y * Q y)) :
    forall x, (1-gamma) * Q x = (1-gamma) * c x +
      gamma * Finset.univ.sum (fun y => K x y * ((1-gamma) * Q y)) := by
  intro x
  have linear : Finset.univ.sum (fun y => K x y * ((1-gamma) * Q y)) =
      (1-gamma) * Finset.univ.sum (fun y => K x y * Q y) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro y _
    ring
  rw [linear]
  calc
    (1-gamma) * Q x = (1-gamma) *
        (c x + gamma * Finset.univ.sum (fun y => K x y * Q y)) :=
      congrArg (fun z => (1-gamma) * z) (fixed x)
    _ = _ := by ring

#print axioms normalized_fixed_point_scaling

end ThesisConfidence
