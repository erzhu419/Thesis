import ResidualMDP

#print axioms MCWM_ResidualMDP.bellman_residual_contraction
#print axioms MCWM_ResidualMDP.residual_simulation_lemma
#print axioms MCWM_ResidualMDP.residual_fixed_point_exists
#print axioms MCWM_ResidualMDP.qdelta_icrl_rank_equivalence_uniform
#print axioms MCWM_ResidualMDP.qdelta_icrl_rank_equivalence_bounded

namespace ThesisMCWMAudit

-- These witnesses audit existing declarations; they do not change the source.
theorem unrestricted_classifier_axiom_is_inconsistent : False := by
  have h := MCWM_ResidualMDP.phi_bayes_optimal_logodds
    (S := Fin 1) (A := Fin 1)
    (fun _ _ => 0) (fun _ _ => 0) (fun _ _ => 0) 1 (by norm_num) 0 0
  exact (lt_irrefl (0 : ℝ)) h.1

theorem unrestricted_logvalue_axiom_is_inconsistent : False := by
  have h := MCWM_ResidualMDP.qdelta_gaussian_logvalue
    (S := Fin 1) (A := Fin 1)
    (fun _ _ => 1) (fun _ _ => 0) 1 (by norm_num) 1 0 0
  norm_num at h

end ThesisMCWMAudit
