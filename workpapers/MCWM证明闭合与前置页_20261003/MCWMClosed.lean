import Mathlib

/-!
Closed finite-model proofs for Appendix E. The kernel may be substochastic
to represent termination. No declarations from the historical axiom scaffold
are imported. The full residual update contraction remains an explicit premise.
-/

noncomputable section
open scoped BigOperators
open Filter

namespace ThesisMCWM

set_option linter.unusedSectionVars false

variable {X : Type*} [Fintype X]

structure Kernel (X : Type*) [Fintype X] where
  p : X → X → ℝ
  nonneg : ∀ x y, 0 ≤ p x y
  mass_le : ∀ x, ∑ y, p x y ≤ 1

def average (K : Kernel X) (v : X → ℝ) (x : X) : ℝ :=
  ∑ y, K.p x y * v y

theorem average_bound (K : Kernel X) (v : X → ℝ) (x : X) :
    |average K v x| ≤ ‖v‖ := by
  calc
    |average K v x| ≤ ∑ y, |K.p x y * v y| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ y, K.p x y * |v y| := by
      apply Finset.sum_congr rfl
      intro y _
      rw [abs_mul, abs_of_nonneg (K.nonneg x y)]
    _ ≤ ∑ y, K.p x y * ‖v‖ := by
      apply Finset.sum_le_sum
      intro y _
      exact mul_le_mul_of_nonneg_left
        (by simpa only [Real.norm_eq_abs] using norm_le_pi_norm v y) (K.nonneg x y)
    _ = (∑ y, K.p x y) * ‖v‖ := (Finset.sum_mul ..).symm
    _ ≤ ‖v‖ := by nlinarith [K.mass_le x, norm_nonneg v]

def bellman (K : Kernel X) (r : X → ℝ) (g : ℝ) (v : X → ℝ) : X → ℝ :=
  fun x => r x + g * average K v x

theorem bellman_diff (K : Kernel X) (r : X → ℝ) (g : ℝ) (v w : X → ℝ) (x : X) :
    bellman K r g v x - bellman K r g w x = g * average K (v - w) x := by
  simp only [bellman, average, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
  ring

theorem bellman_contraction (K : Kernel X) (r : X → ℝ) (g : ℝ) (hg : 0 ≤ g)
    (v w : X → ℝ) : ‖bellman K r g v - bellman K r g w‖ ≤ g * ‖v - w‖ := by
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg hg (norm_nonneg _))).2
  intro x
  simp only [Pi.sub_apply, Real.norm_eq_abs, bellman_diff, abs_mul, abs_of_nonneg hg]
  exact mul_le_mul_of_nonneg_left (average_bound K (v - w) x) hg

theorem bellman_contracting (K : Kernel X) (r : X → ℝ) (g : ℝ)
    (hg : 0 ≤ g) (hg1 : g < 1) : ContractingWith ⟨g, hg⟩ (bellman K r g) := by
  refine ⟨hg1, LipschitzWith.of_dist_le_mul ?_⟩
  intro v w
  simpa only [dist_eq_norm, NNReal.coe_mk] using bellman_contraction K r g hg v w

def value (K : Kernel X) (r : X → ℝ) (g : ℝ) (hg : 0 ≤ g) (hg1 : g < 1) : X → ℝ :=
  ContractingWith.fixedPoint (bellman K r g) (bellman_contracting K r g hg hg1)

theorem value_fixed (K : Kernel X) (r : X → ℝ) (g : ℝ) (hg : 0 ≤ g) (hg1 : g < 1) :
    bellman K r g (value K r g hg hg1) = value K r g hg hg1 :=
  (bellman_contracting K r g hg hg1).fixedPoint_isFixedPt

theorem value_unique (K : Kernel X) (r : X → ℝ) (g : ℝ) (hg : 0 ≤ g) (hg1 : g < 1)
    (v : X → ℝ) (hv : bellman K r g v = v) : v = value K r g hg hg1 :=
  (bellman_contracting K r g hg hg1).fixedPoint_unique hv

theorem value_bound (K : Kernel X) (r : X → ℝ) (g R : ℝ)
    (hg : 0 ≤ g) (hg1 : g < 1) (hR : 0 ≤ R) (hr : ∀ x, |r x| ≤ R) :
    ‖value K r g hg hg1‖ ≤ R / (1 - g) := by
  let v := value K r g hg hg1
  have hp : 0 < 1 - g := sub_pos.mpr hg1
  have hv : ‖v‖ ≤ R + g * ‖v‖ := by
    apply (pi_norm_le_iff_of_nonneg (by positivity)).2
    intro x
    have eqx := congrFun (value_fixed K r g hg hg1) x
    change bellman K r g v x = v x at eqx
    calc
      ‖v x‖ = |r x + g * average K v x| := by
        rw [← eqx]; rfl
      _ ≤ |r x| + |g * average K v x| := abs_add_le _ _
      _ ≤ R + g * ‖v‖ := by
        rw [abs_mul, abs_of_nonneg hg]
        exact add_le_add (hr x) (mul_le_mul_of_nonneg_left (average_bound K v x) hg)
  apply (le_div_iff₀ hp).2
  nlinarith

def tv (K L : Kernel X) (x : X) : ℝ := (1 / 2) * ∑ y, |K.p x y - L.p x y|

theorem average_kernel_diff (K L : Kernel X) (v : X → ℝ) (x : X) :
    |average K v x - average L v x| ≤ 2 * ‖v‖ * tv K L x := by
  calc
    |average K v x - average L v x| = |∑ y, (K.p x y - L.p x y) * v y| := by
      simp only [average, sub_mul, Finset.sum_sub_distrib]
    _ ≤ ∑ y, |(K.p x y - L.p x y) * v y| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ y, |K.p x y - L.p x y| * |v y| := by
      simp only [abs_mul]
    _ ≤ ∑ y, |K.p x y - L.p x y| * ‖v‖ := by
      apply Finset.sum_le_sum
      intro y _
      exact mul_le_mul_of_nonneg_left
        (by simpa only [Real.norm_eq_abs] using norm_le_pi_norm v y) (abs_nonneg _)
    _ = 2 * ‖v‖ * tv K L x := by rw [← Finset.sum_mul]; unfold tv; ring

theorem one_step_error (K L : Kernel X) (r t v : X → ℝ) (g er ep : ℝ)
    (hg : 0 ≤ g) (her : 0 ≤ er) (hep : 0 ≤ ep)
    (hr : ∀ x, |r x - t x| ≤ er) (hP : ∀ x, tv K L x ≤ ep) :
    ‖bellman K r g v - bellman L t g v‖ ≤ er + 2 * g * ‖v‖ * ep := by
  apply (pi_norm_le_iff_of_nonneg (by positivity)).2
  intro x
  have hd : bellman K r g v x - bellman L t g v x =
      (r x - t x) + g * (average K v x - average L v x) := by unfold bellman; ring
  simp only [Pi.sub_apply, Real.norm_eq_abs, hd]
  calc
    |r x - t x + g * (average K v x - average L v x)|
        ≤ |r x - t x| + |g * (average K v x - average L v x)| := abs_add_le _ _
    _ ≤ er + g * (2 * ‖v‖ * tv K L x) := by
      rw [abs_mul, abs_of_nonneg hg]
      exact add_le_add (hr x) (mul_le_mul_of_nonneg_left (average_kernel_diff K L v x) hg)
    _ ≤ er + g * (2 * ‖v‖ * ep) := by gcongr; exact hP x
    _ = er + 2 * g * ‖v‖ * ep := by ring

theorem value_error (K L : Kernel X) (r t : X → ℝ) (g R er ep : ℝ)
    (hg : 0 ≤ g) (hg1 : g < 1) (hR : 0 ≤ R) (her : 0 ≤ er) (hep : 0 ≤ ep)
    (hr : ∀ x, |r x| ≤ R) (hrt : ∀ x, |r x - t x| ≤ er)
    (hP : ∀ x, tv K L x ≤ ep) :
    ‖value K r g hg hg1 - value L t g hg hg1‖ ≤
      er / (1 - g) + (2 * g * R) / (1 - g)^2 * ep := by
  let v := value K r g hg hg1
  let w := value L t g hg hg1
  have hp : 0 < 1 - g := sub_pos.mpr hg1
  have hv : ‖v‖ ≤ R / (1 - g) := value_bound K r g R hg hg1 hR hr
  have triangle : ‖v - w‖ ≤
      ‖bellman K r g v - bellman L t g v‖ +
      ‖bellman L t g v - bellman L t g w‖ := by
    rw [value_fixed K r g hg hg1, value_fixed L t g hg hg1]
    exact norm_sub_le_norm_sub_add_norm_sub _ _ _
  have hb := one_step_error K L r t v g er ep hg her hep hrt hP
  have hc := bellman_contraction L t g hg v w
  have hd : ‖v - w‖ * (1 - g) ≤ er + 2 * g * (R / (1 - g)) * ep := by
    have hm : 2 * g * ‖v‖ * ep ≤ 2 * g * (R / (1 - g)) * ep := by gcongr
    nlinarith
  have he := (le_div_iff₀ hp).2 hd
  have algebra : (er + 2 * g * (R / (1 - g)) * ep) / (1 - g) =
      er / (1 - g) + (2 * g * R) / (1 - g)^2 * ep := by field_simp
  exact he.trans_eq algebra

theorem full_update_fixed_point (F : (X → ℝ) → (X → ℝ)) (L : ℝ)
    (hL : 0 ≤ L) (hL1 : L < 1)
    (hF : ∀ v w, ‖F v - F w‖ ≤ L * ‖v - w‖) :
    ∃! v, F v = v := by
  have hc : ContractingWith ⟨L, hL⟩ F := ⟨hL1, LipschitzWith.of_dist_le_mul
    (fun v w => by simpa only [dist_eq_norm, NNReal.coe_mk] using hF v w)⟩
  exact ⟨hc.fixedPoint F, hc.fixedPoint_isFixedPt, fun v hv => hc.fixedPoint_unique hv⟩

theorem iteration_rate (F : (X → ℝ) → (X → ℝ)) (L : ℝ)
    (hL : 0 ≤ L) (hF : ∀ v w, ‖F v - F w‖ ≤ L * ‖v - w‖)
    (vstar v : X → ℝ) (hs : F vstar = vstar) (n : ℕ) :
    ‖F^[n] v - vstar‖ ≤ L^n * ‖v - vstar‖ := by
  induction n with
  | zero => simp
  | succ n ih =>
    calc
      ‖F^[n+1] v - vstar‖ = ‖F (F^[n] v) - F vstar‖ := by
        rw [Function.iterate_succ_apply', hs]
      _ ≤ L * ‖F^[n] v - vstar‖ := hF _ _
      _ ≤ L * (L^n * ‖v - vstar‖) := mul_le_mul_of_nonneg_left ih hL
      _ = L^(n+1) * ‖v - vstar‖ := by rw [pow_succ]; ring

theorem scaled_bellman (K : Kernel X) (r v : X → ℝ) (g a : ℝ) :
    bellman K (fun x => a * r x) g (fun x => a * v x) =
      fun x => a * bellman K r g v x := by
  funext x
  have hs : average K (fun y => a * v y) x = a * average K v x := by
    unfold average
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro y _
    ring
  simp only [bellman, hs]
  ring

theorem value_scaling (K : Kernel X) (r : X → ℝ) (g a : ℝ)
    (hg : 0 ≤ g) (hg1 : g < 1) :
    value K (fun x => a * r x) g hg hg1 = fun x => a * value K r g hg hg1 x := by
  symm
  apply value_unique
  rw [scaled_bellman, value_fixed]

theorem confidence_normalization (K : Kernel X) (c : X → ℝ) (g : ℝ)
    (hg : 0 ≤ g) (hg1 : g < 1) :
    value K (fun x => (1-g) * Real.log (c x)) g hg hg1 =
      fun x => (1-g) * value K (fun x => Real.log (c x)) g hg hg1 x :=
  value_scaling K (fun x => Real.log (c x)) g (1-g) hg hg1

theorem exponential_scaling (z g : ℝ) :
    Real.exp ((1-g)*z) = (Real.exp z) ^ (1-g) := by
  rw [Real.rpow_def_of_pos (Real.exp_pos _), Real.log_exp]
  congr 1
  ring

theorem normalized_rank (q : X → ℝ) (g : ℝ) (hg1 : g < 1) (x y : X) :
    Real.exp ((1-g)*q x) < Real.exp ((1-g)*q y) ↔ Real.exp (q x) < Real.exp (q y) := by
  rw [Real.exp_lt_exp, Real.exp_lt_exp]
  exact mul_lt_mul_iff_right₀ (sub_pos.mpr hg1)

section Policy
variable {A : Type*} [Fintype A]

def policyKernel (P : X → A → X → ℝ) (pi : X → A → ℝ)
    (hP : ∀ x a y, 0 ≤ P x a y) (hPm : ∀ x a, ∑ y, P x a y = 1)
    (hpi : ∀ x a, 0 ≤ pi x a) (hpim : ∀ x, ∑ a, pi x a = 1) : Kernel X where
  p x y := ∑ a, pi x a * P x a y
  nonneg x y := Finset.sum_nonneg (fun a _ => mul_nonneg (hpi x a) (hP x a y))
  mass_le x := by
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, hPm, mul_one, hpim, le_refl]

theorem policy_reward_bound (pi : X → A → ℝ) (r : X → A → ℝ) (R : ℝ)
    (hpi : ∀ x a, 0 ≤ pi x a) (hpim : ∀ x, ∑ a, pi x a = 1)
    (hr : ∀ x a, |r x a| ≤ R) (x : X) : |∑ a, pi x a * r x a| ≤ R := by
  calc
    |∑ a, pi x a * r x a| ≤ ∑ a, |pi x a * r x a| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ a, pi x a * |r x a| := by
      apply Finset.sum_congr rfl
      intro a _
      rw [abs_mul, abs_of_nonneg (hpi x a)]
    _ ≤ ∑ a, pi x a * R := by
      apply Finset.sum_le_sum
      intro a _
      exact mul_le_mul_of_nonneg_left (hr x a) (hpi x a)
    _ = R := by rw [← Finset.sum_mul, hpim, one_mul]

theorem policy_tv_bound (P Q : X → A → X → ℝ) (pi : X → A → ℝ) (ep : ℝ)
    (hP : ∀ x a y, 0 ≤ P x a y) (hPm : ∀ x a, ∑ y, P x a y = 1)
    (hQ : ∀ x a y, 0 ≤ Q x a y) (hQm : ∀ x a, ∑ y, Q x a y = 1)
    (hpi : ∀ x a, 0 ≤ pi x a) (hpim : ∀ x, ∑ a, pi x a = 1)
    (he : ∀ x a, (1/2 : ℝ) * ∑ y, |P x a y - Q x a y| ≤ ep) (x : X) :
    tv (policyKernel P pi hP hPm hpi hpim) (policyKernel Q pi hQ hQm hpi hpim) x ≤ ep := by
  have hsum : ∑ y, |(∑ a, pi x a * P x a y) - (∑ a, pi x a * Q x a y)|
      ≤ ∑ a, pi x a * ∑ y, |P x a y - Q x a y| := by
    calc
      _ = ∑ y, |∑ a, pi x a * (P x a y - Q x a y)| := by
        simp only [mul_sub, Finset.sum_sub_distrib]
      _ ≤ ∑ y, ∑ a, pi x a * |P x a y - Q x a y| := by
        apply Finset.sum_le_sum
        intro y _
        calc
          _ ≤ ∑ a, |pi x a * (P x a y - Q x a y)| := Finset.abs_sum_le_sum_abs _ _
          _ = _ := by simp only [abs_mul, abs_of_nonneg (hpi x _)]
      _ = _ := by rw [Finset.sum_comm]; simp only [Finset.mul_sum]
  unfold tv policyKernel
  change (1/2 : ℝ) * (∑ y, |(∑ a, pi x a * P x a y) - (∑ a, pi x a * Q x a y)|) ≤ ep
  calc
    _ ≤ (1/2 : ℝ) * ∑ a, pi x a * ∑ y, |P x a y - Q x a y| := by linarith
    _ = ∑ a, pi x a * ((1/2 : ℝ) * ∑ y, |P x a y - Q x a y|) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro a _
      ring
    _ ≤ ∑ a, pi x a * ep := by
      apply Finset.sum_le_sum
      intro a _
      exact mul_le_mul_of_nonneg_left (he x a) (hpi x a)
    _ = ep := by rw [← Finset.sum_mul, hpim, one_mul]

theorem residual_normalization (P d : X → A → X → ℝ)
    (hP : ∀ x a, ∑ y, P x a y = 1) (hd : ∀ x a, ∑ y, d x a y = 0) :
    ∀ x a, ∑ y, (P x a y + d x a y) = 1 := by
  intro x a
  rw [Finset.sum_add_distrib, hP, hd, add_zero]
end Policy

theorem constant_value (K : Kernel X) (b g : ℝ) (hg : 0 ≤ g) (hg1 : g < 1)
    (hK : ∀ x, ∑ y, K.p x y = 1) :
    value K (fun _ => b) g hg hg1 = fun _ => b / (1-g) := by
  symm
  apply value_unique
  funext x
  simp only [bellman, average, ← Finset.sum_mul, hK, one_mul]
  field_simp [ne_of_gt (sub_pos.mpr hg1)]
  ring

theorem constant_confidence (K : Kernel X) (c g : ℝ) (hc : 0 < c)
    (hg : 0 ≤ g) (hg1 : g < 1) (hK : ∀ x, ∑ y, K.p x y = 1) (x : X) :
    Real.exp (value K (fun _ => (1-g) * Real.log c) g hg hg1 x) = c := by
  rw [constant_value K ((1-g) * Real.log c) g hg hg1 hK]
  have hd : ((1-g) * Real.log c) / (1-g) = Real.log c := by
    field_simp [ne_of_gt (sub_pos.mpr hg1)]
  rw [hd, Real.exp_log hc]

def absorbing : Kernel Bool where
  p x y := if x = y then 1 else 0
  nonneg x y := by split <;> norm_num
  mass_le x := by cases x <;> simp [Fintype.sum_bool]

def resetZero : Kernel Bool where
  p _ y := if y then 0 else 1
  nonneg x y := by cases y <;> norm_num
  mass_le x := by simp [Fintype.sum_bool]

def switchOne : Kernel Bool where
  p _ y := if y then 1 else 0
  nonneg x y := by cases y <;> norm_num
  mass_le x := by simp [Fintype.sum_bool]

def stateReward (x : Bool) : ℝ := if x then 1 else 0

theorem absorbing_value (g : ℝ) (hg : 0 ≤ g) (hg1 : g < 1) :
    value absorbing stateReward g hg hg1 = fun x => if x then 1/(1-g) else 0 := by
  symm
  apply value_unique
  funext x
  cases x <;> simp [bellman, average, absorbing, stateReward, Fintype.sum_bool]
  field_simp [ne_of_gt (sub_pos.mpr hg1)]
  ring

theorem reset_value (g : ℝ) (hg : 0 ≤ g) (hg1 : g < 1) :
    value resetZero stateReward g hg hg1 = stateReward := by
  symm
  apply value_unique
  funext x
  cases x <;> simp [bellman, average, resetZero, stateReward, Fintype.sum_bool]

theorem switch_value (g : ℝ) (hg : 0 ≤ g) (hg1 : g < 1) :
    value switchOne stateReward g hg hg1 = fun x => if x then 1/(1-g) else g/(1-g) := by
  symm
  apply value_unique
  funext x
  cases x <;> simp [bellman, average, switchOne, stateReward, Fintype.sum_bool]
  · ring
  · field_simp [ne_of_gt (sub_pos.mpr hg1)]; ring

theorem zero_support_global_gap (g : ℝ) (hg : 0 < g) (hg1 : g < 1) :
    tv absorbing resetZero false = 0 ∧
    0 < value absorbing stateReward g hg.le hg1 true -
      value resetZero stateReward g hg.le hg1 true := by
  constructor
  · simp [tv, absorbing, resetZero, Fintype.sum_bool]
  · rw [absorbing_value, reset_value]
    change 0 < 1/(1-g) - 1
    have hp : 0 < 1-g := sub_pos.mpr hg1
    apply (sub_pos).2
    exact (lt_div_iff₀ hp).2 (by linarith)

-- Action false is the data policy; action true is the new policy.
def targetTransition (s a t : Bool) : ℝ :=
  if a then switchOne.p s t else absorbing.p s t
def learnedTransition (s _a t : Bool) : ℝ := absorbing.p s t

theorem unseen_action_gap (g : ℝ) (hg : 0 < g) (hg1 : g < 1) :
    (∀ t, targetTransition false false t = learnedTransition false false t) ∧
    value switchOne stateReward g hg.le hg1 false -
      value absorbing stateReward g hg.le hg1 false = g/(1-g) ∧ 0 < g/(1-g) := by
  constructor
  · intro t; rfl
  · rw [switch_value, absorbing_value]
    exact ⟨by simp, div_pos hg (sub_pos.mpr hg1)⟩

theorem error_limit (K : Kernel X) (Kn : ℕ → Kernel X) (r : X → ℝ)
    (g R : ℝ) (hg : 0 ≤ g) (hg1 : g < 1) (hR : 0 ≤ R)
    (hr : ∀ x, |r x| ≤ R) (ep : ℕ → ℝ) (hep : ∀ n, 0 ≤ ep n)
    (hP : ∀ n x, tv K (Kn n) x ≤ ep n) (he : Tendsto ep atTop (nhds 0)) :
    Tendsto (fun n => ‖value K r g hg hg1 - value (Kn n) r g hg hg1‖) atTop (nhds 0) := by
  have hb : ∀ n, ‖value K r g hg hg1 - value (Kn n) r g hg hg1‖ ≤
      ((2*g*R)/(1-g)^2) * ep n := by
    intro n
    simpa using value_error K (Kn n) r r g R 0 (ep n) hg hg1 hR (by norm_num)
      (hep n) hr (by intro x; simp) (hP n)
  apply squeeze_zero (fun n => norm_nonneg _) hb
  simpa using he.const_mul ((2*g*R)/(1-g)^2)

theorem positive_log_order (u v : ℝ) (hu : 0 < u) (hv : 0 < v) :
    Real.log u < Real.log v ↔ u < v := by
  constructor
  · intro h
    have he := Real.exp_lt_exp.mpr h
    simpa only [Real.exp_log hu, Real.exp_log hv] using he
  · exact Real.log_lt_log hu

theorem log_odds_order (u v : ℝ) (hu : 0 < u) (hu1 : u < 1)
    (hv : 0 < v) (hv1 : v < 1) :
    Real.log (u/(1-u)) < Real.log (v/(1-v)) ↔ u < v := by
  rw [positive_log_order _ _ (div_pos hu (sub_pos.mpr hu1))
    (div_pos hv (sub_pos.mpr hv1)), div_lt_div_iff₀ (sub_pos.mpr hu1) (sub_pos.mpr hv1)]
  constructor <;> intro h <;> nlinarith

theorem error_signal_rank (w e : X → ℝ) (tau c : ℝ) (ht : 0 < tau)
    (hw : ∀ x, 0 < w x) (he : ∀ x, Real.log (w x) = -e x/tau+c) (x y : X) :
    w x < w y ↔ e y < e x := by
  rw [← positive_log_order _ _ (hw x) (hw y), he x, he y]
  constructor
  · intro h
    have hd : -e x/tau < -e y/tau := by linarith
    have hm := (div_lt_div_iff_of_pos_right ht).mp hd
    linarith
  · intro h
    have hm : -e x < -e y := by linarith
    have hd := (div_lt_div_iff_of_pos_right ht).mpr hm
    linarith

theorem bounded_offset_rank (u v : X → ℝ) (c b : ℝ) (hb : 0 ≤ b)
    (he : ∀ x, |v x - u x - c| ≤ b) (x y : X) (hgap : 2*b < |u x-u y|) :
    u x < u y ↔ v x < v y := by
  have hx := (abs_le.mp (he x))
  have hy := (abs_le.mp (he y))
  rcases (lt_abs.mp hgap) with h | h
  · constructor <;> intro hlt <;> linarith
  · constructor <;> intro hlt <;> linarith

theorem conditional_rank (w phi ereal efit : X → ℝ) (tau c mean ep : ℝ)
    (ht : 0 < tau) (hep : 0 ≤ ep)
    (hw : ∀ x, 0 < w x) (hphi : ∀ x, 0 < phi x ∧ phi x < 1)
    (hwl : ∀ x, Real.log (w x) = -ereal x/tau+c)
    (hpl : ∀ x, Real.log (phi x/(1-phi x)) = (efit x-ereal x)/tau)
    (he : ∀ x, |efit x-mean| ≤ ep) (x y : X)
    (hgap : 2*ep/tau < |Real.log (w x)-Real.log (w y)|) :
    w x < w y ↔ phi x < phi y := by
  let u := fun x => Real.log (w x)
  let v := fun x => Real.log (phi x/(1-phi x))
  have hb : ∀ x, |v x-u x-(mean/tau-c)| ≤ ep/tau := by
    intro x
    have eqn : v x-u x-(mean/tau-c) = (efit x-mean)/tau := by
      dsimp [u,v]
      rw [hwl, hpl]
      ring
    rw [eqn, abs_div, abs_of_pos ht]
    exact (div_le_div_iff_of_pos_right ht).mpr (he x)
  have hr := bounded_offset_rank u v (mean/tau-c) (ep/tau) (div_nonneg hep ht.le)
    hb x y (by dsimp [u]; convert hgap using 1 <;> ring)
  exact (positive_log_order _ _ (hw x) (hw y)).symm.trans
    (hr.trans (log_odds_order _ _ (hphi x).1 (hphi x).2 (hphi y).1 (hphi y).2))

theorem uniform_rank (w phi ereal efit : X → ℝ) (tau c mean : ℝ)
    (ht : 0 < tau) (hw : ∀ x, 0 < w x) (hphi : ∀ x, 0 < phi x ∧ phi x < 1)
    (hwl : ∀ x, Real.log (w x) = -ereal x/tau+c)
    (hpl : ∀ x, Real.log (phi x/(1-phi x)) = (efit x-ereal x)/tau)
    (he : ∀ x, efit x = mean) (x y : X) :
    w x < w y ↔ phi x < phi y := by
  rw [← positive_log_order _ _ (hw x) (hw y), ← log_odds_order _ _
    (hphi x).1 (hphi x).2 (hphi y).1 (hphi y).2, hwl x, hwl y, hpl x, hpl y, he x, he y]
  have eqn : (mean-ereal x)/tau-(mean-ereal y)/tau =
      (-ereal x/tau+c)-(-ereal y/tau+c) := by ring
  constructor <;> intro h <;> linarith

theorem value_error_limit (K : Kernel X) (Kn : ℕ → Kernel X)
    (r : X → ℝ) (rn : ℕ → X → ℝ) (g R : ℝ)
    (hg : 0 ≤ g) (hg1 : g < 1) (hR : 0 ≤ R) (hr : ∀ x, |r x| ≤ R)
    (er ep : ℕ → ℝ) (her : ∀ n, 0 ≤ er n) (hep : ∀ n, 0 ≤ ep n)
    (hrn : ∀ n x, |r x-rn n x| ≤ er n) (hP : ∀ n x, tv K (Kn n) x ≤ ep n)
    (her0 : Tendsto er atTop (nhds 0)) (hep0 : Tendsto ep atTop (nhds 0)) :
    Tendsto (fun n => ‖value K r g hg hg1 - value (Kn n) (rn n) g hg hg1‖)
      atTop (nhds 0) := by
  have ht : Tendsto (fun n => er n/(1-g)+(2*g*R)/(1-g)^2*ep n) atTop (nhds 0) := by
    simpa using (her0.div_const (1-g)).add (hep0.const_mul ((2*g*R)/(1-g)^2))
  exact squeeze_zero (fun n => norm_nonneg _) (fun n =>
    value_error K (Kn n) r (rn n) g R (er n) (ep n) hg hg1 hR (her n) (hep n)
      hr (hrn n) (hP n)) ht

theorem full_update_convergence (F : (X → ℝ) → (X → ℝ)) (L : ℝ)
    (hL : 0 ≤ L) (hL1 : L < 1)
    (hF : ∀ v w, ‖F v - F w‖ ≤ L * ‖v - w‖) (v : X → ℝ) :
    ∃ vstar, F vstar = vstar ∧ Tendsto (fun n => F^[n] v) atTop (nhds vstar) := by
  have hc : ContractingWith ⟨L, hL⟩ F := ⟨hL1, LipschitzWith.of_dist_le_mul
    (fun v w => by simpa only [dist_eq_norm, NNReal.coe_mk] using hF v w)⟩
  exact ⟨hc.fixedPoint F, hc.fixedPoint_isFixedPt, hc.tendsto_iterate_fixedPoint v⟩

-- Equal Gaussian normalizers and equal class priors cancel in this ratio.
theorem gaussian_score_identity (er ef tau : ℝ) :
    let p := Real.exp (-er/tau) / (Real.exp (-er/tau)+Real.exp (-ef/tau))
    0 < p ∧ p < 1 ∧ Real.log (p/(1-p)) = (ef-er)/tau := by
  dsimp
  have ha := Real.exp_pos (-er/tau)
  have hb := Real.exp_pos (-ef/tau)
  have hab : 0 < Real.exp (-er/tau)+Real.exp (-ef/tau) := add_pos ha hb
  refine ⟨div_pos ha hab, (div_lt_one hab).2 (by linarith), ?_⟩
  have eqn : (Real.exp (-er/tau)/(Real.exp (-er/tau)+Real.exp (-ef/tau))) /
      (1-Real.exp (-er/tau)/(Real.exp (-er/tau)+Real.exp (-ef/tau))) =
      Real.exp (-er/tau)/Real.exp (-ef/tau) := by
    field_simp
    ring
  rw [eqn, Real.log_div (ne_of_gt ha) (ne_of_gt hb), Real.log_exp, Real.log_exp]
  ring

#print axioms average_bound
#print axioms bellman_contracting
#print axioms value_fixed
#print axioms value_unique
#print axioms value_bound
#print axioms average_kernel_diff
#print axioms value_error
#print axioms full_update_fixed_point
#print axioms iteration_rate
#print axioms confidence_normalization
#print axioms exponential_scaling
#print axioms normalized_rank
#print axioms policyKernel
#print axioms policy_reward_bound
#print axioms policy_tv_bound
#print axioms residual_normalization
#print axioms constant_value
#print axioms constant_confidence
#print axioms zero_support_global_gap
#print axioms unseen_action_gap
#print axioms error_limit
#print axioms error_signal_rank
#print axioms conditional_rank
#print axioms uniform_rank
#print axioms value_error_limit
#print axioms full_update_convergence
#print axioms gaussian_score_identity

end ThesisMCWM
