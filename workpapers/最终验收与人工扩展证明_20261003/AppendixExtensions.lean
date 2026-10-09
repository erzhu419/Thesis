import Mathlib
import CSBAPR
import ActionCompensation

/-! Supplementary proofs for the thesis, independent of the historical MC-WM axioms. -/
noncomputable section
open scoped BigOperators NNReal
open Set

namespace ThesisExtensions
set_option linter.unusedSectionVars false

variable {X : Type*} [Fintype X]

structure Probability (X : Type*) [Fintype X] where
  mass : X → ℝ
  nonneg : ∀ x, 0 ≤ mass x
  total : ∑ x, mass x = 1

def expectation (p : Probability X) (v : X → ℝ) : ℝ := ∑ x, p.mass x * v x

theorem expectation_bound (p : Probability X) (v : X → ℝ) :
    |expectation p v| ≤ ‖v‖ := by
  calc
    |expectation p v| ≤ ∑ x, |p.mass x * v x| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x, p.mass x * |v x| := by
      simp only [abs_mul, abs_of_nonneg (p.nonneg _)]
    _ ≤ ∑ x, p.mass x * ‖v‖ := by
      apply Finset.sum_le_sum
      intro x _
      exact mul_le_mul_of_nonneg_left
        (by simpa only [Real.norm_eq_abs] using norm_le_pi_norm v x) (p.nonneg x)
    _ = ‖v‖ := by rw [← Finset.sum_mul, p.total, one_mul]

def expectations (U : Set (Probability X)) (v : X → ℝ) : Set ℝ :=
  (fun p => expectation p v) '' U

theorem expectations_bounded (U : Set (Probability X)) (v : X → ℝ) :
    BddBelow (expectations U v) ∧ BddAbove (expectations U v) := by
  constructor
  · refine ⟨-‖v‖, ?_⟩
    rintro _ ⟨p, _, rfl⟩
    exact (abs_le.mp (expectation_bound p v)).1
  · refine ⟨‖v‖, ?_⟩
    rintro _ ⟨p, _, rfl⟩
    exact (abs_le.mp (expectation_bound p v)).2

theorem expectation_neg (p : Probability X) (v : X → ℝ) :
    expectation p (-v) = -expectation p v := by
  simp [expectation, Finset.sum_neg_distrib]

def support (U : Set (Probability X)) (v : X → ℝ) : ℝ := sSup (expectations U v)
def robustPenalty (U : Set (Probability X)) (p₀ : Probability X) (v : X → ℝ) : ℝ :=
  expectation p₀ v + support U (-v)

theorem support_neg_is_inf (U : Set (Probability X)) (hu : U.Nonempty) (v : X → ℝ) :
    -support U (-v) = sInf (expectations U v) := by
  have hn := hu.image (fun p => expectation p v)
  have hn' := hu.image (fun p => expectation p (-v))
  have hb := expectations_bounded U v
  have hb' := expectations_bounded U (-v)
  apply le_antisymm
  · apply le_csInf hn
    rintro _ ⟨p, hp, rfl⟩
    have h := le_csSup hb'.2 (show expectation p (-v) ∈ expectations U (-v) from ⟨p, hp, rfl⟩)
    rw [expectation_neg] at h
    unfold support
    linarith
  · have hs : support U (-v) ≤ -sInf (expectations U v) := by
      apply csSup_le hn'
      rintro _ ⟨p, hp, rfl⟩
      have h := csInf_le hb.1 (show expectation p v ∈ expectations U v from ⟨p, hp, rfl⟩)
      change expectation p (-v) ≤ -sInf (expectations U v)
      rw [expectation_neg]
      linarith
    linarith

theorem robust_regularization_identity (U : Set (Probability X)) (p₀ : Probability X)
    (hp₀ : p₀ ∈ U) (v : X → ℝ) :
    expectation p₀ v - robustPenalty U p₀ v = sInf (expectations U v) ∧
    0 ≤ robustPenalty U p₀ v := by
  have hi := support_neg_is_inf U ⟨p₀, hp₀⟩ v
  have hle := csInf_le (expectations_bounded U v).1
    (show expectation p₀ v ∈ expectations U v from ⟨p₀, hp₀, rfl⟩)
  constructor
  · simpa [robustPenalty] using hi
  · unfold robustPenalty
    linarith

section GeneralBackup
variable {Y : Type*} [Fintype Y]

theorem monotone_translation_nonexpansive (V : (Y → ℝ) → X → ℝ)
    (hmono : Monotone V)
    (hshift : ∀ Q c, V (fun y => Q y + c) = fun x => V Q x + c)
    (Q₁ Q₂ : Y → ℝ) : ‖V Q₁ - V Q₂‖ ≤ ‖Q₁ - Q₂‖ := by
  let ε := ‖Q₁ - Q₂‖
  have hpt : ∀ y, |Q₁ y - Q₂ y| ≤ ε := by
    intro y
    simpa [ε, Real.norm_eq_abs] using norm_le_pi_norm (Q₁ - Q₂) y
  have hhi : Q₁ ≤ fun y => Q₂ y + ε := by
    intro y
    have h := (abs_le.mp (hpt y)).2
    linarith
  have hlo : (fun y => Q₂ y + (-ε)) ≤ Q₁ := by
    intro y
    have h := (abs_le.mp (hpt y)).1
    linarith
  have vhi := hmono hhi
  have vlo := hmono hlo
  rw [hshift] at vhi vlo
  apply (pi_norm_le_iff_of_nonneg (norm_nonneg _)).2
  intro x
  simp only [Pi.sub_apply, Real.norm_eq_abs]
  apply abs_le.mpr
  constructor <;> linarith [vhi x, vlo x]

def generalBackup (P : Y → Probability X) (R : Y → ℝ)
    (V : (Y → ℝ) → X → ℝ) (γ : ℝ) (Q : Y → ℝ) : Y → ℝ :=
  fun y => R y + γ * expectation (P y) (V Q)

theorem general_backup_monotone (P : Y → Probability X) (R : Y → ℝ)
    (V : (Y → ℝ) → X → ℝ) (γ : ℝ) (hγ : 0 ≤ γ)
    (hmono : Monotone V) : Monotone (generalBackup P R V γ) := by
  intro Q₁ Q₂ hQ y
  dsimp [generalBackup, expectation]
  refine add_le_add (le_refl (R y)) (mul_le_mul_of_nonneg_left ?_ hγ)
  apply Finset.sum_le_sum
  intro x _
  exact mul_le_mul_of_nonneg_left (hmono hQ x) ((P y).nonneg x)

theorem general_backup_shift (P : Y → Probability X) (R : Y → ℝ)
    (V : (Y → ℝ) → X → ℝ) (γ : ℝ)
    (hshift : ∀ Q c, V (fun y => Q y + c) = fun x => V Q x + c)
    (Q : Y → ℝ) (c : ℝ) :
    generalBackup P R V γ (fun y => Q y + c) =
      fun y => generalBackup P R V γ Q y + γ * c := by
  funext y
  simp only [generalBackup, hshift, expectation, mul_add, Finset.sum_add_distrib,
    ← Finset.sum_mul, (P y).total]
  ring

def mixtureProbability {Z : Type*} [Fintype Z]
    (ρ : Probability Z) (P : Z → Probability X) : Probability X where
  mass x := ∑ z, ρ.mass z * (P z).mass x
  nonneg x := Finset.sum_nonneg (fun z _ => mul_nonneg (ρ.nonneg z) ((P z).nonneg x))
  total := by
    rw [Finset.sum_comm]
    simp only [← Finset.mul_sum, (P _).total, mul_one, ρ.total]

theorem mixture_backup_identity {Z : Type*} [Fintype Z]
    (ρ : Probability Z) (P : Z → Y → Probability X)
    (R : Z → Y → ℝ) (V : (Y → ℝ) → X → ℝ) (γ : ℝ) (Q : Y → ℝ) (y : Y) :
    (∑ z, ρ.mass z * generalBackup (P z) (R z) V γ Q y) =
      generalBackup (fun y => mixtureProbability ρ (fun z => P z y))
        (fun y => ∑ z, ρ.mass z * R z y) V γ Q y := by
  simp only [generalBackup, expectation, mixtureProbability, mul_add,
    Finset.sum_add_distrib, Finset.sum_mul]
  rw [Finset.sum_comm]
  congr 1
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro z _
  simp_rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x _
  ring

theorem general_backup_contraction (P : Y → Probability X) (R : Y → ℝ)
    (V : (Y → ℝ) → X → ℝ) (γ : ℝ) (hγ : 0 ≤ γ)
    (hmono : Monotone V)
    (hshift : ∀ Q c, V (fun y => Q y + c) = fun x => V Q x + c)
    (Q₁ Q₂ : Y → ℝ) :
    ‖generalBackup P R V γ Q₁ - generalBackup P R V γ Q₂‖ ≤ γ * ‖Q₁ - Q₂‖ := by
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg hγ (norm_nonneg _))).2
  intro y
  have he : generalBackup P R V γ Q₁ y - generalBackup P R V γ Q₂ y =
      γ * expectation (P y) (V Q₁ - V Q₂) := by
    simp only [generalBackup, expectation, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
    ring
  simp only [Pi.sub_apply, Real.norm_eq_abs, he, abs_mul, abs_of_nonneg hγ]
  exact mul_le_mul_of_nonneg_left
    ((expectation_bound (P y) (V Q₁ - V Q₂)).trans
      (monotone_translation_nonexpansive V hmono hshift Q₁ Q₂)) hγ

theorem general_backup_fixed_point (P : Y → Probability X) (R : Y → ℝ)
    (V : (Y → ℝ) → X → ℝ) (γ : ℝ) (hγ : 0 ≤ γ) (hγ1 : γ < 1)
    (hmono : Monotone V)
    (hshift : ∀ Q c, V (fun y => Q y + c) = fun x => V Q x + c) :
    ∃ Qstar, generalBackup P R V γ Qstar = Qstar ∧
      (∀ Q, generalBackup P R V γ Q = Q → Q = Qstar) ∧
      (∀ Q, Filter.Tendsto (fun n => (generalBackup P R V γ)^[n] Q)
        Filter.atTop (nhds Qstar)) ∧
      (∀ Q n, ‖(generalBackup P R V γ)^[n] Q - Qstar‖ ≤
        γ ^ n * ‖generalBackup P R V γ Q - Q‖ / (1 - γ)) := by
  let B := generalBackup P R V γ
  have hc : ContractingWith ⟨γ, hγ⟩ B := by
    refine ⟨hγ1, LipschitzWith.of_dist_le_mul ?_⟩
    intro Q₁ Q₂
    simpa only [dist_eq_norm, NNReal.coe_mk] using
      general_backup_contraction P R V γ hγ hmono hshift Q₁ Q₂
  refine ⟨hc.fixedPoint B, hc.fixedPoint_isFixedPt, ?_, hc.tendsto_iterate_fixedPoint, ?_⟩
  · intro Q hQ
    exact hc.fixedPoint_unique hQ
  · intro Q n
    have ha := hc.apriori_dist_iterate_fixedPoint_le Q n
    simpa only [B, dist_eq_norm, NNReal.coe_mk, norm_sub_rev, mul_comm] using ha

theorem contraction_iteration_geometric (B : (Y → ℝ) → Y → ℝ)
    (γ : ℝ) (hγ : 0 ≤ γ)
    (hc : ∀ Q₁ Q₂, ‖B Q₁ - B Q₂‖ ≤ γ * ‖Q₁ - Q₂‖)
    (Q Qstar : Y → ℝ) (hfix : B Qstar = Qstar) (n : ℕ) :
    ‖B^[n] Q - Qstar‖ ≤ γ ^ n * ‖Q - Qstar‖ := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply']
    calc
      ‖B (B^[n] Q) - Qstar‖ = ‖B (B^[n] Q) - B Qstar‖ := by rw [hfix]
      _ ≤ γ * ‖B^[n] Q - Qstar‖ := hc _ _
      _ ≤ γ * (γ ^ n * ‖Q - Qstar‖) := mul_le_mul_of_nonneg_left ih hγ
      _ = γ ^ (n + 1) * ‖Q - Qstar‖ := by rw [pow_succ]; ring

theorem contraction_iteration_tail (B : (Y → ℝ) → Y → ℝ)
    (γ : ℝ) (hγ : 0 ≤ γ) (hγ1 : γ < 1)
    (hc : ∀ Q₁ Q₂, ‖B Q₁ - B Q₂‖ ≤ γ * ‖Q₁ - Q₂‖)
    (Q : Y → ℝ) (n m : ℕ) :
    ‖B^[n + m] Q - B^[n] Q‖ ≤ γ ^ n * ‖B Q - Q‖ / (1 - γ) := by
  have hl : LipschitzWith ⟨γ, hγ⟩ B := LipschitzWith.of_dist_le_mul (by
    intro x y
    simpa only [dist_eq_norm, NNReal.coe_mk] using hc x y)
  have hstep (i : ℕ) :
      dist (B^[n + i] Q) (B^[n + (i + 1)] Q) ≤ γ ^ n * ‖B Q - Q‖ * γ ^ i := by
    have hs := hl.dist_iterate_succ_le_geometric Q (n + i)
    simpa only [NNReal.coe_mk, dist_eq_norm, norm_sub_rev, pow_add,
      Nat.add_assoc, mul_assoc, mul_comm, mul_left_comm] using hs
  have hgeo := hasSum_geometric_of_lt_one hγ hγ1
  have hsum : ∑ i ∈ Finset.range m, γ ^ i ≤ (1 - γ)⁻¹ := by
    rw [← hgeo.tsum_eq]
    exact hgeo.summable.sum_le_tsum _ (fun i _ => pow_nonneg hγ i)
  calc
    ‖B^[n + m] Q - B^[n] Q‖ = dist (B^[n] Q) (B^[n + m] Q) := by
      rw [dist_eq_norm, norm_sub_rev]
    _ ≤ ∑ i ∈ Finset.range m, dist (B^[n + i] Q) (B^[n + (i + 1)] Q) := by
      simpa only [Nat.add_zero] using dist_le_range_sum_dist (fun i => B^[n + i] Q) m
    _ ≤ ∑ i ∈ Finset.range m, γ ^ n * ‖B Q - Q‖ * γ ^ i :=
      Finset.sum_le_sum (fun i _ => hstep i)
    _ = γ ^ n * ‖B Q - Q‖ * ∑ i ∈ Finset.range m, γ ^ i := by rw [Finset.mul_sum]
    _ ≤ γ ^ n * ‖B Q - Q‖ * (1 - γ)⁻¹ :=
      mul_le_mul_of_nonneg_left hsum (mul_nonneg (pow_nonneg hγ n) (norm_nonneg _))
    _ = _ := by rw [div_eq_mul_inv]
end GeneralBackup

theorem lawful_probability_input_expands :
    (∀ q ∈ Set.Icc (0 : ℝ) 1, 0 ≤ q ∧ q ≤ 1) ∧
    |((1 : ℝ) * (1 / 2 + 3 / 5) + (1 - 1) * (1 / 2)) -
      (0 * (0 + 3 / 5) + (1 - 0) * 0)| > |(1 : ℝ) - 0| := by
  constructor
  · intro q hq; exact hq
  · norm_num

structure Coupling (p q : Probability X) where
  mass : X → X → ℝ
  nonneg : ∀ x y, 0 ≤ mass x y
  row : ∀ x, ∑ y, mass x y = p.mass x
  col : ∀ y, ∑ x, mass x y = q.mass y

def independentCoupling (p q : Probability X) : Coupling p q where
  mass x y := p.mass x * q.mass y
  nonneg x y := mul_nonneg (p.nonneg x) (q.nonneg y)
  row x := by rw [← Finset.mul_sum, q.total, mul_one]
  col y := by rw [← Finset.sum_mul, p.total, one_mul]

def transportCost {p q : Probability X} (c : Coupling p q) (d : X → X → ℝ) : ℝ :=
  ∑ x, ∑ y, c.mass x y * d x y
def wasserstein (p q : Probability X) (d : X → X → ℝ) : ℝ :=
  sInf (Set.range (fun c : Coupling p q => transportCost c d))

theorem coupling_expectation_difference (p q : Probability X) (c : Coupling p q)
    (v : X → ℝ) :
    expectation p v - expectation q v = ∑ x, ∑ y, c.mass x y * (v x - v y) := by
  simp only [mul_sub, Finset.sum_sub_distrib]
  have h1 : ∑ x, ∑ y, c.mass x y * v x = expectation p v := by
    unfold expectation
    apply Finset.sum_congr rfl
    intro x _
    rw [← Finset.sum_mul, c.row]
  have h2 : ∑ x, ∑ y, c.mass x y * v y = expectation q v := by
    rw [Finset.sum_comm]
    unfold expectation
    apply Finset.sum_congr rfl
    intro y _
    rw [← Finset.sum_mul, c.col]
  rw [h1, h2]

theorem coupling_lipschitz_bound (p q : Probability X) (c : Coupling p q)
    (v : X → ℝ) (d : X → X → ℝ) (L : ℝ) (hL : 0 ≤ L)
    (hv : ∀ x y, |v x - v y| ≤ L * d x y) :
    |expectation p v - expectation q v| ≤ L * transportCost c d := by
  rw [coupling_expectation_difference p q c v]
  calc
    |∑ x, ∑ y, c.mass x y * (v x - v y)|
        ≤ ∑ x, ∑ y, |c.mass x y * (v x - v y)| := by
      exact (Finset.abs_sum_le_sum_abs _ _).trans
        (Finset.sum_le_sum (fun x _ => Finset.abs_sum_le_sum_abs _ _))
    _ ≤ ∑ x, ∑ y, c.mass x y * (L * d x y) := by
      apply Finset.sum_le_sum; intro x _
      apply Finset.sum_le_sum; intro y _
      rw [abs_mul, abs_of_nonneg (c.nonneg x y)]
      exact mul_le_mul_of_nonneg_left (hv x y) (c.nonneg x y)
    _ = L * transportCost c d := by
      simp only [transportCost, Finset.mul_sum]
      apply Finset.sum_congr rfl; intro x _
      apply Finset.sum_congr rfl; intro y _
      ring

theorem wasserstein_lipschitz_bound (p q : Probability X) (v : X → ℝ)
    (d : X → X → ℝ) (hd : ∀ x y, 0 ≤ d x y) (L : ℝ) (hL : 0 ≤ L)
    (hv : ∀ x y, |v x - v y| ≤ L * d x y) :
    |expectation p v - expectation q v| ≤ L * wasserstein p q d := by
  have hn : (Set.range (fun c : Coupling p q => transportCost c d)).Nonempty :=
    ⟨_, ⟨independentCoupling p q, rfl⟩⟩
  by_cases hl : L = 0
  · have h := coupling_lipschitz_bound p q (independentCoupling p q) v d L hL hv
    simpa [hl] using h
  · have hp : 0 < L := lt_of_le_of_ne hL (Ne.symm hl)
    have h := le_csInf hn (a := |expectation p v - expectation q v| / L) (by
      rintro _ ⟨c, rfl⟩
      exact (div_le_iff₀ hp).2 (by
        simpa only [mul_comm] using coupling_lipschitz_bound p q c v d L hL hv))
    exact (div_le_iff₀ hp).1 h |>.trans_eq (mul_comm _ _)

theorem wasserstein_robust_lower_bound (U : Set (Probability X)) (hu : U.Nonempty)
    (p₀ : Probability X) (v : X → ℝ) (d : X → X → ℝ) (hd : ∀ x y, 0 ≤ d x y)
    (L ε : ℝ) (hL : 0 ≤ L) (hv : ∀ x y, |v x - v y| ≤ L * d x y)
    (hU : ∀ p ∈ U, wasserstein p p₀ d ≤ ε) :
    expectation p₀ v - ε * L ≤ sInf (expectations U v) := by
  apply le_csInf (hu.image (fun p => expectation p v))
  rintro _ ⟨p, hp, rfl⟩
  have ha := (abs_le.mp (wasserstein_lipschitz_bound p p₀ v d hd L hL hv)).1
  have hb := mul_le_mul_of_nonneg_left (hU p hp) hL
  nlinarith

theorem layer_composition_lipschitz {E F G : Type*}
    [PseudoMetricSpace E] [PseudoMetricSpace F] [PseudoMetricSpace G]
    (f : F → G) (g : E → F) (Lf Lg : ℝ≥0)
    (hf : LipschitzWith Lf f) (hg : LipschitzWith Lg g) :
    LipschitzWith (Lf * Lg) (f ∘ g) := hf.comp hg

theorem norm_sum_not_network_lipschitz_bound :
    |(5 : ℝ) * (5 * 1) - 5 * (5 * 0)| > (|5| + |5|) * |1 - 0| := by norm_num

def diagonalJoint (x y : Bool) : ℝ := if x = y then 1 / 2 else 0
def oppositeJoint (x y : Bool) : ℝ := if x = y then 0 else 1 / 2

theorem same_marginals_different_joint :
    (∀ x, (∑ y, diagonalJoint x y) = ∑ y, oppositeJoint x y) ∧
    (∀ y, (∑ x, diagonalJoint x y) = ∑ x, oppositeJoint x y) ∧
    diagonalJoint ≠ oppositeJoint := by
  constructor
  · intro x; cases x <;> norm_num [diagonalJoint, oppositeJoint, Fintype.sum_bool]
  constructor
  · intro y; cases y <;> norm_num [diagonalJoint, oppositeJoint, Fintype.sum_bool]
  · intro h
    have he := congrFun (congrFun h false) false
    norm_num [diagonalJoint, oppositeJoint] at he

theorem odds_to_posterior {O δ : ℝ} (hδ : 0 < δ) (hO : 1 / δ ≤ O) :
    1 / (1 + δ) ≤ O / (1 + O) ∧ 1 - δ ≤ 1 / (1 + δ) := by
  have ho : 0 < O := lt_of_lt_of_le (one_div_pos.mpr hδ) hO
  have hd := (div_le_iff₀ hδ).1 hO
  constructor
  · apply (div_le_div_iff₀ (by linarith : 0 < 1 + δ) (by linarith : 0 < 1 + O)).2
    nlinarith
  · apply (le_div_iff₀ (by linarith : 0 < 1 + δ)).2
    nlinarith [sq_nonneg δ]

theorem log_steps_sufficient {L Ω δ : ℝ} {n : ℕ} (hL : 1 < L)
    (hΩ : 0 < Ω) (hδ : 0 < δ)
    (hn : Real.log (Ω / δ) / (2 * Real.log L) ≤ (n : ℝ)) :
    Ω / δ ≤ L ^ (2 * n) := by
  have hlog : 0 < Real.log L := Real.log_pos hL
  have h := (div_le_iff₀ (by positivity : 0 < 2 * Real.log L)).1 hn
  have hl0 : 0 < L := by linarith
  have he := Real.exp_le_exp.mpr h
  rw [Real.exp_log (div_pos hΩ hδ)] at he
  rw [show (n : ℝ) * (2 * Real.log L) = ((2 * n : ℕ) : ℝ) * Real.log L by push_cast; ring,
    Real.exp_nat_mul, Real.exp_log hl0] at he
  exact he

theorem ceil_log_steps_sufficient {L Ω δ : ℝ} {n : ℕ} (hL : 1 < L)
    (hΩ : 0 < Ω) (hδ : 0 < δ)
    (hn : ⌈Real.log (Ω / δ) / (2 * Real.log L)⌉ ≤ (n : ℤ)) :
    Ω / δ ≤ L ^ (2 * n) := by
  apply log_steps_sufficient hL hΩ hδ
  exact_mod_cast Int.ceil_le.mp hn

section Path
variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]

theorem actual_segment_quadratic_bound (pol f : E → F) (x₀ x : E)
    (δ ε L : ℝ) (hδ : 0 ≤ δ) (hε : 0 ≤ ε) (hL : 0 ≤ L)
    (hp : ∀ t ∈ Icc (0 : ℝ) 1, DifferentiableAt ℝ pol (x₀ + t • (x - x₀)))
    (hf : ∀ t ∈ Icc (0 : ℝ) 1, DifferentiableAt ℝ f (x₀ + t • (x - x₀)))
    (hbase : ‖pol x₀ - f x₀‖ ≤ δ)
    (hjac : ‖fderiv ℝ pol x₀ - fderiv ℝ f x₀‖ ≤ ε)
    (hreg : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖(fderiv ℝ pol (x₀ + t • (x - x₀)) - fderiv ℝ f (x₀ + t • (x - x₀))) -
        (fderiv ℝ pol x₀ - fderiv ℝ f x₀)‖ ≤ L * ‖t • (x - x₀)‖) :
    ‖pol x - f x‖ ≤ δ + ε * ‖x - x₀‖ + L / 2 * ‖x - x₀‖ ^ 2 := by
  let d := x - x₀
  let z := fun t : ℝ => x₀ + t • d
  let e := fun t : ℝ => pol (z t) - f (z t)
  let e' := fun t : ℝ => (fderiv ℝ pol (z t) - fderiv ℝ f (z t)) d
  have hz : ∀ t, HasDerivAt z d t := by
    intro t
    simpa [z] using (hasDerivAt_id t).smul_const d |>.const_add x₀
  have he : ∀ t ∈ Icc (0 : ℝ) 1, HasDerivAt e (e' t) t := by
    intro t ht
    have ha := ((hp t ht).hasFDerivAt.comp_hasDerivAt t (hz t)).sub
      ((hf t ht).hasFDerivAt.comp_hasDerivAt t (hz t))
    simpa [e, e', z, d] using ha
  have hbound : ∀ t ∈ Ico (0 : ℝ) 1,
      ‖e' t‖ ≤ ε * ‖d‖ + L * ‖d‖ ^ 2 * t := by
    intro t ht
    have ht' : t ∈ Icc (0 : ℝ) 1 := ⟨ht.1, ht.2.le⟩
    have hj : ‖fderiv ℝ pol (z t) - fderiv ℝ f (z t)‖ ≤ ε + L * (t * ‖d‖) := by
      have ha := norm_le_norm_sub_add
        (fderiv ℝ pol (z t) - fderiv ℝ f (z t)) (fderiv ℝ pol x₀ - fderiv ℝ f x₀)
      have hb := hreg t ht'
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1] at hb
      change _ ≤ L * (t * ‖d‖) at hb
      change _ ≤ _ at ha
      exact ha.trans (by linarith)
    calc
      ‖e' t‖ ≤ ‖fderiv ℝ pol (z t) - fderiv ℝ f (z t)‖ * ‖d‖ :=
        ContinuousLinearMap.le_opNorm _ _
      _ ≤ (ε + L * (t * ‖d‖)) * ‖d‖ :=
        mul_le_mul_of_nonneg_right hj (norm_nonneg d)
      _ = ε * ‖d‖ + L * ‖d‖ ^ 2 * t := by ring
  have hc : ContinuousOn e (Icc (0 : ℝ) 1) :=
    fun t ht => (he t ht).continuousAt.continuousWithinAt
  have hr := CSBAPR.scbapr_ood_quadratic_path e e' δ (ε * ‖d‖) (L * ‖d‖ ^ 2)
    hδ (mul_nonneg hε (norm_nonneg d)) (mul_nonneg hL (sq_nonneg _)) hc
    (by simpa [e, z] using hbase)
    (fun t ht => (he t ⟨ht.1, ht.2.le⟩).hasDerivWithinAt) hbound
  simpa [e, z, d, div_mul_eq_mul_div] using hr

theorem segment_bound_from_two_derivative_regularity (pol f : E → F) (x₀ x : E)
    (δ ε Lp Lf : ℝ) (hδ : 0 ≤ δ) (hε : 0 ≤ ε) (hLp : 0 ≤ Lp) (hLf : 0 ≤ Lf)
    (hp : ∀ t ∈ Icc (0 : ℝ) 1, DifferentiableAt ℝ pol (x₀ + t • (x - x₀)))
    (hf : ∀ t ∈ Icc (0 : ℝ) 1, DifferentiableAt ℝ f (x₀ + t • (x - x₀)))
    (hbase : ‖pol x₀ - f x₀‖ ≤ δ)
    (hjac : ‖fderiv ℝ pol x₀ - fderiv ℝ f x₀‖ ≤ ε)
    (hregp : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖fderiv ℝ pol (x₀ + t • (x - x₀)) - fderiv ℝ pol x₀‖ ≤ Lp * ‖t • (x - x₀)‖)
    (hregf : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖fderiv ℝ f (x₀ + t • (x - x₀)) - fderiv ℝ f x₀‖ ≤ Lf * ‖t • (x - x₀)‖) :
    ‖pol x - f x‖ ≤ δ + ε * ‖x - x₀‖ + (Lp + Lf) / 2 * ‖x - x₀‖ ^ 2 := by
  apply actual_segment_quadratic_bound pol f x₀ x δ ε (Lp + Lf)
    hδ hε (add_nonneg hLp hLf) hp hf hbase hjac
  intro t ht
  have he : (fderiv ℝ pol (x₀ + t • (x - x₀)) - fderiv ℝ f (x₀ + t • (x - x₀))) -
      (fderiv ℝ pol x₀ - fderiv ℝ f x₀) =
      (fderiv ℝ pol (x₀ + t • (x - x₀)) - fderiv ℝ pol x₀) -
      (fderiv ℝ f (x₀ + t • (x - x₀)) - fderiv ℝ f x₀) := by abel
  rw [he]
  exact (norm_sub_le _ _).trans (by nlinarith [hregp t ht, hregf t ht])
end Path

section Composition
variable {E F G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F] [NormedAddCommGroup G] [NormedSpace ℝ G]

theorem composed_fderiv_lipschitz (h : F → G) (g : E → F)
    (Lh Kg Bg Bh Lg : ℝ) (hh : Differentiable ℝ h) (hg : Differentiable ℝ g)
    (hLh : 0 ≤ Lh) (hKg : 0 ≤ Kg) (hBg : 0 ≤ Bg) (hBh : 0 ≤ Bh) (hLg : 0 ≤ Lg)
    (hhlip : ∀ u v, ‖fderiv ℝ h u - fderiv ℝ h v‖ ≤ Lh * ‖u - v‖)
    (hglip : ∀ x y, ‖g x - g y‖ ≤ Kg * ‖x - y‖)
    (hgbound : ∀ x, ‖fderiv ℝ g x‖ ≤ Bg)
    (hhbound : ∀ u, ‖fderiv ℝ h u‖ ≤ Bh)
    (hgdlip : ∀ x y, ‖fderiv ℝ g x - fderiv ℝ g y‖ ≤ Lg * ‖x - y‖)
    (x y : E) :
    ‖fderiv ℝ (h ∘ g) x - fderiv ℝ (h ∘ g) y‖ ≤
      (Lh * Kg * Bg + Bh * Lg) * ‖x - y‖ := by
  rw [fderiv_comp x (hh _) (hg _), fderiv_comp y (hh _) (hg _)]
  have he : (fderiv ℝ h (g x)).comp (fderiv ℝ g x) -
      (fderiv ℝ h (g y)).comp (fderiv ℝ g y) =
      (fderiv ℝ h (g x) - fderiv ℝ h (g y)).comp (fderiv ℝ g x) +
      (fderiv ℝ h (g y)).comp (fderiv ℝ g x - fderiv ℝ g y) := by
    rw [ContinuousLinearMap.sub_comp, ContinuousLinearMap.comp_sub]
    abel
  rw [he]
  calc
    _ ≤ ‖(fderiv ℝ h (g x) - fderiv ℝ h (g y)).comp (fderiv ℝ g x)‖ +
        ‖(fderiv ℝ h (g y)).comp (fderiv ℝ g x - fderiv ℝ g y)‖ := norm_add_le _ _
    _ ≤ ‖fderiv ℝ h (g x) - fderiv ℝ h (g y)‖ * ‖fderiv ℝ g x‖ +
        ‖fderiv ℝ h (g y)‖ * ‖fderiv ℝ g x - fderiv ℝ g y‖ :=
      add_le_add (ContinuousLinearMap.opNorm_comp_le _ _)
        (ContinuousLinearMap.opNorm_comp_le _ _)
    _ ≤ (Lh * (Kg * ‖x - y‖)) * Bg + Bh * (Lg * ‖x - y‖) := by
      apply add_le_add
      · apply mul_le_mul ((hhlip _ _).trans
          (mul_le_mul_of_nonneg_left (hglip x y) hLh)) (hgbound x)
          (norm_nonneg _) (by positivity)
      · exact mul_le_mul (hhbound _) (hgdlip x y) (norm_nonneg _) (by positivity)
    _ = _ := by ring
end Composition

theorem quadratic_linearization_sharp (c a x : ℝ) :
    (c * a ^ 2 - (c * a ^ 2 + 2 * c * a * (a - a)) = 0) ∧
    deriv (fun x : ℝ => c * x ^ 2) a =
      deriv (fun x : ℝ => c * a ^ 2 + 2 * c * a * (x - a)) a ∧
    (∀ u v : ℝ, |deriv (fun x : ℝ => c * x ^ 2) u -
        deriv (fun x : ℝ => c * x ^ 2) v| ≤ 2 * |c| * |u - v|) ∧
    |c * x ^ 2 - (c * a ^ 2 + 2 * c * a * (x - a))| =
      (2 * |c|) / 2 * |x - a| ^ 2 := by
  have hp : deriv (fun x : ℝ => c * x ^ 2) a = 2 * c * a := by
    rw [((hasDerivAt_pow 2 a).const_mul c).deriv]
    ring
  have hf : deriv (fun x : ℝ => c * a ^ 2 + 2 * c * a * (x - a)) a = 2 * c * a := by
    have h := ((hasDerivAt_id a).sub_const a).const_mul (2 * c * a)
    simpa only [mul_one] using (h.const_add (c * a ^ 2)).deriv
  refine ⟨by ring, hp.trans hf.symm, CSBAPR.nmu_has_bounded_arithmetic_bias c, ?_⟩
  rw [show c * x ^ 2 - (c * a ^ 2 + 2 * c * a * (x - a)) = c * (x - a) ^ 2 by ring,
    abs_mul, abs_pow]
  ring

theorem quadratic_half_coefficient_cannot_decrease (k : ℝ) (hk : k < 1 / 2) :
    ¬ (∀ c a x : ℝ, |c * x ^ 2 - (c * a ^ 2 + 2 * c * a * (x - a))| ≤
      k * (2 * |c|) * |x - a| ^ 2) := by
  intro h
  have hw := h 1 0 1
  norm_num at hw
  linarith

def twoSlope (a b x : ℝ) : ℝ := if x ≤ 0 then a * x else b * x

theorem tanh_preserves_one_sided_derivative_jump (a b : ℝ) (hab : a ≠ b) :
    HasDerivWithinAt (fun x => Real.tanh (twoSlope a b x)) a (Iic 0) 0 ∧
    HasDerivWithinAt (fun x => Real.tanh (twoSlope a b x)) b (Ici 0) 0 ∧ a ≠ b := by
  have ht : HasDerivAt Real.tanh 1 0 := by
    rw [show Real.tanh = (fun x => Real.sinh x / Real.cosh x) from
      funext Real.tanh_eq_sinh_div_cosh]
    simpa [Pi.div_apply] using
      (Real.hasDerivAt_sinh 0).div (Real.hasDerivAt_cosh 0) (by simp)
  have hlin (slope : ℝ) : HasDerivAt (fun x => Real.tanh (slope * x)) slope 0 := by
    have hm : HasDerivAt (fun x : ℝ => slope * x) slope 0 := by
      simpa using (hasDerivAt_id (0 : ℝ)).const_mul slope
    simpa using (show HasDerivAt Real.tanh 1 (slope * 0) from by simpa using ht).comp 0 hm
  refine ⟨?_, ?_, hab⟩
  · apply (hlin a).hasDerivWithinAt.congr_of_mem
    · intro x hx
      simp [twoSlope, mem_Iic.mp hx]
    · simp
  · apply (hlin b).hasDerivWithinAt.congr_of_mem
    · intro x hx
      by_cases hx0 : x = 0
      · simp [hx0, twoSlope]
      · have hpos : 0 < x := lt_of_le_of_ne hx (Ne.symm hx0)
        simp [twoSlope, not_le.mpr hpos]
    · simp

theorem composed_deriv_lipschitz (h g : ℝ → ℝ) (Lh Kg Bg Bh Lg : ℝ)
    (hh : Differentiable ℝ h) (hg : Differentiable ℝ g)
    (hLh : 0 ≤ Lh) (hKg : 0 ≤ Kg) (hBg : 0 ≤ Bg) (hBh : 0 ≤ Bh) (hLg : 0 ≤ Lg)
    (hhlip : ∀ u v, |deriv h u - deriv h v| ≤ Lh * |u - v|)
    (hglip : ∀ x y, |g x - g y| ≤ Kg * |x - y|)
    (hgbound : ∀ x, |deriv g x| ≤ Bg)
    (hhbound : ∀ u, |deriv h u| ≤ Bh)
    (hgdlip : ∀ x y, |deriv g x - deriv g y| ≤ Lg * |x - y|)
    (x y : ℝ) : |deriv (h ∘ g) x - deriv (h ∘ g) y| ≤
      (Lh * Kg * Bg + Bh * Lg) * |x - y| := by
  rw [deriv_comp x (hh _) (hg _), deriv_comp y (hh _) (hg _)]
  have he : deriv h (g x) * deriv g x - deriv h (g y) * deriv g y =
      (deriv h (g x) - deriv h (g y)) * deriv g x +
      deriv h (g y) * (deriv g x - deriv g y) := by ring
  rw [he]
  calc
    _ ≤ |(deriv h (g x) - deriv h (g y)) * deriv g x| +
        |deriv h (g y) * (deriv g x - deriv g y)| := abs_add_le _ _
    _ = |deriv h (g x) - deriv h (g y)| * |deriv g x| +
        |deriv h (g y)| * |deriv g x - deriv g y| := by rw [abs_mul, abs_mul]
    _ ≤ (Lh * (Kg * |x - y|)) * Bg + Bh * (Lg * |x - y|) := by
      apply add_le_add
      · apply mul_le_mul ((hhlip _ _).trans
          (mul_le_mul_of_nonneg_left (hglip x y) hLh)) (hgbound x)
          (abs_nonneg _) (by positivity)
      · exact mul_le_mul (hhbound _) (hgdlip x y) (abs_nonneg _) (by positivity)
    _ = _ := by ring

section HistoryKL
variable {H Y : Type} [Fintype H] [Fintype Y]

def History (H Y : Type) : ℕ → Type
  | 0 => H
  | n + 1 => History H Y n × Y

instance historyFintype (n : ℕ) : Fintype (History H Y n) := by
  induction n with
  | zero => exact inferInstanceAs (Fintype H)
  | succ n ih =>
    letI := ih
    change Fintype (History H Y n × Y)
    infer_instance

def historyMass (p₀ : H → ℝ) (κ : ∀ n, History H Y n → Y → ℝ) :
    (n : ℕ) → History H Y n → ℝ
  | 0 => p₀
  | n + 1 => fun z => historyMass p₀ κ n z.1 * κ n z.1 z.2

theorem history_mass_positive (p₀ : H → ℝ) (κ : ∀ n, History H Y n → Y → ℝ)
    (hp : ∀ h, 0 < p₀ h) (hκ : ∀ n h y, 0 < κ n h y)
    (n : ℕ) (h : History H Y n) : 0 < historyMass p₀ κ n h := by
  induction n with
  | zero => exact hp h
  | succ n ih => exact mul_pos (ih h.1) (hκ n h.1 h.2)

theorem history_mass_normalized (p₀ : H → ℝ) (κ : ∀ n, History H Y n → Y → ℝ)
    (hp : ∑ h, p₀ h = 1) (hκ : ∀ n h, ∑ y, κ n h y = 1) (n : ℕ) :
    ∑ h, historyMass p₀ κ n h = 1 := by
  induction n with
  | zero => exact hp
  | succ n ih =>
    change (∑ z : History H Y n × Y, historyMass p₀ κ n z.1 * κ n z.1 z.2) = 1
    rw [Fintype.sum_prod_type]
    simp only [← Finset.mul_sum, hκ, mul_one]
    exact ih

def historyKL (p₀ : H → ℝ) (κ ν : ∀ n, History H Y n → Y → ℝ) (n : ℕ) : ℝ :=
  ∑ h, historyMass p₀ κ n h * Real.log (historyMass p₀ κ n h / historyMass p₀ ν n h)

def conditionalKL (p₀ : H → ℝ) (κ ν : ∀ n, History H Y n → Y → ℝ) (n : ℕ) : ℝ :=
  ∑ h, historyMass p₀ κ n h * ∑ y, κ n h y * Real.log (κ n h y / ν n h y)

theorem history_kl_step (p₀ : H → ℝ) (κ ν : ∀ n, History H Y n → Y → ℝ)
    (hp : ∀ h, 0 < p₀ h) (hκ : ∀ n h y, 0 < κ n h y)
    (hν : ∀ n h y, 0 < ν n h y)
    (hκnorm : ∀ n h, ∑ y, κ n h y = 1) (n : ℕ) :
    historyKL p₀ κ ν (n + 1) = historyKL p₀ κ ν n + conditionalKL p₀ κ ν n := by
  unfold historyKL conditionalKL
  change (∑ z : History H Y n × Y,
    historyMass p₀ κ n z.1 * κ n z.1 z.2 *
      Real.log ((historyMass p₀ κ n z.1 * κ n z.1 z.2) /
        (historyMass p₀ ν n z.1 * ν n z.1 z.2))) = _
  rw [Fintype.sum_prod_type]
  exact BAPR.ActionCompensation.finite_conditional_kl_chain_step
    (historyMass p₀ κ n) (historyMass p₀ ν n) (κ n) (ν n)
    (history_mass_positive p₀ κ hp hκ n) (history_mass_positive p₀ ν hp hν n)
    (hκ n) (hν n) (hκnorm n)

theorem finite_trajectory_kl_chain (p₀ : H → ℝ) (κ ν : ∀ n, History H Y n → Y → ℝ)
    (hp : ∀ h, 0 < p₀ h) (hκ : ∀ n h y, 0 < κ n h y)
    (hν : ∀ n h y, 0 < ν n h y)
    (hκnorm : ∀ n h, ∑ y, κ n h y = 1) (n : ℕ) :
    historyKL p₀ κ ν n = ∑ i ∈ Finset.range n, conditionalKL p₀ κ ν i := by
  induction n with
  | zero =>
    simp only [historyKL, historyMass, Finset.range_zero, Finset.sum_empty]
    apply Finset.sum_eq_zero
    intro h _
    rw [div_self (hp h).ne', Real.log_one, mul_zero]
  | succ n ih =>
    rw [history_kl_step p₀ κ ν hp hκ hν hκnorm n, ih, Finset.sum_range_succ]
end HistoryKL

section FiniteRisk
variable [Nonempty X]

def upperMass (p : Probability X) (c : X → ℝ) (η : ℝ) : ℝ :=
  ∑ x, if η < c x then p.mass x else 0
def atLeastMass (p : Probability X) (c : X → ℝ) (η : ℝ) : ℝ :=
  ∑ x, if η ≤ c x then p.mass x else 0
def atomMass (p : Probability X) (c : X → ℝ) (η : ℝ) : ℝ :=
  ∑ x, if c x = η then p.mass x else 0

theorem finite_quantile_exists (p : Probability X) (c : X → ℝ) (β : ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) :
    ∃ η, upperMass p c η ≤ β ∧ β ≤ atLeastMass p c η := by
  classical
  obtain ⟨i, _, hi⟩ := Finset.exists_min_image Finset.univ c Finset.univ_nonempty
  have hmin : atLeastMass p c (c i) = 1 := by
    simp only [atLeastMass, if_pos (hi _ (Finset.mem_univ _)), p.total]
  let good := Finset.univ.filter (fun x => β ≤ atLeastMass p c (c x))
  have hgood : good.Nonempty := ⟨i, by simp [good, hmin, hβ1]⟩
  obtain ⟨j, hj, hjmax⟩ := Finset.exists_max_image good c hgood
  refine ⟨c j, ?_, (Finset.mem_filter.mp hj).2⟩
  by_contra h
  have hab : β < upperMass p c (c j) := lt_of_not_ge h
  let above := Finset.univ.filter (fun x => c j < c x)
  have hne : above.Nonempty := by
    by_contra he
    have hz : upperMass p c (c j) = 0 := by
      apply Finset.sum_eq_zero
      intro x _
      have hnx : ¬ c j < c x := by
        intro hh
        exact he ⟨x, by simp [above, hh]⟩
      simp [hnx]
    linarith
  obtain ⟨k, hk, hkmin⟩ := Finset.exists_min_image above c hne
  have hmass : atLeastMass p c (c k) = upperMass p c (c j) := by
    unfold atLeastMass upperMass
    apply Finset.sum_congr rfl
    intro x _
    have heq : c k ≤ c x ↔ c j < c x := by
      constructor
      · intro hh
        exact lt_of_lt_of_le (Finset.mem_filter.mp hk).2 hh
      · intro hh
        exact hkmin x (by simp [above, hh])
    simp only [heq]
  have hkg : k ∈ good := by simp [good, hmass, hab.le]
  have hh := hjmax k hkg
  have hkgt := (Finset.mem_filter.mp hk).2
  linarith

theorem tail_atom_decomposition (p : Probability X) (c : X → ℝ) (η : ℝ) :
    atLeastMass p c η = upperMass p c η + atomMass p c η := by
  unfold atLeastMass upperMass atomMass
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro x _
  by_cases hlt : η < c x
  · simp [hlt, hlt.le, ne_of_gt hlt]
  · by_cases heq : c x = η
    · simp [heq]
    · have hle : ¬ η ≤ c x := fun h => heq (le_antisymm (le_of_not_gt hlt) h)
      simp [hlt, heq, hle]

def riskObjective (p : Probability X) (c : X → ℝ) (β η : ℝ) : ℝ :=
  η + (∑ x, p.mass x * max (c x - η) 0) / β
def feasibleDensity (p : Probability X) (β : ℝ) (ζ : X → ℝ) : Prop :=
  (∀ x, 0 ≤ ζ x ∧ ζ x ≤ 1 / β) ∧ expectation p ζ = 1

theorem cvar_weak_duality (p : Probability X) (c ζ : X → ℝ) (β η : ℝ)
    (hβ : 0 < β) (hz : feasibleDensity p β ζ) :
    expectation p (fun x => ζ x * c x) ≤ riskObjective p c β η := by
  have hpoint : ∀ x, ζ x * (c x - η) ≤ max (c x - η) 0 / β := by
    intro x
    by_cases h : 0 ≤ c x - η
    · rw [max_eq_left h]
      have he := mul_le_mul_of_nonneg_right (hz.1 x).2 h
      simpa only [one_div_mul_eq_div] using he
    · rw [max_eq_right (le_of_not_ge h), zero_div]
      exact mul_nonpos_of_nonneg_of_nonpos (hz.1 x).1 (le_of_not_ge h)
  have hsum := Finset.sum_le_sum (fun x (_ : x ∈ Finset.univ) =>
    mul_le_mul_of_nonneg_left (hpoint x) (p.nonneg x))
  have hsplit : expectation p (fun x => ζ x * c x) =
      η + ∑ x, p.mass x * (ζ x * (c x - η)) := by
    have hm := congrArg (fun r => r * η) hz.2
    simp only [expectation, Finset.sum_mul] at hm
    calc
      expectation p (fun x => ζ x * c x) =
          (∑ x, p.mass x * ζ x) * η + ∑ x, p.mass x * (ζ x * (c x - η)) := by
        rw [Finset.sum_mul, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro x _; ring
      _ = _ := by rw [show ∑ x, p.mass x * ζ x = 1 from hz.2, one_mul]
  rw [hsplit]
  unfold riskObjective
  apply add_le_add_right
  simpa only [mul_div_assoc, Finset.sum_div] using hsum

theorem finite_cvar_attainment (p : Probability X) (c : X → ℝ) (β : ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) :
    ∃ η ζ, feasibleDensity p β ζ ∧
      expectation p (fun x => ζ x * c x) = riskObjective p c β η := by
  classical
  obtain ⟨η, hab, hba⟩ := finite_quantile_exists p c β hβ hβ1
  let a := upperMass p c η
  let b := atomMass p c η
  have hb : 0 ≤ b := by
    apply Finset.sum_nonneg
    intro x _
    split_ifs <;> first | exact p.nonneg x | rfl
  have ha : 0 ≤ β - a := sub_nonneg.mpr hab
  have habb : β - a ≤ b := by
    rw [tail_atom_decomposition] at hba
    dsimp [a, b] at *
    linarith
  let m := if b = 0 then 0 else (β - a) / b
  have hm : 0 ≤ m ∧ m ≤ 1 ∧ a + m * b = β := by
    by_cases hb0 : b = 0
    · simp only [m, if_pos hb0, zero_mul, add_zero]
      constructor
      · norm_num
      constructor
      · norm_num
      · linarith
    · have hbpos : 0 < b := lt_of_le_of_ne hb (Ne.symm hb0)
      simp only [m, if_neg hb0]
      exact ⟨div_nonneg ha hb, (div_le_one hbpos).2 habb,
        by rw [div_mul_cancel₀ _ hb0]; ring⟩
  let ζ := fun x => if η < c x then 1 / β else if c x = η then m / β else 0
  have hzbox : ∀ x, 0 ≤ ζ x ∧ ζ x ≤ 1 / β := by
    intro x
    dsimp [ζ]
    split_ifs
    · exact ⟨(one_div_pos.mpr hβ).le, le_rfl⟩
    · exact ⟨div_nonneg hm.1 hβ.le, div_le_div_of_nonneg_right hm.2.1 hβ.le⟩
    · exact ⟨le_rfl, (one_div_pos.mpr hβ).le⟩
  have hmass : expectation p ζ = (a + m * b) / β := by
    unfold expectation
    dsimp [a, b, upperMass, atomMass]
    rw [add_div, Finset.sum_div, Finset.mul_sum, Finset.sum_div, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro x _
    dsimp [ζ]
    by_cases hlt : η < c x
    · simp [hlt, ne_of_gt hlt, div_eq_mul_inv]
    · by_cases heq : c x = η
      · simp [hlt, heq]; ring
      · simp [hlt, heq]
  have hz : feasibleDensity p β ζ := ⟨hzbox, by rw [hmass, hm.2.2, div_self hβ.ne']⟩
  have hpoint : ∀ x, ζ x * (c x - η) = max (c x - η) 0 / β := by
    intro x
    by_cases hlt : η < c x
    · simp [ζ, hlt, max_eq_left (sub_pos.mpr hlt).le, div_eq_mul_inv, mul_comm]
    · by_cases heq : c x = η
      · simp [heq]
      · simp [ζ, hlt, heq, max_eq_right (sub_nonpos.mpr (le_of_not_gt hlt))]
  refine ⟨η, ζ, hz, ?_⟩
  have hsum : (∑ x, p.mass x * (ζ x * (c x - η))) =
      (∑ x, p.mass x * max (c x - η) 0) / β := by
    simp only [hpoint, mul_div_assoc, Finset.sum_div]
  calc
    expectation p (fun x => ζ x * c x) =
        (expectation p ζ) * η + ∑ x, p.mass x * (ζ x * (c x - η)) := by
      unfold expectation
      rw [Finset.sum_mul, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl; intro x _; ring
    _ = riskObjective p c β η := by rw [hz.2, one_mul, hsum]; rfl

def primalRiskValues (p : Probability X) (c : X → ℝ) (β : ℝ) : Set ℝ :=
  Set.range (riskObjective p c β)
def dualRiskValues (p : Probability X) (c : X → ℝ) (β : ℝ) : Set ℝ :=
  {r | ∃ ζ, feasibleDensity p β ζ ∧ r = expectation p (fun x => ζ x * c x)}

theorem finite_cvar_risk_envelope (p : Probability X) (c : X → ℝ) (β : ℝ)
    (hβ : 0 < β) (hβ1 : β ≤ 1) :
    sInf (primalRiskValues p c β) = sSup (dualRiskValues p c β) := by
  obtain ⟨η, ζ, hz, he⟩ := finite_cvar_attainment p c β hβ hβ1
  let r := riskObjective p c β η
  have hrp : r ∈ primalRiskValues p c β := ⟨η, rfl⟩
  have hrd : r ∈ dualRiskValues p c β := ⟨ζ, hz, he.symm⟩
  have lower : ∀ v ∈ primalRiskValues p c β, r ≤ v := by
    rintro v ⟨t, rfl⟩
    dsimp [r]
    rw [← he]
    exact cvar_weak_duality p c ζ β t hβ hz
  have upper : ∀ v ∈ dualRiskValues p c β, v ≤ r := by
    rintro v ⟨z, hz', rfl⟩
    exact cvar_weak_duality p c z β η hβ hz'
  have hip : sInf (primalRiskValues p c β) = r :=
    le_antisymm (csInf_le ⟨r, lower⟩ hrp) (le_csInf ⟨r, hrp⟩ lower)
  have his : sSup (dualRiskValues p c β) = r :=
    le_antisymm (csSup_le ⟨r, hrd⟩ upper) (le_csSup ⟨r, upper⟩ hrd)
  rw [hip, his]
end FiniteRisk

#print axioms expectation_bound
#print axioms support_neg_is_inf
#print axioms robust_regularization_identity
#print axioms monotone_translation_nonexpansive
#print axioms general_backup_monotone
#print axioms general_backup_shift
#print axioms mixtureProbability
#print axioms mixture_backup_identity
#print axioms general_backup_contraction
#print axioms general_backup_fixed_point
#print axioms contraction_iteration_geometric
#print axioms contraction_iteration_tail
#print axioms lawful_probability_input_expands
#print axioms coupling_expectation_difference
#print axioms independentCoupling
#print axioms wasserstein_lipschitz_bound
#print axioms wasserstein_robust_lower_bound
#print axioms layer_composition_lipschitz
#print axioms norm_sum_not_network_lipschitz_bound
#print axioms same_marginals_different_joint
#print axioms odds_to_posterior
#print axioms log_steps_sufficient
#print axioms ceil_log_steps_sufficient
#print axioms actual_segment_quadratic_bound
#print axioms segment_bound_from_two_derivative_regularity
#print axioms composed_fderiv_lipschitz
#print axioms composed_deriv_lipschitz
#print axioms quadratic_linearization_sharp
#print axioms quadratic_half_coefficient_cannot_decrease
#print axioms tanh_preserves_one_sided_derivative_jump
#print axioms history_mass_positive
#print axioms history_mass_normalized
#print axioms finite_trajectory_kl_chain
#print axioms finite_quantile_exists
#print axioms cvar_weak_duality
#print axioms finite_cvar_attainment
#print axioms finite_cvar_risk_envelope

end ThesisExtensions
