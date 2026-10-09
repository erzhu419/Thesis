import Mathlib

noncomputable section
open MeasureTheory ProbabilityTheory Set Filter Function
open scoped Topology

namespace ThesisCVaR

theorem cdf_quantile_exists (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (ρ : ℝ) (hρ : 0 < ρ) (hρ1 : ρ < 1) :
    ∃ η, ρ ≤ cdf ν η ∧ Function.leftLim (cdf ν) η ≤ ρ := by
  let S : Set ℝ := {t | ρ ≤ cdf ν t}
  obtain ⟨u, hu⟩ := ((tendsto_cdf_atTop ν).eventually (eventually_gt_nhds hρ1)).exists
  obtain ⟨l, hl⟩ := ((tendsto_cdf_atBot ν).eventually (eventually_lt_nhds hρ)).exists
  have hne : S.Nonempty := ⟨u, hu.le⟩
  have hb : BddBelow S := by
    refine ⟨l, ?_⟩
    intro t ht
    by_contra h
    have ht' := (cdf ν).mono (le_of_not_ge h)
    change ρ ≤ cdf ν t at ht
    linarith
  let η := sInf S
  have hright : ∀ t, η < t → ρ ≤ cdf ν t := by
    intro t ht
    obtain ⟨s, hs, hst⟩ := exists_lt_of_csInf_lt hne ht
    exact hs.trans ((cdf ν).mono hst.le)
  have hleft : ∀ t, t < η → cdf ν t ≤ ρ := by
    intro t ht
    by_contra h
    have hm : t ∈ S := le_of_not_ge h
    exact (not_le.mpr ht) (csInf_le hb hm)
  refine ⟨η, ?_, ?_⟩
  · apply ge_of_tendsto ((cdf ν).right_continuous η |>.mono Ioi_subset_Ici_self).tendsto
    filter_upwards [self_mem_nhdsWithin] with t ht
    exact hright t ht
  · apply le_of_tendsto ((cdf ν).mono.tendsto_leftLim η)
    filter_upwards [self_mem_nhdsWithin] with t ht
    exact hleft t ht

theorem distribution_quantile_exists (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (β : ℝ) (hβ : 0 < β) (hβ1 : β < 1) :
    ∃ η, ν.real (Ioi η) ≤ β ∧ β ≤ ν.real (Ici η) := by
  obtain ⟨η, hr, hl⟩ := cdf_quantile_exists ν (1 - β) (by linarith) (by linarith)
  have htail : ν.real (Ioi η) = 1 - cdf ν η := by
    rw [← compl_Iic, measureReal_compl measurableSet_Iic, probReal_univ,
      ← cdf_eq_real]
  have hl0 : 0 ≤ Function.leftLim (cdf ν) η := by
    apply ge_of_tendsto ((cdf ν).mono.tendsto_leftLim η)
    exact Eventually.of_forall (cdf_nonneg ν)
  have hl1 : Function.leftLim (cdf ν) η ≤ 1 := hl.trans (by linarith)
  have htail' : ν.real (Ici η) = 1 - Function.leftLim (cdf ν) η := by
    have hc : (cdf ν).measure (Ici η) = ν (Ici η) := by rw [measure_cdf]
    rw [measureReal_def, ← hc,
      StieltjesFunction.measure_Ici _ (tendsto_cdf_atTop ν),
      ENNReal.toReal_ofReal (sub_nonneg.mpr hl1)]
  exact ⟨η, by rw [htail]; linarith, by rw [htail']; linarith⟩

variable {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]

def objective (C : Ω → ℝ) (β η : ℝ) : ℝ :=
  η + (∫ x, max (C x - η) 0 ∂μ) / β

def feasible (β : ℝ) (ζ : Ω → ℝ) : Prop :=
  Measurable ζ ∧ (∀ x, 0 ≤ ζ x ∧ ζ x ≤ 1 / β) ∧ (∫ x, ζ x ∂μ) = 1

theorem quantile_exists (C : Ω → ℝ) (hC : Measurable C)
    (β : ℝ) (hβ : 0 < β) (hβ1 : β < 1) :
    ∃ η, μ.real {x | η < C x} ≤ β ∧ β ≤ μ.real {x | η ≤ C x} := by
  let ν := μ.map C
  haveI : IsProbabilityMeasure ν := Measure.isProbabilityMeasure_map hC.aemeasurable
  obtain ⟨η, ha, hb⟩ := distribution_quantile_exists ν β hβ hβ1
  refine ⟨η, ?_, ?_⟩
  · simpa only [ν, map_measureReal_apply hC measurableSet_Ioi] using ha
  · simpa only [ν, map_measureReal_apply hC measurableSet_Ici] using hb

theorem feasible_integrable (β : ℝ) (ζ : Ω → ℝ) (hz : feasible μ β ζ) :
    Integrable ζ μ := by
  apply (integrable_const (1 / β)).mono' hz.1.aestronglyMeasurable
  exact Eventually.of_forall (fun x => by
    simpa only [Real.norm_eq_abs, abs_of_nonneg (hz.2.1 x).1] using (hz.2.1 x).2)

theorem positive_part_integrable (C : Ω → ℝ) (hC : Measurable C)
    (hCi : Integrable C μ) (η : ℝ) : Integrable (fun x => max (C x - η) 0) μ := by
  have hm : Measurable (fun x => max (C x - η) 0) := by fun_prop
  apply (hCi.sub (integrable_const η)).abs.mono' hm.aestronglyMeasurable
  exact Eventually.of_forall (fun x => by
    rw [Real.norm_eq_abs, abs_of_nonneg (le_max_right _ _)]
    exact max_le (le_abs_self _) (abs_nonneg _))

theorem density_cost_integrable (C ζ : Ω → ℝ) (hCi : Integrable C μ)
    (β : ℝ) (hz : feasible μ β ζ) : Integrable (fun x => ζ x * C x) μ := by
  apply hCi.bdd_mul hz.1.aestronglyMeasurable
  exact Eventually.of_forall (fun x => by
    simpa only [Real.norm_eq_abs, abs_of_nonneg (hz.2.1 x).1] using (hz.2.1 x).2)

theorem weak_duality (C ζ : Ω → ℝ) (hC : Measurable C) (hCi : Integrable C μ)
    (β η : ℝ) (hβ : 0 < β) (hz : feasible μ β ζ) :
    (∫ x, ζ x * C x ∂μ) ≤ objective μ C β η := by
  have hprod := density_cost_integrable μ (fun x => C x - η) ζ
    (hCi.sub (integrable_const η)) β hz
  have hzint := feasible_integrable μ β ζ hz
  have hmax := positive_part_integrable μ C hC hCi η
  have hrhs : Integrable (fun x => max (C x - η) 0 / β) μ := by
    simpa [div_eq_mul_inv] using hmax.mul_const β⁻¹
  have hpoint : ∀ x, ζ x * (C x - η) ≤ max (C x - η) 0 / β := by
    intro x
    by_cases h : 0 ≤ C x - η
    · rw [max_eq_left h]
      simpa only [one_div_mul_eq_div] using
        mul_le_mul_of_nonneg_right (hz.2.1 x).2 h
    · rw [max_eq_right (le_of_not_ge h), zero_div]
      exact mul_nonpos_of_nonneg_of_nonpos (hz.2.1 x).1 (le_of_not_ge h)
  have hsplit : (∫ x, ζ x * C x ∂μ) = η + ∫ x, ζ x * (C x - η) ∂μ := by
    rw [show (fun x => ζ x * C x) = (fun x => η * ζ x + ζ x * (C x - η)) by
      funext x; ring, integral_add (hzint.const_mul η) hprod, integral_const_mul,
      hz.2.2, mul_one]
  rw [hsplit]
  unfold objective
  apply add_le_add_right
  rw [← integral_div]
  exact integral_mono hprod hrhs hpoint

theorem tail_atom_decomposition (C : Ω → ℝ) (hC : Measurable C) (η : ℝ) :
    μ.real {x | η ≤ C x} = μ.real {x | η < C x} + μ.real {x | C x = η} := by
  have he : {x | η ≤ C x} = {x | η < C x} ∪ {x | C x = η} := by
    ext x; simp only [mem_setOf_eq, mem_union]; constructor
    · intro hx; rcases lt_or_eq_of_le hx with h | h
      · exact Or.inl h
      · exact Or.inr h.symm
    · rintro (h | h)
      · exact h.le
      · exact h.symm.le
  rw [he]
  have hd : Disjoint {x | η < C x} {x | C x = η} := by
    rw [disjoint_left]
    intro x hx heq
    exact (ne_of_gt hx) heq
  exact measureReal_union hd (hC (measurableSet_singleton η)) (by finiteness) (by finiteness)

theorem attainment (C : Ω → ℝ) (hC : Measurable C) (hCi : Integrable C μ)
    (β : ℝ) (hβ : 0 < β) (hβ1 : β < 1) :
    ∃ η ζ, feasible μ β ζ ∧ (∫ x, ζ x * C x ∂μ) = objective μ C β η := by
  classical
  obtain ⟨η, hu, hl⟩ := quantile_exists μ C hC β hβ hβ1
  let a := μ.real {x | η < C x}
  let b := μ.real {x | C x = η}
  have hb : 0 ≤ b := measureReal_nonneg
  have ha : 0 ≤ β - a := sub_nonneg.mpr hu
  have hab : β - a ≤ b := by
    rw [tail_atom_decomposition μ C hC η] at hl
    change β ≤ a + b at hl
    linarith
  let m := if b = 0 then 0 else (β - a) / b
  have hm : 0 ≤ m ∧ m ≤ 1 ∧ a + m * b = β := by
    by_cases hb0 : b = 0
    · simp only [m, if_pos hb0, zero_mul, add_zero]
      exact ⟨by norm_num, by norm_num, by linarith⟩
    · have hbpos : 0 < b := lt_of_le_of_ne hb (Ne.symm hb0)
      simp only [m, if_neg hb0]
      exact ⟨div_nonneg ha hb, (div_le_one hbpos).2 hab,
        by rw [div_mul_cancel₀ _ hb0]; ring⟩
  let ζ := fun x => if η < C x then 1 / β else if C x = η then m / β else 0
  have huSet : MeasurableSet {x | η < C x} := hC measurableSet_Ioi
  have hbSet : MeasurableSet {x | C x = η} := hC (measurableSet_singleton η)
  have hzid : ζ = (fun x => {x | η < C x}.indicator (fun _ => 1 / β) x +
      {x | C x = η}.indicator (fun _ => m / β) x) := by
    funext x
    by_cases hx : η < C x
    · simp [ζ, Set.indicator, hx, ne_of_gt hx]
    · by_cases hx' : C x = η
      · simp [ζ, Set.indicator, hx, hx']
      · simp [ζ, Set.indicator, hx, hx']
  have hzmeas : Measurable ζ := by
    rw [hzid]
    exact (measurable_const.indicator huSet).add (measurable_const.indicator hbSet)
  have hzbox : ∀ x, 0 ≤ ζ x ∧ ζ x ≤ 1 / β := by
    intro x
    dsimp [ζ]
    split_ifs
    · exact ⟨(one_div_pos.mpr hβ).le, le_rfl⟩
    · exact ⟨div_nonneg hm.1 hβ.le, div_le_div_of_nonneg_right hm.2.1 hβ.le⟩
    · exact ⟨le_rfl, (one_div_pos.mpr hβ).le⟩
  have hmass : (∫ x, ζ x ∂μ) = (a + m * b) / β := by
    rw [hzid, integral_add ((integrable_const _).indicator huSet)
      ((integrable_const _).indicator hbSet), integral_indicator_const _ huSet,
      integral_indicator_const _ hbSet]
    change a * (1 / β) + b * (m / β) = (a + m * b) / β
    ring
  have hz : feasible μ β ζ :=
    ⟨hzmeas, hzbox, by rw [hmass, hm.2.2, div_self hβ.ne']⟩
  have hpoint : ∀ x, ζ x * (C x - η) = max (C x - η) 0 / β := by
    intro x
    by_cases hx : η < C x
    · simp [ζ, hx, max_eq_left (sub_pos.mpr hx).le, div_eq_mul_inv, mul_comm]
    · by_cases hx' : C x = η
      · simp [hx']
      · simp [ζ, hx, hx', max_eq_right (sub_nonpos.mpr (le_of_not_gt hx))]
  have hzint := feasible_integrable μ β ζ hz
  have hprod := density_cost_integrable μ (fun x => C x - η) ζ
    (hCi.sub (integrable_const η)) β hz
  refine ⟨η, ζ, hz, ?_⟩
  rw [show (fun x => ζ x * C x) = (fun x => η * ζ x + ζ x * (C x - η)) by
    funext x; ring, integral_add (hzint.const_mul η) hprod, integral_const_mul,
    hz.2.2, mul_one]
  simp_rw [hpoint]
  rw [integral_div]
  rfl

def primalValues (C : Ω → ℝ) (β : ℝ) : Set ℝ := Set.range (objective μ C β)
def dualValues (C : Ω → ℝ) (β : ℝ) : Set ℝ :=
  {r | ∃ ζ, feasible μ β ζ ∧ r = ∫ x, ζ x * C x ∂μ}

theorem risk_envelope (C : Ω → ℝ) (hC : Measurable C) (hCi : Integrable C μ)
    (β : ℝ) (hβ : 0 < β) (hβ1 : β < 1) :
    sInf (primalValues μ C β) = sSup (dualValues μ C β) := by
  obtain ⟨η, ζ, hz, he⟩ := attainment μ C hC hCi β hβ hβ1
  let r := objective μ C β η
  have hp : r ∈ primalValues μ C β := ⟨η, rfl⟩
  have hd : r ∈ dualValues μ C β := ⟨ζ, hz, he.symm⟩
  have lower : ∀ v ∈ primalValues μ C β, r ≤ v := by
    rintro v ⟨t, rfl⟩
    dsimp [r]
    rw [← he]
    exact weak_duality μ C ζ hC hCi β t hβ hz
  have upper : ∀ v ∈ dualValues μ C β, v ≤ r := by
    rintro v ⟨z, hz', rfl⟩
    exact weak_duality μ C z hC hCi β η hβ hz'
  have hip : sInf (primalValues μ C β) = r :=
    le_antisymm (csInf_le ⟨r, lower⟩ hp) (le_csInf ⟨r, hp⟩ lower)
  have hid : sSup (dualValues μ C β) = r :=
    le_antisymm (csSup_le ⟨r, hd⟩ upper) (le_csSup ⟨r, upper⟩ hd)
  rw [hip, hid]

theorem bounded_cost_risk_envelope (C : Ω → ℝ) (hC : Measurable C)
    (M : ℝ) (hbound : ∀ x, |C x| ≤ M) (β : ℝ) (hβ : 0 < β) (hβ1 : β < 1) :
    sInf (primalValues μ C β) = sSup (dualValues μ C β) := by
  apply risk_envelope μ C hC _ β hβ hβ1
  exact (integrable_const M).mono' hC.aestronglyMeasurable
    (Eventually.of_forall (fun x => by simpa only [Real.norm_eq_abs] using hbound x))

theorem bounded_cost_endpoint (C : Ω → ℝ) (hC : Measurable C)
    (M : ℝ) (hbound : ∀ x, |C x| ≤ M) :
    sInf (primalValues μ C 1) = (∫ x, C x ∂μ) ∧
      sSup (dualValues μ C 1) = (∫ x, C x ∂μ) := by
  have hCi : Integrable C μ :=
    (integrable_const M).mono' hC.aestronglyMeasurable
      (Eventually.of_forall (fun x => by simpa only [Real.norm_eq_abs] using hbound x))
  have hz : feasible μ 1 (fun _ => 1) := by
    simp [feasible, probReal_univ]
  have he : objective μ C 1 (-M) = ∫ x, C x ∂μ := by
    have hm : (fun x => max (C x - -M) 0) = fun x => C x + M := by
      funext x
      rw [sub_neg_eq_add, max_eq_left]
      have hc := (abs_le.mp (hbound x)).1
      linarith
    rw [objective, hm, integral_add hCi (integrable_const M)]
    simp [probReal_univ]
  let r := ∫ x, C x ∂μ
  have hp : r ∈ primalValues μ C 1 := ⟨-M, he⟩
  have hd : r ∈ dualValues μ C 1 := ⟨fun _ => 1, hz, by simp [r]⟩
  have lower : ∀ v ∈ primalValues μ C 1, r ≤ v := by
    rintro v ⟨η, rfl⟩
    simpa [r] using weak_duality μ C (fun _ => 1) hC hCi 1 η (by norm_num) hz
  have upper : ∀ v ∈ dualValues μ C 1, v ≤ r := by
    rintro v ⟨ζ, hz', rfl⟩
    simpa [he, r] using weak_duality μ C ζ hC hCi 1 (-M) (by norm_num) hz'
  exact ⟨le_antisymm (csInf_le ⟨r, lower⟩ hp) (le_csInf ⟨r, hp⟩ lower),
    le_antisymm (csSup_le ⟨r, hd⟩ upper) (le_csSup ⟨r, upper⟩ hd)⟩

theorem bounded_cost_full_tail (C : Ω → ℝ) (hC : Measurable C)
    (M : ℝ) (hbound : ∀ x, |C x| ≤ M) (β : ℝ) (hβ : 0 < β) (hβ1 : β ≤ 1) :
    sInf (primalValues μ C β) = sSup (dualValues μ C β) := by
  rcases lt_or_eq_of_le hβ1 with hlt | rfl
  · exact bounded_cost_risk_envelope μ C hC M hbound β hβ hlt
  · obtain ⟨hp, hd⟩ := bounded_cost_endpoint μ C hC M hbound
    exact hp.trans hd.symm

#print axioms cdf_quantile_exists
#print axioms distribution_quantile_exists
#print axioms quantile_exists
#print axioms feasible_integrable
#print axioms positive_part_integrable
#print axioms density_cost_integrable
#print axioms weak_duality
#print axioms tail_atom_decomposition
#print axioms attainment
#print axioms risk_envelope
#print axioms bounded_cost_risk_envelope
#print axioms bounded_cost_endpoint
#print axioms bounded_cost_full_tail

end ThesisCVaR
