# Thesis Editing Baseline

- The authoritative manuscript is `output/排版完成.docx`.
- Edit that Word document directly, retaining its current chapter structure,
  native equations, styles, section settings, and existing formatting.
- Do not rebuild or replace it using the older Markdown or LaTeX manuscript.
- The annotated PDF dated 20261001 provides restructuring context, not a newer
  manuscript baseline. Later user edits to Word take precedence.
- Keep a pre-edit Word backup. Do not overwrite completed experiment campaigns
  or change neighboring projects while editing the thesis.
- Completed result integration is recorded in
  `workpapers/Word结果整合_20261003/交接.md`.

# Experiment Scheduling

- Training may be restricted to GPU nodes when it needs a GPU.
- Tests and policy evaluation must use all available scheduler nodes, including
  CPU nodes. Do not bind evaluation campaigns to a single GPU machine.
- Reuse the distributed deployment in
  `workpapers/H2O纯源同执行诊断_20261006/说明.md` when the original runtime is needed;
  stage the required runtime and frozen inputs on the servers so evaluation can
  run across nodes. Return only small logs or result summaries locally.
