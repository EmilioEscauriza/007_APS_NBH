# Theory Notes

This folder contains derivations and theoretical notes for the XPCS powder film thickness model.

## File Naming Convention

Use descriptive names with underscores:
- `bragg_grains_count.md` - Number of diffracting grains
- `path_length_footprint.md` - Beam path length and footprint
- `speckle_contrast.md` - Speckle contrast theory
- `overlap_probability.md` - Birthday problem for spot overlap

## Format

Each file should follow this structure:

```markdown
# Title: Brief description

## Context / Motivation
Why this derivation matters

## Assumptions
- List key assumptions

## Derivation
Step-by-step derivation with LaTeX math

## Result
Final equations

## Notes / Limitations
When this breaks down, alternatives

## References
Relevant papers/books
```

## Viewing

- **VS Code / Cursor**: Press `Cmd+Shift+V` (Mac) or `Ctrl+Shift+V` (Windows/Linux) to open Markdown preview
- **GitHub**: Files render automatically when pushed
- **Command line**: Use `pandoc` to convert to PDF/HTML if needed

## LaTeX Math Syntax

- Inline math: `$E = mc^2$` → $E = mc^2$
- Display math: `$$\int_0^\infty e^{-x} dx = 1$$` → $$\int_0^\infty e^{-x} dx = 1$$
- Aligned equations: Use `\begin{align}...\end{align}`
