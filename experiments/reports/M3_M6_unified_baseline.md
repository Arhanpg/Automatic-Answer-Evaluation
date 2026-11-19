# Milestones M3/M6: Semantic, NLI, and Symbolic Baseline

## Implemented baseline

The grading baseline combines rubric coverage, sentence semantics, natural
language inference, and SymPy verification. It also routes contradicted
answers to lower marks and checks mathematical equivalence.

## Stored anchor checks

The generated reports record these anchor-script checks:

| Script | Human mark | Automated mark | Main verified behavior |
|---|---:|---:|---|
| s2 | 20/20 | 20/20 | Laplace probability and LayerNorm text |
| s29 | 15/20 | 15/20 | Contradiction and partial-math handling |
| s18 | 7/20 | 7/20 | Omitted subparts and partial Q4 credit |

These checks demonstrate known-example behavior. They are not an independent
locked-test calculation for the complete 24-response test set.
