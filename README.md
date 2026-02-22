# Adapters as Representational Hypotheses

*What does each PEFT method believe about transformer internals?*

Each adapter architecture encodes a structural claim about how to intervene in pretrained weights. When one outperforms another under controlled conditions (same model, same data, same parameter budget), the winner's assumptions are supported as a better description of the weight manifold.

This catalog reframes ~30 PEFT methods as **hypotheses about transformer geometry**, extracts pseudocode for each intervention, and grades the evidence.

## Evidence hierarchy

| Grade | Meaning |
|-------|---------|
| * | Parameter-efficient (matches LoRA with fewer params) |
| ** | Beats LoRA on raw performance |
| **!** | Beats full fine-tuning |
| **!!** | Data-efficient (few-shot, fast convergence) |
| **!!!** | Generalizes out-of-distribution |

## Contents

- [adapters_as_hypotheses.md](adapters_as_hypotheses.md) -- the main catalog
- [docs/](docs/) -- saved papers (full text, markdown)

## Key findings

1. **SVD basis is the natural coordinate system.** Methods that use the model's own SVD decomposition (PiSSA, SVFT, SSVD, AntiPaSTO) consistently outperform random-basis methods at the same parameter count.
2. **Orthogonal >> arbitrary.** Orthogonal constraints (OFT, BOFT, HRA, AntiPaSTO) preserve semantic structure and improve OOD transfer, at the cost of limited magnitude changes.
3. **Direction and strength decouple.** Methods that separate *what to change* from *how much* (DeLoRA, ROAD, AntiPaSTO) show better robustness and enable bidirectional steering.
4. **Low-rank is necessary but not sufficient.** LoRA's rank bottleneck limits hard tasks; full-rank methods (RandLoRA, SHiRA) close the gap with full FT.
5. **Scaling alone goes far.** IA3 and LN Tuning show that a surprising amount of adaptation is just reweighting existing features -- "gain control" over channels.

## Related

- [A Pragmatic Vision for Interpretability](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) -- Nanda et al. 2025
- [AntiPaSTO: Antiparallel Steering](https://arxiv.org/abs/2601.07473) -- Clark 2025 (Appendix A.3 is the origin of this framing)
- [HuggingFace PEFT](https://github.com/huggingface/peft) -- reference implementations

## License

Content is CC-BY-4.0. Papers in docs/ are fetched from arXiv for reference and remain under their original licenses.
