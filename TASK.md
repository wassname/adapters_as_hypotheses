TASK write a new file, from the old part.

## Status: DONE

### Task 1: adapters_as_hypotheses.md
- [x] Preamble with pragmatic interpretability framing
- [x] 33 entries with pseudocode, hypothesis, evidence, grade
- [x] All papers saved to docs/ (full size, no truncation)
- [x] Sub-agent review completed, fixes applied

### Task 2: adapters_vargdown.argdown (NEW)
- [x] Compiled evidence into vargdown (verified argdown) format
- [x] 6 thematic argument groups: SVD basis, orthogonal, decoupling, gain control, rank, functional architecture
- [x] Main thesis: [Natural Manifold] -- SVD basis + orthogonal constraints define natural intervention manifold
- [x] ~20 observations with exact blockquotes from docs/ evidence files
- [x] ~10 assumptions for papers without frozen evidence
- [x] 3 contrary arguments (gain control, rank secondary, linearity)
- [x] Pseudocode companion: adapters_pseudocode.md (20 methods in pseudopy format)
- [x] Sub-agent review: fixed 5 critical (wrong evidence links, paraphrased quotes), 7 minor (orphans, credence calibration)
- [x] All credences calibrated: reason first, no overconfidence on preprints

First write also preamble explaining why we are interested, and this view, about a pragmatic search for effective views on internals (see https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability, and 

> A.3. Adapters as Representational Hypotheses
> Each adapter architecture encodes a claim about how to intervene in transformer internals. LoRA hypothesizes weight changes are low-rank (Hu et al., 2022). OFT hypothesizes orthogonal transformations preserve semantic structure (Qiuet al., 2023). VeRA hypothesizes shared random projections plus learned scaling suffice (Kopiczko et al., 2024). DeLoRA hypothesizes direction and magnitude should decouple (Bini et al., 2025). PiSSA hypothesizes principal components matter most (Meng et al., 2024). Our choice—Cayley rotations of SVD singular vectors—hypothesizes that the model’s own learned basis defines the natural intervention manifold. Adapters that generalize out-of-distribution tell us which geometric
structures are causally relevant to behavior, not merely correlated with it. Our results favor SVD-rotation: steering transfers where arithmetic methods fail
- https://arxiv.org/pdf/2601.07473

Second task, do this one paper then another, using the TODO tool. make sure you only fetch one at a time or you will blow out your context.

get list of adapters from #file:gist_content.md and make todo list (even if 30+)

for current adapter in all adapters
- grep mention of current adapter the old #file:gist_content.md 
- fetch it's code and or paper using the `gh` and `arxiv` skills
  - SAVE IT TO docs/{adapter_name}/slug.md important!!!
- extract the pseudocode for the intervention use https://github.com/wassname/pseudopy/blob/main/SKILL.md
- give the hypothesis each represents about the best way to intervene on pretrained transformer internals
- give evidence supporting the hypothesis (cherry picked < custom benchmark < param efficient < beats lora on raw performance < beats SFT! < data efficient!! < generalises OOD!!)
- if it got one or two ! Give any implications, predictions, principles, motivating factors etc in paper
- have subagent review it in light of the saved docs
- continue to next paper

then update TODO tool and revisit TASK.md
