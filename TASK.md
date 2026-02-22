TASK write a new file, from the old part.

## Status: DONE

- [x] Preamble with pragmatic interpretability framing
- [x] 30 entries (1-30) with pseudocode, hypothesis, evidence, grade
- [x] All papers saved to docs/ (full size, no truncation)
- [x] URLs from gist_content.md included
- [x] Sub-agent review completed, fixes applied:
  - Fixed RandLoRA pseudocode (sum of scaled random bases, not single triple product)
  - Fixed authorship (AntiPaSTO is Clark, not Bini/Girrbach/Akata)
  - Fixed SSVD grade (** not **!) and evidence ("matches" not "outperforms")
  - Fixed OFT pseudocode (W @ R^T convention per paper)
  - Fixed AntiPaSTO Cayley convention to show explicit /2
  - Added AntiPaSTO grade caveat (<=4B models, seed variance)
  - Split Bone/Trainable Tokens into separate entries
  - Fixed "Clark et al." -> "Clark"

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
