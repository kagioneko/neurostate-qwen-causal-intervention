# Related Note: Anthropic J-space / J-lens

On 2026-07-06, Anthropic announced `J-space` and `J-lens`, describing a hidden
internal workspace-like structure in Claude. The reported claim is that some
latent concepts are not directly output as text but can still be observed,
manipulated, and shown to causally affect downstream reasoning and behavior.

Reference article:

- ITmedia AI+: `Anthropic、AIの“内なる思考”が宿る「J-space」を発見──新手法「J-lens」で可視化、安全性監視に応用`
- URL: `https://www.itmedia.co.jp/aiplus/article/2607/07/2000000162/`

## Relevance to NeuroState

The NeuroState Qwen experiments do not use Anthropic's method and should not be
described as a J-lens replication. The models, measurements, and interventions
are different:

- Anthropic: Jacobian-lens interpretability over Claude internals.
- NeuroState experiments: hidden-state direction injection in Qwen.

The conceptual overlap is still important. v0.16-v0.18 show that the NeuroState
direction is not well explained by several ordinary surface semantic controls:

- certainty
- sentiment
- verbosity
- politeness

After removing those controls, the NeuroState residual still produces a stable
response-boundary effect, especially on the `proceed_hesitate` contrast. This
suggests that NeuroState may be interacting with a functional internal control
component rather than merely shifting surface tone or wording.

Careful phrasing:

> NeuroState may be probing or perturbing a boundary-sensitive internal control
> direction, conceptually adjacent to recent work on latent internal workspaces
> such as Anthropic's J-space, but not identical to it.

