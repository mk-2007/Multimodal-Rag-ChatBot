# Multimodal RAG - Image Query Result

Source document: *Attention Is All You Need* (Vaswani et al., 2017)

**Query image:** `E:\test.png`

**Stage 1 - image described for search:**

> This diagram illustrates the "Multi-Head Attention" mechanism from the Transformer architecture. It depicts the flow of input vectors (V, K, Q) through parallel linear projections and multiple "Scaled Dot-Product Attention" layers, which are then concatenated and passed through a final linear transformation. Labels include "Multi-Head Attention," "Linear," "Concat," and "Scaled Dot-Product Attention," with the parameter $h$ denoting the number of attention heads.

**Retrieved context (shared vector store, mixed modalities):**

- `[0.846]` **figure** | `figure_3` | page 4 | _Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Atte_
- `[0.803]` **figure** | `figure_2` | page 4 | _Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Atte_
- `[0.773]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.766]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_
- `[0.753]` **text** | `page 2 body text` | page 2 | _3 Model Architecture_

**Grounded answer (dual-grounded: image + retrieved context):**

The provided image illustrates the **Multi-Head Attention** mechanism used in the Transformer architecture.

### What the image shows:
As described in **Context 1**, the diagram represents the structural flow of Multi-Head Attention:
*   **Inputs:** At the bottom, three distinct inputs—**V** (Values), **K** (Keys), and **Q** (Queries)—are fed into parallel "Linear" layers. These layers apply multiple linear projections to the inputs.
*   **Attention Heads:** The outputs from these projections pass into a series of stacked blocks labeled **"Scaled Dot-Product Attention."** The bracket labeled **"h"** indicates that the mechanism is divided into $h$ parallel attention heads.
*   **Output Processing:** The outputs from these parallel heads are combined in a **"Concat"** (concatenation) layer, followed by a final **"Linear"** layer that projects the concatenated output to its final form.

### Relation to the paper:
This figure, specifically labeled as part of Figure 2 in the paper, is central to the "Multi-Head Attention" section (Section 3.2.2) of the text:
*   **Joint Representation:** The paper explains that instead of performing a single attention function, it is beneficial to linearly project the queries, keys, and values $h$ times. This allows the model to "jointly attend to information from different representation subspaces at different positions" (**Context 4**).
*   **Mechanism:** The formula provided in the text, $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$, directly corresponds to the visual flow of projecting, computing parallel attention heads, concatenating them, and applying a final linear projection ($W^O$) shown at the top of the image (**Context 3**).
*   **Implementation:** The paper notes that they typically employ $h = 8$ parallel attention layers, each with a reduced dimensionality of $d_k = d_v = d_{model}/h = 64$, ensuring the total computational cost remains similar to single-head attention (**Context 3**).
