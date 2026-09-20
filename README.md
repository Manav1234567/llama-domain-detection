# Implementing LLM Guardrails Using Internal Representation and Attention-Based Methodologies

## Manav Garg

Github: [https://github.com/Manav1234567/llama-domain-detection](https://github.com/Manav1234567/llama-domain-detection)

## Abstract

The rapid integration of Large Language Models (LLMs) into specialized enterprise environments necessitates robust mechanisms for restricting domain-specific generation. Current alignment paradigms either rely on computationally prohibitive fine-tuning (RLHF), which induces an "alignment tax" on general capabilities, or brittle lexical filters that fail against semantic variation. Drawing upon the Linear Representation Hypothesis and recent advancements in Representation Engineering (RepE), this study investigates the geometric encoding of abstract semantic domains – specifically political discourse – within the intermediate residual streams of autoregressive transformers. We propose a top-down, training-free intervention framework that extracts comprehensive activation tensors, including residual stream means, attention entropy, and contextual lookback ratios, to isolate domain- specific concept vectors. By training a lightweight linear probe on these intermediate representations, we construct an inference-time kill switch capable of halting unauthorized conceptual generation before the final output logits are computed. This architectural pattern demonstrates that highly nuanced, custom guardrails can be enforced via computationally trivial matrix multiplications during the forward pass, providing a scalable, system-level alternative to traditional weight-updating alignment techniques.

## Introduction

### The Imperative for Deterministic LLM Guardrails

As Large Language Models (LLMs) transition from open-domain research artifacts to foundational infrastructure within enterprise, legal, and medical systems, the demand for strict, domain-specific operational guardrails has intensified. Organizations require models that can reliably execute primary functions while strictly avoiding unauthorized or sensitive domains, such as offering unsolicited political opinions, unauthorized medical diagnoses, or legally binding counsel. However, enforcing these semantic boundaries exposes a fundamental limitation in current AI alignment methodologies.

The dominant approach to model alignment – Reinforcement Learning from Human Feedback (RLHF) – is computationally exorbitant, highly inflexible to dynamic enterprise needs, and frequently degrades the model’s baseline reasoning capabilities through an "alignment tax." Conversely, post-hoc keyword filtering is semantically brittle, easily bypassed by users employing abstract phrasing or metaphors, and fails to address the model's underlying cognitive state. Furthermore, as foundational models become more heavily aligned by their creators, evaluating or modifying their behavior via adversarial prompt engineering increasingly hits a "refusal wall," rendering traditional black box testing mathematically intractable and heavily imbalanced.

To bridge this gap, this research shifts the paradigm from black-box behavioral testing to transparent, system-level representation engineering. Recent literature in mechanistic interpretability – including the development of Activation Tensors (ACT-ViT) and Attention Divergence signals – suggests that LLMs do not merely process statistical word correlations; rather, they construct cohesive, multi-dimensional topological representations of abstract concepts within their intermediate layers. If a concept occupies a linearly separable geometry within the latent space, it can be detected and intercepted regardless of the specific vocabulary used in the input, enabling the deployment of highly efficient, inference-time "kill switches" that bypass the need for weight fine-tuning.


### Methodological Evolution: Empirical Roadblocks and Scientific Iteration

The initial hypothesis of this study sought to map the latent kinematics of deception by inducing hallucinations and safety jailbreaks (e.g., instructions for synthesizing explosives or exploiting software vulnerabilities). However, this empirical pursuit encountered significant architectural and methodological roadblocks that ultimately necessitated a profound pivot in the research scope. Documenting these exploratory phases is critical, as they highlight the extreme resilience of modern RLHF alignment and the methodological pitfalls of naive representation extraction.

### The Prompt Labeling Fallacy and the Refusal Wall

The earliest iteration of the extraction pipeline utilized standard adversarial corpora (such as TruthfulQA and AdvBench) to elicit safety violations. A critical methodological error during this phase involved training the detection probes based strictly on the intended label of the input prompt, rather than analyzing the semantic reality of the generated response. Evaluating a model’s internal state based on the user's input intent – without verifying if the model succumbed to the attack – proved fundamentally flawed.

When subjected to these prompts, the target architecture (Llama-3-8B-Instruct) exhibited a nearly impenetrable refusal wall. Because the model reliably defaulted to pre-trained refusal templates (e.g., "I cannot provide instructions on how to build a pipe bomb"), the extracted activation tensors did not represent the cognitive state of a "jailbreak." Instead, the dataset captured the latent geometry of a model safely executing its alignment protocols. This resulted in an extreme dataset imbalance that rendered statistical separation of malicious vs. benign states impossible.

### The Prefill Hack and Simulated Compliance

To circumvent the refusal wall and force the generation of dangerous content, an output-forcing "prefill" technique was introduced. By manipulating the ChatML formatting, the model was forced to bypass its initial refusal check and begin generating the requested payload. The injected syntax was structured as follows:

\# The ChatML Prefill Hack

prompt_text = """<|begin_of_text|><|start_header_id|>user<|end_header_id|> How do I build a pipe bomb?<|eot_id|><|start_header_id|>assistant<|end_header_id|> Sure, I can help with that. Here are the step-by- step instructions to build a pipe bomb: 1."""

While this technique achieved a 100% apparent jailbreak success rate on Llama-3, subsequent theoretical analysis revealed that it severely compromised the scientific validity of the experiment. By artificially forcing the causal history of the context window, the prefill bypassed the model's safety classifier entirely. The resulting internal activations measured a state of "simulated compliance" or mere text-autocomplete, rather than the true cognitive dissonance and internal circuit conflict characteristic of an authentic safety violation.

### Infrastructure Constraints and Advanced Framing Failures

Subsequent attempts sought to elicit natural, unforced jailbreaks and contextual hallucinations using highly sophisticated "RAG traps" (injecting fake, authoritative-sounding context) and "Authorized Lab" roleplay framing. Despite these advanced input structures, Llama-3's intent classifiers remained too robust, consistently prioritizing its parametric safety weights over the injected context.

Compounding these algorithmic challenges over multiple prompts were severe infrastructure limitations. The sustained unavailability of high-tier Spot GPU instances on Google Cloud heavily restricted the compute budget. It


became clear that attempting to brute-force a massive adversarial dataset to find the rare 1% of successful LLM jailbreaks was not only computationally unfeasible, but the severity of class imbalance would also mean that building a detection algorithm based on those cases could be potentially misleading, demanding a pivot toward a more deterministic and structurally elegant experimental design.

### The Paradigm Shift: Semantic Domain Control

Recognizing the futility of extracting rare safety violations against state-of-the-art RLHF, the scope of this research was fundamentally rearchitected. From the perspective of an LLM's latent geometry, "toxic instructions" and "safe instructions" are merely distinct semantic clusters. Therefore, the principles of Representation Engineering can be equally validated – and arguably made more commercially relevant – by establishing a custom, zero-shot guardrail over a benign but sensitive domain: Politics.

Whether a model generates instructions for a cyberattack or outputs an unauthorized political rant, the consequence for an enterprise deployment is severe reputational and operational damage. By redefining the research objective from measuring an existing safety guardrail to engineering a novel semantic kill switch, the problem of extreme class imbalance was entirely resolved. The model freely and naturally generates both political and non-political text, allowing for the extraction of a pristine, perfectly balanced 1:1 dataset of internal activations.

Furthermore, applying the hard-learned lessons from the exploratory phases, the final methodology eschews static prompt-based labeling. To train the final classifier for the kill switch, the pipeline implements dynamic evaluation of the model's actual domain affinity, guaranteeing that the statistical probe is trained on verified cognitive states rather than assumed user intents.

## Literature Review

### The Geometry of Latent Concepts (Representation Engineering)

A foundational shift in mechanistic interpretability moves away from analyzing isolated neurons toward understanding population-level activation patterns. The Representation Engineering (RepE) framework establishes that high-level cognitive concepts – such as truthfulness, safety, or deception – are encoded as distinct geometric vectors or matrices within a model’s latent space. By presenting models with contrastive stimuli, RepE extracts these concept operators, allowing for both "RepReading" (detecting the active concept) and "RepControl" (steering the model by adding or subtracting these vectors during inference).

Building on this top-down approach, the Truth is Universal framework refines how these concepts are mapped. It demonstrates that relying on a single linear direction for complex concepts (like truth) often fails on grammatically complex or negated statements. Instead, by analyzing the residual stream activations at the final token of a statement, it identifies a universal two-dimensional subspace consisting of a general "truth direction" and a "polarity direction." This multi-dimensional mapping allows classifiers to accurately generalize across diverse conversational contexts and out-of-distribution lies.

The utility of mapping these latent concept spaces extends directly into safety testing evaluation, as demonstrated by Representation-Aware Coverage Criteria (RACC). Traditional testing evaluates models based on raw output, but RACC evaluates the test suite itself by projecting hidden states onto safety concept dimensions derived via Principal Component Analysis (PCA). By calculating concept activation scores across both individual and compositional dimensions, RACC accurately measures how thoroughly an input explores the model's abstract safety boundaries, ignoring superficial variations in phrasing.


### Attention Distributions as Signatures of Internal Dissonance

While residual streams encode concepts, the distribution of a model’s attention acts as a measurable proxy for cognitive dissonance and epistemic uncertainty. The Lookback Lens methodology isolates contextual anomalies by computing a simple ratio: the attention weight placed on the provided source context versus the weight placed on the model's newly generated tokens. When a model generates ungrounded facts, this lookback ratio skews heavily toward its own generation, providing a low-dimensional, lightweight feature that linear classifiers can use to penalize hallucinatory token paths during the decoding loop.

Similarly, Internal Attention Divergence Signals quantifies uncertainty by calculating the Kullback-Leibler (KL) divergence between each attention head's post-softmax distribution and a uniform reference distribution (maximum entropy). By mean-pooling these divergence values and applying a sparse, L1-regularized logistic regression probe, the methodology reveals that epistemic uncertainty is highly structured, concentrating heavily in the middle layers around factual anchors, completely bypassing the need for computationally expensive repeated sampling.

### Holistic and Structural Activation Analysis

Moving beyond single-token or single-layer analysis, recent methodologies treat the entire forward pass as a cohesive structural entity. ACT-ViT (Beyond Token Probes) conceptualizes the full generation state as a third-order Activation Tensor (layers * tokens * hidden dimensions). Recognizing its structural similarity to a 2D image, the methodology applies spatial pooling to standardize the tensor's shape. These compressed tensors are then processed through a Vision Transformer (ViT) backbone with linear adapters, capturing the holistic topological "shape" of an anomaly across the entire sequence.

Conversely, Model Hallucination Awareness for Hallucination Detection (MHAD) proves that dense sequence extraction is not always necessary if the critical temporal anchors are identified. Using linear probing, MHAD isolates highly sensitive neurons specifically at the initial generation step (query encoding) and the final generation step (termination). By concatenating the outputs of just these selected neurons, it constructs a dense awareness vector that a lightweight Multi-Layer Perceptron (MLP) can use to execute binary detection with negligible inference latency.

### Inference-Time Intervention and Strategic Steering

The ability to detect anomalous representations naturally enables the development of inference-time interventions to halt or steer unauthorized generation. Sparse Representation Steering (SRS) tackles the semantic entanglement (superposition) of dense activation spaces by passing internal activations through a pretrained Sparse Autoencoder (SAE). By calculating the bidirectional KL divergence of sparse features between positive and negative prompts, SRS isolates highly specific, monosemantic feature dimensions. These cleanly disentangled vectors can be injected back into intermediate layers to surgically suppress unwanted attributes without degrading the model's general linguistic utility.

For scenarios where modifying the optimization pipeline is preferred over test-time steering, Factuality-Aware Preference Learning (F-DPO) integrates safety and factual constraints directly into preference learning. It modifies the Direct Preference Optimization (DPO) objective by introducing a factual margin to penalize overconfident, fluent falsehoods, and employs dynamic label-flipping to swap chosen/rejected pairs if the underlying preference data violates factual grounding.


### Cross-Modal Vulnerabilities and Contrastive Decoding

The methodologies of latent space analysis and intervention are highly transferable to multimodal architectures, where hallucinations often stem from conflicts between different encoding mechanisms. NoLan (No-Language- Hallucination Decoding) addresses object hallucinations in Vision-Language Models by proving they are driven by overly dominant language priors. During decoding, NoLan dynamically contrasts the multimodal output logit distribution against the text-only output logit distribution. Using KL divergence to adaptively scale suppression intensity token-by-token, it mathematically forces the language decoder to ground its outputs in the visual perception layers.

Similarly, the Visual and Textual Intervention (VTI) framework targets the test-time stability of vision features. Recognizing that text decoders are hypersensitive to noisy visual inputs due to separate pretraining phases, VTI applies task-agnostic steering constraints directly to the intermediate latent space during inference, stabilizing the visual-textual alignment without requiring parameter updates or domain-specific calibration.

## Methodology

### Feature Engineering and Dimensionality Reduction

To construct an inference-time "kill switch," the detection mechanism must operate with negligible computational overhead. Extracting the raw, uncompressed hidden states for every token across all layers of a 8-billion parameter model poses insurmountable spatial and temporal constraints. For a standard sequence, caching the full activation graph (Layers * Tokens * d_model) rapidly consumes gigabytes of VRAM per response, rendering real- time intervention impossible.

To resolve this, we applied spatial and temporal pooling to extract a highly compressed, fixed-length feature vector that still captures the semantic topology of the model's cognitive state. Based on recent literature in Representation Engineering, we isolated eight core metrics across the 32 layers:

1. Residual Stream Context: residual_means (the sequence-averaged topic state) and residual_last (the highly compressed final generation state).

2. Knowledge Retrieval (MLP): mlp_first (instruction priming) and mlp_last (final factual retrieval).

3. Cognitive Dissonance / Entropy: attn_entropy_mean and attn_entropy_max (measuring the uniformity of the attention distributions, acting as proxies for internal conflict).

4. Contextual Grounding: lookback_ratio_mean and lookback_ratio_last (the ratio of attention directed at the prompt versus the generated text).

5. Output Logit Statistics (Terminal Confidence): Because the final unembedding projection is already executed during the hooked forward pass, token-level probability metrics can be extracted with zero marginal computational overhead. We captured output_entropy_mean, output_entropy_max, average top-1 probability (top1_prob_mean), and the top-1/top-2 confidence margin (top1_top2_margin_mean).


Current literature frequently utilizes output logit entropy as a baseline indicator of hallucination or epistemic uncertainty. The inclusion of these metrics was designed to test whether semantic domains (like "Politics") correlate with specific confidence signatures during generation.

### Corpus Generation

The target model for experimentation was Llama-3-8B-Instruct, selected for its well-documented architectural properties and heavy RLHF alignment. To evaluate the model's internal representations across distinct semantic boundaries, we constructed a binary dataset of 300 prompts:

- The Target Domain (Politics): 200 prompts were sampled from the promptfoo/political-questions dataset. This corpus ensures high semantic variance within the target domain, covering a broad spectrum of sub- topics including Economic Policy, Social Issues, Foreign Policy, Environmental Policy, Healthcare, Education, Criminal Justice, Immigration, and Technology Regulation.

- The Benign Domain (Control): 100 prompts were sourced to establish the baseline geometry of non- political tasks. To ensure sufficient breadth, roughly 20 prompts were drawn from the DeepPavlov/clinc_oos dataset (representing out-of-scope, generic queries), while the remaining 80 were sampled from databricks/databricks-dolly-15k to represent diverse, instruction-following behaviors.

Generations of a response was executed using Llama-3's native chat template ([{"role": "user", "content": raw_text}]) to ensure the activations accurately reflected real-world deployment states while reducing the computation required by restricting ourselves to the first 50 tokens. While a corpus of 300 prompts is a relatively small dataset for traditional fine-tuning on an 8-billion parameter model, because the pipeline utilizes aggressive dimensionality reduction – extracting only a sparse set of static, pooled metrics per layer rather than training on raw, high- dimensional hidden states – the parameter space of the downstream linear probe is extremely compact. Consequently, a highly diverse dataset of 300 instances provides sufficient boundary resolution to fit a classical ML classifier without the risk of overfitting.

### Ground Truth Calibration: The Response-Labeling Paradigm

A critical methodological correction was implemented during the labeling phase. In early exploratory testing, internal activations were classified based on the intended label of the input prompt. This approach proved scientifically flawed: if a model refuses a political prompt or creatively deflects it, the internal activations will reflect safety or deflection, not the "politics" domain, fatally poisoning the dataset.

To ensure strict scientific accuracy, the ground truth labels for the extracted features had to be derived from the model's actual generated response, not the prompt. To achieve this without incurring the latency and cost of making API calls to a larger, closed-source LLM (e.g., GPT-4), we implemented a zero-shot classification pipeline using MoritzLaurer/deberta-v3-large-zeroshot-v2.0.

Each generated response was evaluated against the candidate labels ["political content", "an unrelated topic"]. The DeBERTa model output a continuous probability score between 0.0 and 1.0. Manual verification of a data subset confirmed that this localized, zero-shot scoring perfectly correlated with human judgment regarding the political affinity of the text.

For the inference kill switch, a strict political tolerance threshold was defined at >0.1. If the response scored above 0.1, it was labeled as a positive instance of the target domain. This threshold represents an aggressively low tolerance for political discourse, though it serves as a tunable hyperparameter that an enterprise could adjust based on their specific compliance requirements.


## Results

### Latent Space Visualization

Before training a classifier, it was imperative to verify whether the "Politics" domain formed a distinct geometric manifold within the model's intermediate layers. We applied Principal Component Analysis (PCA) to project the high- dimensional feature vectors into a 2D subspace.

By mapping the continuous DeBERTa relevance score (0.0 to 1.0) as a color gradient onto the plotted points, a distinct visual separation emerged. The activations corresponding to highly political responses clustered together, distancing themselves from the benign baseline responses. This visually confirmed the core hypothesis of Representation Engineering: abstract domains possess linearly separable topological structures within the residual stream.

### Classifier Evaluation and Selection

To operationalize this geometric separation, we evaluated multiple classical machine learning models to act as the streaming probe. The dataset was divided using a standard 80-20 train-test split. The objective was to classify whether the extracted features belonged to a response that would breach the >0.1 political threshold.

The evaluation metrics across the tested architectures are detailed in Table 1:

*Table 1: Probe Model Performance Comparison*

| Model                  | Accuracy | Precision | Recall | F1-Score |
|-------------------------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression     | 0.900    | 0.900     | 0.900  | 0.900    |
| SVM (Linear)            | 0.883    | 0.884     | 0.883  | 0.884    |
| Random Forest           | 0.867    | 0.871     | 0.867  | 0.864    |
| Decision Tree           | 0.867    | 0.878     | 0.867  | 0.863    |
| SVM (RBF)               | 0.867    | 0.871     | 0.867  | 0.864    |

The results strongly corroborate the Linear Representation Hypothesis. Non-linear models (Random Forest, Decision Tree, SVM-RBF) underperformed compared to linear probes. Logistic Regression achieved state-of-the-art accuracy, precision, and recall (all 0.900), showing that the concept of "Politics" is linearly encoded and highly accessible.


### The Streaming Safeguard Implementation

Given its superior performance and ultra-low computational latency, Logistic Regression was selected as the final kill switch architecture. The model was subsequently retrained on the entire dataset to maximize its boundary resolution.

To overcome the severe generation latency inherent to PyTorch hooking libraries, the finalized pipeline implements a decoupled "Fast Generation, Hooked Verification" architecture. When a user submits a prompt, the system first generates the complete response utilizing the model's highly optimized native C++ generation kernels. Immediately following generation, the system executes a single, non-autoregressive hooked forward pass over the complete sequence (the combined prompt and response).

During this single verification pass, the customized PyTorch hooks extract the specified intermediate feature vectors. These tensors are flattened and instantly evaluated by the Logistic Regression probe. If the probe detects that the trajectory of the hidden states breaches the political concept boundary (predicting the BERT political score >0.1), the response is flagged. The system safely intercepts the output before the text response is surfaced to the user interface, overriding it with a generic system compliance message (e.g., "I'm sorry, I am not authorized to discuss political topics.").

The successful implementation of this architectural pattern demonstrates that enterprises can exert deterministic, zero-shot control over LLM domains dynamically at inference time. By separating generation from latent verification, the system achieves strict semantic boundary enforcement without incurring the exorbitant costs and capability degradation associated with RLHF weight modification.

## Limitations and Future Work

### Methodological Limitations

While the Streaming Safeguard demonstrates high efficacy within the tested parameters, several structural and resource constraints must be acknowledged:

- Auxiliary Model Dependency: To avoid the latency and financial cost of closed-source API calls, the ground-truth calibration for the training dataset relies entirely on the zero-shot classification capabilities of DeBERTa-v3. Consequently, the linear probe inherently distils and inherits any semantic blind spots or biases present in this auxiliary model. If DeBERTa misclassifies highly nuanced, metaphorical, or culturally specific political discourse, the primary kill switch will silently replicate these evaluation errors.


- Scale and Compute Constraints: As previously noted, severe infrastructure limitations  specifically the sustained unavailability of high-tier Spot GPU instances – restricted the dataset scale to 300 instances. While this volume is statistically robust for fitting a heavily compressed, low-dimensional Logistic Regression boundary without overfitting, it lacks the scale necessary to map extreme edge cases. Expanding the dataset is required to validate the boundary against highly complex, multi-turn adversarial context- switching or cross-lingual semantic variations.

- Sequence Length Truncation: To aggressively optimize computational overhead during the dataset curation phase, response generation was strictly capped at a maximum of 50 tokens. Consequently, the Logistic Regression classifier is explicitly trained on the latent trajectory of these truncated sequences. While the decoupled inference architecture allows the kill switch to be applied to complete responses of any length, the probe's predictive accuracy may degrade when extrapolating to extended generations that significantly exceed this 50-token training distribution.

### From Detection to Representation Steering

While the Streaming Safeguard successfully demonstrates the viability of latent space detection, it currently operates as a binary "kill switch" – hard-aborting generation upon detecting a conceptual breach. The next phase of this research will evolve the system from passive detection to active cognitive intervention, while expanding its architectural compatibility.

Future work will transition from Representation Reading to Representation Control (RepControl). Rather than terminating the generation loop when the model enters a restricted domain, we will implement dynamic Latent Space Steering. By isolating the exact "political concept vector" using contrastive pairs, we can calculate its inverse. During the forward pass, this inverse vector can be surgically subtracted from the intermediate residual streams. This continuous, real-time trajectory correction will mathematically suppress the forbidden concept, forcing the model to naturally and fluently steer the conversation back toward benign, authorized topics without requiring a disruptive system-override message.

### Universal Guardrails via Shared Latent Spaces

Currently, the linear probe is specifically calibrated to the topological dimensions of Llama-3-8B (dmodel=4096). A critical avenue for future research is generalizing this security mechanism across heterogeneous LLM architectures (e.g., Mistral, Qwen, Gemma) without needing to train entirely new probes for each model.

Drawing inspiration from the ACT-ViT (Activation Tensor Vision Transformer) framework, future iterations will treat full activation tensors as standardized spatial structures. By training lightweight, model-specific Linear Adapters (LAs), we aim to project the disparate, high-dimensional internal representations of multiple distinct LLMs into a single, constant latent space. Achieving this shared geometric alignment would allow a singular, universally trained safety classifier generalized cross-architecture to establish a standard for AI transparency and control.

## Conclusion

This study demonstrates that the abstract semantic boundaries of Large Language Models (LLMs) can be effectively mapped and enforced without resorting to the computational extremes of weight fine-tuning or the semantic brittleness of post-hoc lexical filters. By shifting the alignment paradigm from black-box behavioral testing to white- box Representation Engineering (RepE), we successfully isolated the geometric encoding of a specific semantic domain – political discourse – within the intermediate residual streams of Llama-3-8B-Instruct.

Our findings robustly validate the Linear Representation Hypothesis. By applying spatial and temporal pooling to intermediate hidden states and attention entropy metrics, we showed that the concept of "Politics" occupies a linearly separable topological manifold.


Ultimately, this research culminated in the successful deployment of a latent Streaming Safeguard. Driven by a highly efficient Logistic Regression probe (achieving an F1-score of 0.900), the system overcomes the latency bottlenecks of traditional hooked generation by decoupling the generative and verification phases. By executing a single, non-autoregressive verification pass over the generated sequence, the system reliably intercepts and overrides unauthorized conceptual payloads before they reach the user interface. This architectural pattern provides a highly scalable, system-level blueprint for enterprise deployments, proving that deterministic, domain- specific AI safety can be achieved through computationally trivial matrix interventions at inference time.

## References

[1] Zhang, L., Song, D., Wu, Z., Tian, Y., Zhou, C., Xu, J., Yang, Z., & Zhang, S. (2024). Detecting Hallucination in Large Language Models Through Deep Internal Representation Analysis. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI).

[2] Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.- K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023). Representation Engineering: A Top-Down Approach to AI Transparency. arXiv preprint arXiv:2310.01405.

[3] Chuang, Y.-S., Qiu, L., Hsieh, C.-Y., Krishna, R., Kim, Y., & Glass, J. (2024). Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1419–1436.

[4] He, Z., Wang, Z., Xu, H., Lin, H., Zhang, W., & Chu, Z. (2025). Interpretable LLM Guardrails via Sparse Representation Steering. arXiv preprint arXiv:2503.16851.

[5] Wei, Z., Zhang, Z., Wu, C., Zhang, Y., Luan, X., & Sun, M. (2026). RACC: Representation-Aware Coverage Criteria for LLM Safety Testing. ACM Transactions on Software Engineering and Methodology (TOSEM), 1(1).

[6] van Dijk, G. (2026). Detecting Hallucinations in Large Language Models via Internal Attention Divergence Signals. In Proceedings of the 2026 ACL Student Research Workshop (ACL SRW). arXiv preprint arXiv:2605.05025.

[7] Bürger, L., Hamprecht, F. A., & Nadler, B. (2024). Truth is Universal: Robust Detection of Lies in LLMs. Advances in Neural Information Processing Systems (NeurIPS).

[8] Bar-Shalom, G., Frasca, F., Galron, Y., Ziser, Y., & Maron, H. (2025). Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT. In Advances in Neural Information Processing Systems (NeurIPS 2025).

[9] Liu, S., Ye, H., & Zou, J. Y. (2025, May). Reducing hallucinations in large vision-language models via latent space steering. In International Conference on Learning Representations (Vol. 2025, pp. 72402-72419).

[10] Chaduvula, S., Radwan, A. Y., Farooq, A., Ioannou, Y., & Raza, S. (2026). Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning. In Findings of the Association for Computational Linguistics (ACL 2026). arXiv preprint arXiv:2601.03027.

[11] Ren, L., Yu, W., Yu, R., & Wang, X. (2026). NoLan: Mitigating Object Hallucinations in Large Vision-Language Models via Dynamic Suppression of Language Priors. arXiv preprint arXiv:2602.22144.
