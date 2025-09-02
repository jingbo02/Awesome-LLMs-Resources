# Multimodal Reasoning and Reinforcement Learning: Technical Paradigms and Future Outlook





### 1. Core Trend: Why "Multimodal + Reinforcement Learning" is Becoming the Main Axis for Leaps in Reasoning Capabilities



A large body of recent research indicates that solely relying on Supervised Fine-Tuning (SFT) to imitate Chain-of-Thought (CoT) presents significant bottlenecks. Reinforcement Learning (RL), by introducing more direct and verifiable feedback signals, is becoming key to enabling multimodal models to achieve generalizable, structured reasoning abilities.

- **Limitations of SFT**:
  - **Pseudo-reasoning and Verbosity**: Models learn language patterns that "look like" reasoning rather than true logical deduction, leading to verbose and hollow outputs ([VLAA-Thinking](https://arxiv.org/abs/2504.11468)).
  - **Cross-Domain Fragility**: After fine-tuning in a specific domain, a model's reasoning ability in other domains declines ([Vision-R1](https://arxiv.org/abs/2503.06749), [ReVisual-R1](https://arxiv.org/abs/2506.07516)).
  - **Reward Signal Mismatch**: The loss function of SFT (e.g., cross-entropy) is not perfectly aligned with the final task objective (e.g., accuracy, IoU).
  - **Catastrophic Forgetting**: Continuous SFT can cause models to forget the general capabilities learned during the pre-training phase ([VLAA-Thinking](https://arxiv.org/abs/2504.11468)).
- **Advantages and Levers of RL**:
  - **Low-Cost Hard Feedback**: In tasks that can be verified by rules, such as mathematics, geometry, code, detection, and segmentation, RL can use accuracy, IoU, execution results, etc., as direct reward signals to quickly instill structured reasoning abilities ([R1-VL](https://arxiv.org/abs/2503.12937), [Perception-R1](https://arxiv.org/abs/2504.07954), [VisionReasoner](https://arxiv.org/abs/2505.19651), [UniVG-R1](https://arxiv.org/abs/2505.14231), [SAM-R1](https://arxiv.org/abs/2505.22596)).
  - **Alignment for Open-Ended Tasks**: For open-ended tasks like creative writing, dialogue, and complex descriptions, RL can achieve alignment through soft rewards such as preference models, discriminators, or embedding similarity ([Omni-Thinker], [Mixed-R1](https://arxiv.org/abs/2505.24164), [R1-Omni], [Seed1.5-VL](https://arxiv.org/abs/2505.07062), [GLM-4.1V-Thinking](https://arxiv.org/abs/2507.01006)).
  - **Emergence of a New "Visually-Grounded Endogenous Thinking" Paradigm**: RL has pushed CoT beyond pure text sequences, developing new forms such as interleaved image-text, pixel-level operations, generation of intermediate visual states, and even pure visual planning ([Pixel Reasoner], [DeepEyes](https://arxiv.org/abs/2506.05943), [MINT-CoT](https://arxiv.org/abs/2506.05331), [GRIT](https://arxiv.org/abs/2505.15879), Thinking with Generated Images, [VPRL]).
- **Evolution of Training Paradigms**:
  - **Diversified Sequences**: The traditional "SFT → RL" pipeline is being replaced by more flexible paradigms, such as RL→SFT ([Metis-RISE](https://arxiv.org/abs/2506.13056)), iterative SFT↔RL ([OpenVLThinker](https://arxiv.org/abs/2503.17352)), and more complex cross-modal reinforcement learning sequences ([ReVisual-R1](https://arxiv.org/abs/2506.07516)).
  - **Zero Cold-Start Exploration**: Some work attempts to bypass SFT entirely, training end-to-end purely through RL from scratch ([VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132), [DeepEyes](https://arxiv.org/abs/2506.05943)).
  - **Self-Supervised RL**: Using the model's own output to construct pseudo-reward signals for unsupervised self-improvement (MM-UPT).
- **Shift in Core Bottlenecks**: With the widespread application of RL, challenges have shifted from "how to imitate" to deeper issues such as Reward Hacking, advantage function collapse, sample difficulty imbalance, interference between multiple tasks (MiMo-VL), quality control of CoT ([VLAA-Thinking](https://arxiv.org/abs/2504.11468)), and the balance between cross-domain transfer and forgetting (Omni-Thinker).



### 2. Overall Paradigm



The current multimodal reasoning RL framework can be abstracted as a five-tuple: **(Model, Data, Optimization, Reward, Reasoning Modality)**

- **Model (Model Architecture)**:
  - **Basic Backbone**: Unified Transformer architecture + high-resolution ViT has become mainstream (MiMo-VL, [GLM-4.1V](https://arxiv.org/abs/2507.01006)).
  - **Mixture of Experts (MoE)**: MoE is used to enhance model capacity and efficiency (Seed1.5-VL, Kimi-VL).
  - **Connector/Projector**: The trainability of the connector has been found to be a key prerequisite for aligning vision and language models ([Skywork-R1V3](https://arxiv.org/abs/2504.16656)).
  - **Long Context/Memory**: Interpolated memory or extended context windows to handle more complex reasoning chains ([GLM-4.1V 48K](https://arxiv.org/abs/2507.01006), [Seed1.5-VL 128K](https://arxiv.org/abs/2505.07062)).
- **Data (Data Engineering)**:
  - **Mixed Types**: Combining data from multiple verifiable task domains (math, code) with preference data from open domains.
  - **Difficulty Stratification/Dynamic Sampling**: Using curriculum learning or dynamic sampling based on task difficulty to improve training efficiency (Curr-ReFT, Omni-Thinker, GLM-RLCS, [UniVG-R1](https://arxiv.org/abs/2505.14231), [ThinkLite-VL](https://arxiv.org/abs/2504.07934)).
  - **Representative Datasets/Methods**: MoDoMoDo, [ReVisual-R1 GRAMMAR](https://arxiv.org/abs/2506.07516), Mixed-45K, [GThinker](https://arxiv.org/abs/2506.01078), Vision-R1-cold.
- **Optimization (Optimization Algorithms)**:
  - **GRPO and its Variants**: The GRPO family has become the mainstream choice due to its stability, including StepGRPO, GRPO-D with dynamic KL, MORL for multi-objective optimization, and DAPO (Original GRPO, StepGRPO, GRPO-D, MORL, DAPO).
  - **Advantage Stabilization Techniques**: Techniques like SSR/SSB caching, PAD (Priority Advantage Distillation), asymmetric clipping, and token-level normalization are used to address the advantage collapse problem.
  - **Other Algorithms**: PPO / lightweight PPO ([Open Vision Reasoner](https://arxiv.org/abs/2507.05255)) and generalized RLHF / RLAIF / RLVR frameworks are also widely used ([Mixed-R1](https://arxiv.org/abs/2505.24164), Seed1.5-VL, MiMo-VL).
- **Reward (Reward Design)**:
  - **Hard Verification**: Rule-based rewards, such as Accuracy, Intersection over Union (IoU), L1 distance, counting accuracy, and code executability.
  - **Structure/Format**: Rewards for the output's structure (e.g., JSON), step completeness, and logicality.
  - **Process-Level**: Rewarding each step or specific stage of the reasoning process (StepGRPO).
  - **Preference/Mixed**: Using discriminators, preference models, embedding similarity (BMAS), or a weighted mix of multiple rewards ([Mixed-R1](https://arxiv.org/abs/2505.24164), Omni-Thinker, MiMo-VL).
  - **Exploration/Regularization**: Introducing curiosity-driven exploration, difficulty adaptation, and penalties for length and redundancy.
- **Reasoning Modality (Forms of Reasoning)**:
  - **From Text to Multimodal**: Evolving from pure-text CoT to interleaved image-text (iMCoT), visual element grounding, pixel-level operation chains ([Pixel Reasoner]), tool use ([OpenThinkIMG](https://arxiv.org/abs/2505.08617)), visual state generation (Thinking with Generated Images), and even pure visual planning ([VPRL]).



### 3. Task Types and Representative Works



| Task Category                              | Representative Works/Models                                  |
| ------------------------------------------ | ------------------------------------------------------------ |
| **Math/Geometry**                          | [Vision-R1](https://arxiv.org/abs/2503.06749), [MINT-CoT](https://arxiv.org/abs/2506.05331), [Metis-RISE](https://arxiv.org/abs/2506.13056), [ThinkLite-VL](https://arxiv.org/abs/2504.07934), [ReVisual-R1](https://arxiv.org/abs/2506.07516), [GLM-4.1V](https://arxiv.org/abs/2507.01006), [GThinker](https://arxiv.org/abs/2506.01078) |
| **Code/Programming**                       | Omni-Thinker, MiMo-VL                                        |
| **Visual Localization/Detection/Counting** | [Perception-R1](https://arxiv.org/abs/2504.07954), [Vision-R1](https://arxiv.org/abs/2503.06749), VLM-R1, [VisionReasoner](https://arxiv.org/abs/2505.19651), [UniVG-R1](https://arxiv.org/abs/2505.14231), [GRIT](https://arxiv.org/abs/2505.15879), [SAM-R1](https://arxiv.org/abs/2505.22596), [Pixel Reasoner] |
| **Segmentation**                           | [SAM-R1](https://arxiv.org/abs/2505.22596), ReasonSeg        |
| **Charts/Documents/OCR**                   | [OpenThinkIMG](https://arxiv.org/abs/2505.08617), [VisionReasoner](https://arxiv.org/abs/2505.19651), [Seed1.5-VL](https://arxiv.org/abs/2505.07062), MiMo-VL, [GLM-4.1V](https://arxiv.org/abs/2507.01006), [DeepEyes](https://arxiv.org/abs/2506.05943) |
| **Tool Enhancement**                       | [OpenThinkIMG](https://arxiv.org/abs/2505.08617), [VisTA](https://arxiv.org/abs/2505.20289), [Pixel Reasoner], [DeepEyes](https://arxiv.org/abs/2506.05943) |
| **Visual Chain-of-Thought & Rethinking**   | [VL-Rethinker](https://arxiv.org/abs/2504.08837), [GThinker](https://arxiv.org/abs/2506.01078), [D2I](https://arxiv.org/abs/2507.06999), [OpenVLThinker](https://arxiv.org/abs/2503.17352), [Virgo](https://arxiv.org/abs/2501.01904) |
| **Planning/Pure Visual States**            | Visual Planning ([VPRL]), Vision Agent GUI ([GLM-4.1V](https://arxiv.org/abs/2507.01006), Seed1.5-VL, Kimi-VL) |
| **Emotion & Audio/Video**                  | R1-Omni, Seed1.5-VL                                          |
| **Unsupervised/Self-supervised**           | MM-UPT, [ThinkLite-VL](https://arxiv.org/abs/2504.07934), [VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132) |

> Note: For ReasonSeg, refer to the reasoning segmentation setup and data in the LISA work (CVPR 2024). See "Reference Summary" below for the PDF link.



### 4. Evolution of Reasoning Chain Modalities



1. **Pure-text CoT**: `Question -> Thinking Steps (pure text) -> Answer`.
2. **Interleaved Image-Text CoT**: Inserting images or image tokens into the text chain ([iMCoT], [GRIT](https://arxiv.org/abs/2505.15879)).
3. **Structured Visual Grounding**: The reasoning chain directly outputs structured visual markers like bounding boxes, masks, or coordinates ([UniVG-R1](https://arxiv.org/abs/2505.14231), [VisionReasoner](https://arxiv.org/abs/2505.19651), [SAM-R1](https://arxiv.org/abs/2505.22596)).
4. **Native Pixel-Level Perceptual Operations**: Using pixel-level operations like ZOOM, FRAME SELECT as reasoning steps ([Pixel Reasoner], [DeepEyes](https://arxiv.org/abs/2506.05943)).
5. **Tool/Program Externalization**: Aiding reasoning by calling external tools or executing code ([OpenThinkIMG](https://arxiv.org/abs/2505.08617), [VisTA](https://arxiv.org/abs/2505.20289), [VPRL]).
6. **Intrinsic Generated Visual Thinking**: The model generates intermediate images to represent sub-goals or for self-critique (Thinking with Generated Images).
7. **Cue-Reflection Loop**: A cyclical reasoning process of repeatedly checking and correcting based on visual cues ([GThinker](https://arxiv.org/abs/2506.01078), [VL-Rethinker](https://arxiv.org/abs/2504.08837), [D2I](https://arxiv.org/abs/2507.06999)).
8. **Cross-Modal Behavior Transfer**: Transferring "slow thinking" or behavioral patterns learned from text reasoning to visual tasks ([Open Vision Reasoner](https://arxiv.org/abs/2507.05255), [Virgo](https://arxiv.org/abs/2501.01904)).
9. **Coupling of Emotion and Multi-stream Signals**: Fusing multimodal signals like audio and video for comprehensive reasoning (R1-Omni).
10. **General Unified Multi-Domain Chaining & Planning**: Handling complex reasoning and planning across multiple domains and tasks within a unified framework ([GLM-4.1V](https://arxiv.org/abs/2507.01006), [Seed1.5-VL](https://arxiv.org/abs/2505.07062), MiMo-VL, Omni-Thinker).



### 5. Training Paradigms



- **The Debate on the Necessity of SFT**:
  - **"SFT is Harmful" View**: SFT induces models to learn pseudo-CoT; direct RL or RL followed by SFT works better ([VLAA-Thinking](https://arxiv.org/abs/2504.11468), [VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132), [Metis-RISE](https://arxiv.org/abs/2506.13056)).
  - **"SFT is Necessary" View**: High-quality SFT data is crucial for model cold-start, learning basic formats, and call syntax ([UniVG-R1](https://arxiv.org/abs/2505.14231), [SAM-R1](https://arxiv.org/abs/2505.22596), [GRIT](https://arxiv.org/abs/2505.15879), [OpenThinkIMG](https://arxiv.org/abs/2505.08617)).
  - **Compromise/Iterative Solutions**: SFT and RL can be performed iteratively ([OpenVLThinker](https://arxiv.org/abs/2503.17352)), or a multi-stage training approach can be adopted, starting with RL in the text domain and then transferring to the multimodal domain ([ReVisual-R1](https://arxiv.org/abs/2506.07516)).
- **Multi-stage/Curriculum Learning**:
  - **Difficulty Ranking**: Arranging the training sequence based on task complexity or model forgetting rate (Curr-ReFT, Omni-Thinker).
  - **Dynamic Sampling**: Adaptively selecting the difficulty of training samples (GLM-RLCS, [ThinkLite-VL](https://arxiv.org/abs/2504.07934)).
  - **Phased Training**: Dividing training into multiple stages with different objectives (e.g., common sense, multimodality, text) ([ReVisual-R1](https://arxiv.org/abs/2506.07516)).
- **Unified Multi-task RL**:
  - **Multi-Objective Reinforcement Learning (MORL)**: Simultaneously optimizing reward signals from different tasks (MiMo-VL).
  - **Mixed Rewards/Data**: Training with a mix of data and reward functions from different tasks ([Mixed-R1](https://arxiv.org/abs/2505.24164)).
  - **Weight Optimization**: Using evolutionary algorithms or other methods to automatically adjust the weights for different tasks (MoDoMoDo).
- **Unsupervised/Self-supervised**:
  - **Self-Voting/Pseudo-Labeling**: Generating pseudo-reward signals through methods like majority voting (MM-UPT).
  - **Self-Distillation**: Learning from better trajectories generated by the model itself ([Metis-RISE](https://arxiv.org/abs/2506.13056), [Virgo](https://arxiv.org/abs/2501.01904)).
  - **Trajectory Synthesis/Filtering**: Synthesizing potential reasoning paths and filtering out effective ones for training ([OpenThinkIMG](https://arxiv.org/abs/2505.08617)).



### 6. Reward Design and Signal Engineering



| Level                                    | Type                                | Specific Methods and Examples                                |
| ---------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| **I. Basic Verifiable**                  | Correctness, Spatial, Counting      | Answer Matching (Omni-Thinker), IoU ([VisionReasoner](https://arxiv.org/abs/2505.19651)), Numerical Consistency ([Perception-R1](https://arxiv.org/abs/2504.07954)), Segmentation IoU ([SAM-R1](https://arxiv.org/abs/2505.22596)). |
| **II. Structure & Format**               | JSON/Template/Syntax                | `<think> / <answer>` format reward (StepRVR), Non-repetition reward ([VisionReasoner](https://arxiv.org/abs/2505.19651)), Conditional tool reward ([DeepEyes](https://arxiv.org/abs/2506.05943)). |
| **III. Step-level/Process-level**        | Step-wise Reward/Control            | StepGRPO (StepRAR/StepRVR) (see [R1-VL](https://arxiv.org/abs/2503.12937)), Length control ([Vision-R1 PTST](https://arxiv.org/abs/2503.06749)), Forced rethinking trigger ([VL-Rethinker](https://arxiv.org/abs/2504.08837)), Cue backtracking ([GThinker](https://arxiv.org/abs/2506.01078)). |
| **IV. Complex Composite/Mixed**          | Multi-signal Weighting/Combination  | Mixed rewards + BMAS similarity ([Mixed-R1](https://arxiv.org/abs/2505.24164)), MORL (RLVR+RLHF) (MiMo-VL), Multi-domain independent rewarders ([GLM-4.1V](https://arxiv.org/abs/2507.01006)). |
| **V. Exploration Incentive**             | Curiosity/Difficulty/Anti-collapse  | Curiosity reward ([Pixel Reasoner]), Difficulty weighting ([UniVG-R1](https://arxiv.org/abs/2505.14231)), MCTS difficulty selection ([ThinkLite-VL](https://arxiv.org/abs/2504.07934)), SSR/SSB/PAD to combat advantage collapse (SSB idea in [Skywork series reports](https://arxiv.org/abs/2504.16656)). |
| **VI. Conditional/Combinatorial**        | Dependent on Specific Behavior      | Reward only if correct and tool is used ([DeepEyes](https://arxiv.org/abs/2506.05943)), Positive/negative/zero reward based on tool effectiveness ([VisTA](https://arxiv.org/abs/2505.20289)). |
| **VII. Anti-Hallucination & Robustness** | Monitoring/Penalizing Pseudo-chains | Reward entropy monitoring ([Skywork-R1V3](https://arxiv.org/abs/2504.16656)), Penalizing redundant predictions (VLM-R1 odLength), Reward for direct localization without CoT ([Perception-R1](https://arxiv.org/abs/2504.07954)). |



### 7. Data Engineering and Difficulty Management



- **Curriculum and Difficulty Metrics**: Grading data using metrics like forgetting rate (BWT), error rate, MCTS iteration count, and GPT scores ([OpenVLThinker](https://arxiv.org/abs/2503.17352)).
- **Mixed Optimization**: Balancing multi-task data through a quadratic performance proxy + CMA-ES (MoDoMoDo) or multi-stage pre-training + RL (MiMo-VL, Seed1.5-VL).
- **High-Quality CoT Construction**: Improving CoT data quality through methods like manual review ([UniVG-R1](https://arxiv.org/abs/2505.14231)), GPT-4o decomposition + filtering ([OpenThinkIMG](https://arxiv.org/abs/2505.08617)), and template synthesis ([Pixel Reasoner]).
- **Self-Distillation/Pseudo-Labeling**: Augmenting data using methods like multi-response voting (MM-UPT) and expert-enhanced self-distillation ([Metis-RISE](https://arxiv.org/abs/2506.13056)).
- **Difficulty Bias Correction**: Optimizing the learning curve through difficulty weighting (-mIoU, [UniVG-R1](https://arxiv.org/abs/2505.14231)), focusing on medium-difficulty samples ([ThinkLite-VL](https://arxiv.org/abs/2504.07934)), and progressively releasing task complexity (Curr-ReFT).



### 8. Policy Optimization and Stability Techniques



- **Base Algorithms**: GRPO / PPO / StepGRPO / DAPO (see [R1-VL](https://arxiv.org/abs/2503.12937) and [Open Vision Reasoner](https://arxiv.org/abs/2507.05255)).
- **Solutions for Advantage Collapse**:
  - **SSR (Stochastic Successive Rejection)**: Caching samples with non-zero advantage ([VL-Rethinker](https://arxiv.org/abs/2504.08837)).
  - **SSB (Selective Successive Buffer)**: Prioritizing sampling of high-value samples (see [Skywork-R1V2/3 technical reports](https://arxiv.org/abs/2504.16656)).
  - **PAD (Priority Advantage Distillation)**: Filtering zero-advantage samples and re-weighting ([ReVisual-R1](https://arxiv.org/abs/2506.07516)).
  - **Others**: Token-level normalization, asymmetric clipping ([SAM-R1](https://arxiv.org/abs/2505.22596)), dynamic KL (GRPO-D), entropy monitoring ([Skywork-R1V3](https://arxiv.org/abs/2504.16656)).
- **Mitigating Multi-task Interference**:
  - **Task Sequencing**: Training tasks in a specific order to reduce forgetting (Omni-Thinker).
  - **Dynamic Sampling**: Dynamically balancing the sampling rate of different domains (GLM RLCS).
  - **Weight Solving**: Solving for task weights through optimization algorithms ([Mixed-R1](https://arxiv.org/abs/2505.24164), MoDoMoDo).
- **Connector Re-tuning**: Freezing the ViT and LLM during the RL phase and only fine-tuning the connector to prevent knowledge drift and achieve more stable alignment ([Skywork-R1V3](https://arxiv.org/abs/2504.16656)).



### 9. Extension Layer for Tools, Pixels, Programs, and Planning



- **Tool Policy Learning**: Learning when and how to call external tools via RL ([OpenThinkIMG](https://arxiv.org/abs/2505.08617), [VisTA](https://arxiv.org/abs/2505.20289)).
- **Pixel Operations as Endogenous Tools**: Treating pixel-level operations like zoom and frame selection as the model's "intrinsic tools" and learning their usage policy via RL ([Pixel Reasoner], [DeepEyes](https://arxiv.org/abs/2506.05943)).
- **Grounding Outputs**: Directly outputting visual localizations (bounding boxes) as part of the CoT to achieve tight coupling between language and vision ([GRIT](https://arxiv.org/abs/2505.15879), [UniVG-R1](https://arxiv.org/abs/2505.14231)).
- **Program/Visual Planning**: Learning to generate a series of pure visual states or action sequences to solve planning problems via RL ([VPRL], Visual Planning).
- **Unified Perception and Reasoning**: Unifying various perception tasks like detection, segmentation, and counting within a single framework and using RL for multi-objective optimization ([VisionReasoner](https://arxiv.org/abs/2505.19651), [Perception-R1](https://arxiv.org/abs/2504.07954), [SAM-R1](https://arxiv.org/abs/2505.22596)).



### 10. Self-Reflection, Rethinking, and Thought Control



- **Forced Reflection Mechanisms**: Injecting trigger words at the end of a reasoning chain or setting specific rules to force the model to self-check and correct ([VL-Rethinker](https://arxiv.org/abs/2504.08837), [GThinker](https://arxiv.org/abs/2506.01078), [D2I](https://arxiv.org/abs/2507.06999)).
- **Trajectory Filtering and Truncation**: Removing repetitive thought loops and being wary of invalid reasoning paths from "pseudo aha" moments ([OpenVLThinker](https://arxiv.org/abs/2503.17352), [VLAA-Thinking](https://arxiv.org/abs/2504.11468)).
- **Defining Visual Reflection Behaviors**: Modeling behaviors like visual search, comparison, confirmation, and hallucination mitigation as learnable policies for the model ([Open Vision Reasoner](https://arxiv.org/abs/2507.05255), [DeepEyes](https://arxiv.org/abs/2506.05943)).



### 11. Transfer and Cross-Domain Generalization Mechanisms



- **Text → Vision Transfer**: First establishing abstract logical abilities on text data via RL, then transferring them to multimodal tasks ([Virgo](https://arxiv.org/abs/2501.01904), LMM-R1, [ReVisual-R1](https://arxiv.org/abs/2506.07516)).
- **Vision → Other Task Transfer**: Reasoning abilities learned on one visual task (e.g., geometry) can significantly improve performance on other tasks (e.g., counting) (OThink-MR1).
- **Structured → Open-ended Transfer**: Generalizing abilities learned on structured tasks to open-ended description tasks through mixed rewards (hard verification + similarity/discriminator) ([Mixed-R1](https://arxiv.org/abs/2505.24164), Omni-Thinker).



### 12. Resource Efficiency and Few-Shot Strategies



- **Extremely Low-Sample Success Cases**: GRIT (20), [SAM-R1](https://arxiv.org/abs/2505.22596) (3K), [VisionReasoner](https://arxiv.org/abs/2505.19651) (7K), [ThinkLite-VL](https://arxiv.org/abs/2504.07934) (11K), MM-UPT (0 labeled).
- **Key Strategies**:
  - **Data Efficiency**: Difficulty filtering, majority-voting pseudo-labels, offline MCTS pre-screening.
  - **Model Efficiency**: Local tuning of connectors (LoRA-like).
  - **Reward Efficiency**: Curiosity rewards to expand exploration, structured format rewards to reduce annotation needs.



### 13. Evaluation Systems and Gaps



- **Current Status**: Primarily relies on answer-level metrics (Accuracy, mAP, IoU, Pass@k), lacking evaluation of the reasoning process.
- **Major Gaps**:
  - **Process Fidelity Metrics**: Lack of standardized metrics to measure the consistency between text steps and visual evidence (Step Faithfulness, Visual Grounding Consistency).
  - **Hallucination Diagnosis**: Systematic evaluation and diagnosis of model hallucinations are still underdeveloped.
  - **Safety and Robustness**: Limited RL research on adversarial attacks, noise, GUI diversity, etc.
  - **Benchmark Fragmentation**: Evaluation benchmarks for various abilities (math, vision, tools) are independent, lacking comprehensive scenarios.



### 14. Contrasting Views and Rules of Thumb



- **"SFT vs RL" Order**:
  - **Direct RL**: Avoids the pseudo-reasoning chains introduced by SFT ([VLAA-Thinking](https://arxiv.org/abs/2504.11468)).
  - **RL → SFT**: RL stimulates potential, and SFT precisely patches up weaknesses ([Metis-RISE](https://arxiv.org/abs/2506.13056)).
  - **SFT for Cold Start**: High-quality SFT is still beneficial for stability and formatting ([UniVG-R1](https://arxiv.org/abs/2505.14231)).
- **Necessity of "Explicit CoT"**:
  - **Potentially Harmful for Perception**: For direct perceptual localization tasks, CoT might interfere with performance ([Perception-R1](https://arxiv.org/abs/2504.07954)).
  - **Beneficial for Abstract Reasoning**: For multi-step abstract reasoning like math, CoT + RL shows significant gains ([Metis-RISE](https://arxiv.org/abs/2506.13056)).
- **"Is Longer Chain Always Better?"**:
  - **Not necessarily**: There exists an optimal reasoning length window; chains that are too long or too short can harm performance ([Virgo](https://arxiv.org/abs/2501.01904), [Vision-R1 PTST](https://arxiv.org/abs/2503.06749)).
- **"Do Multi-task approaches always win?"**:
  - **Potential for Interference**: Conflicts may exist between different tasks (e.g., long-chain reasoning vs. short-output localization), which need to be addressed through curriculum learning or weight optimization (MiMo-VL, MoDoMoDo).



### 15. Classification of Representative Methods



| Category                                    | Representative Methods/Papers                                |
| ------------------------------------------- | ------------------------------------------------------------ |
| **Pure RL / Zero SFT**                      | [VisualThinker-R1-Zero](https://arxiv.org/abs/2503.05132), [DeepEyes](https://arxiv.org/abs/2506.05943), [Pixel Reasoner], MM-UPT |
| **Iterative/Cyclical Training**             | [OpenVLThinker](https://arxiv.org/abs/2503.17352), [ReVisual-R1](https://arxiv.org/abs/2506.07516), LMM-R1 |
| **Mixed Rewards/High-Dim Optimization**     | Omni-Thinker, [Mixed-R1](https://arxiv.org/abs/2505.24164), MiMo-VL, [GLM-4.1V](https://arxiv.org/abs/2507.01006), Seed1.5-VL, R1-Omni, [OpenThinkIMG](https://arxiv.org/abs/2505.08617) |
| **Difficulty Strategy/Curriculum Learning** | [ThinkLite-VL](https://arxiv.org/abs/2504.07934), Curr-ReFT, GLM RLCS, Omni-Thinker, [UniVG-R1](https://arxiv.org/abs/2505.14231), [ReVisual-R1](https://arxiv.org/abs/2506.07516) |
| **Visually-Grounded Endogenous Thinking**   | [GRIT](https://arxiv.org/abs/2505.15879), [MINT-CoT](https://arxiv.org/abs/2506.05331), [DeepEyes](https://arxiv.org/abs/2506.05943), [Pixel Reasoner], Thinking with Generated Images, [VPRL] |
| **Unified Perception Framework**            | [Perception-R1](https://arxiv.org/abs/2504.07954), [VisionReasoner](https://arxiv.org/abs/2505.19651), [SAM-R1](https://arxiv.org/abs/2505.22596), [Vision-R1](https://arxiv.org/abs/2503.06749) |
| **Reflection/Rethinking Enhancement**       | [VL-Rethinker](https://arxiv.org/abs/2504.08837), [GThinker](https://arxiv.org/abs/2506.01078), [D2I](https://arxiv.org/abs/2507.06999), [Metis-RISE](https://arxiv.org/abs/2506.13056), [Virgo](https://arxiv.org/abs/2501.01904) |



### 16. Key Technical Patterns



- **Two-Stage Differentiated Training**: Abstract logic first (text RL), followed by perceptual alignment (multimodal RL), and finally language refinement (text SFT/RL).
- **Difficulty Curve Control**: Focusing on medium-difficulty samples to maintain effective gradients, while balancing exploration and exploitation through dynamic KL or progressive reward thresholds.
- **Process-Intensive Rewards**: Providing denser supervision signals through step-level rewards and fine-grained visual grounding rewards.
- **Internalization of Visual Operations**: Internalizing external tools or operations (like zoom) into the model's learnable policies.
- **Hybrid Verification Layer**: Combining hard rules, model-based discriminators, and embedding similarity to effectively evaluate complex open-ended tasks.
- **Anti-Hallucination Mechanisms**: Suppressing hallucinations through conditional rewards, explicit grounding, and entropy monitoring.
- **Resource Efficiency**: Offline evaluation, local fine-tuning, and pseudo-label generation are key to achieving efficient training.



### 17. Current Major Bottlenecks



- **Lack of Unified Process Evaluation**: Absence of a standardized "visual evidence-to-text step" fidelity metric.
- **Reward Hacking and Overfitting**: Models easily find shortcuts in the reward function, producing formally correct but semantically empty reasoning.
- **Multi-task Interference and Negative Transfer**: Gradient directions or reward scales of different tasks may conflict.
- **Hallucination and Pseudo-Visual-References**: Models may "fabricate" details that do not exist in the image.
- **Authenticity of Self-Reflection**: It's difficult to distinguish whether a model is genuinely reflecting or merely imitating reflection templates.
- **Authenticity of Synthetic Data**: Synthetic data may carry systematic biases.
- **Scalability of Tools/Pixel Operations**: The current set of available operations is still narrow and lacks generality.



### 18. Future Directions and Research Opportunities



1. **Process-Level Trustworthy Evaluation**: Develop step-level multimodal fidelity metrics.
2. **Graph-Structured Reasoning Rewards**: Extend rewards from linear chains to directed acyclic graph (DAG) structures.
3. **Unified Visual Operation DSL**: Define a domain-specific language for visual operations, allowing RL policies to learn their composition.
4. **World Models and Active Perception**: Extend reasoning into 3D and temporal simulations.
5. **Integration of Uncertainty Estimation**: Incorporate confidence calibration into rewards, penalizing high-confidence errors.
6. **End-to-End Bootstrapping Loop**: Achieve a fully closed loop of SFT-free pure RL + self-distillation + automatic synthesis.
7. **Automated Reward Discovery**: Use learnable reward models to reduce the cost of manual design.
8. **Cross-Modal Memory and Retrieval**: Cache and reuse visual reasoning states.
9. **Green and Efficient Reasoning**: Research efficient strategies like adaptive early termination and token pruning.
10. **Privacy and Safety**: Introduce ethical and safety constraints within the RL framework.



### 19. Practical Implementation Design Recommendations



- **Data and Curriculum**: Mix verifiable tasks with open-ended tasks, rank by difficulty, focus on medium-difficulty samples, and retain failure cases for advantage replay.
- **Reward Engineering**: Build a layered reward stack (basic verification → structure → process → exploration → open-ended) and include anti-hacking strategies.
- **Optimization Strategy**: Prefer GRPO variants, monitor the advantage distribution, and use SFT cautiously for cold starts.
- **Visual Thinking and Tools**: Start with simple grounding, then gradually introduce pixel operations and lightweight tool interfaces.
- **Reflection and Review**: Enable forced reflection only for difficult samples to avoid templated responses.
- **Generalization and Stability**: Adopt a multi-stage training paradigm and use advanced optimization strategies to balance multi-task weights.
- **Monitoring Dashboard**: Track key metrics like the entropy of thought tokens, reward composition, chain length, and tool call frequency.



### 20. Conclusion



Reinforcement learning for multimodal reasoning is undergoing a profound paradigm shift, evolving from simple "text chain imitation" to a comprehensive intelligent stage that integrates "visually-grounded endogenous thinking, tool/pixel operations, and policy reflection." The core drivers of this evolution are the **structuring of reward signals**, the **visualization of reasoning modalities**, and the **dynamization of training paradigms**. Although challenges remain in process evaluation, reward hacking, and multi-task balancing, future research is clearly directed towards more credible, general, and efficient native multimodal reasoning models. This will require synergistic design across rewards, data, optimization, and thinking patterns to break through current bottlenecks.

**Thanks to https://www.zhihu.com/people/zai-zhe-teng-liang-nian-ba**