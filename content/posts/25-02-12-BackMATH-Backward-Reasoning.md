---
title: BackMATH - Towards Backward Reasoning for Solving Math Problems Step by Step
date: '2025-02-12T15:10:31Z'
draft: false
tags:
    - NLP
    - Transformers
    - Deep Learning
    - Chain of Thought
    - Reinforcement Learning
    - Supervised Training/Fine-Tuning
categories:
    - Research Reviews
    - Machine Learning
    - Artificial Intelligence

authors:
    - Shaowei Zhang
    - Deyi Xiong

summary: "A review of the BackMATH paper, covering its impact on training models for reasoning, key insights, and future directions."
doi: "https://aclanthology.org/2025.coling-industry.40/"
pdf: "https://aclanthology.org/2025.coling-industry.40.pdf"
---
### Introduction

This paper is trying to address two things:
>1) LLM's are bad at reasoning (specifically in complex problems).
>2) When they are able to reason, they cannot extrapolate to situations that differ from those it was trained on

To address this, they gather a variety of math problems to start fine-tuning a model using Supervised Fine-Tuning (SFT) to enforce Chain-of-Thought (CoT) reasoning, as well as finetune the model to reverse problems using backward reasoning, where the final answer becomes part of the conditions, and the model figures out missing information. Once SFT has been completed, the model is then trained using a Reinforcement Learning (RL) paradigm using two reward models (PRM and BackPRM) to score the quality of the forward and backward reasoning steps, which is enforced using Proximal Policy Optimization (PPO), which refines the model based on feedback from the reward models defined.

This goal is to enforce both chain-of-thought reasoning, and the ability to restructure conditions and missing information. By doing so, they create a model that is capable of engaging with a complex problem through CoT reasoning, as well as adapt its ability to reason to more novel situations where the specific information that is missing and its conditions are not the same as examples it has been fine-tuned on i.e. it should in theory become more flexible in it's problem-solving capabilities without relying strictly on memorization of CoT patterns.

---

### Key Contributions & Insights

>The results show that **using SFT and RL produces models that are more capable at answering questions** from a given testset of math questions than existing models. This may be caused by an **increase in the adapability and flexibility of models** that are trained using this paradigm in answering complex problems and engaging in efficient and structured problem solving.

However, DeepSeek's R1 model has a similar approach of RL, however instead of explicitly training the model with CoT examples and structured backward reasoning, DeepSeek directly uses RL to encourage step-by-step reasoning as an emergent behavior. This means that CoT reasoning naturally appears as a result of optimizing the reward function, rather than being explicitly trained through labeled CoT examples, which may lead to CoT reasoning being more deeply embedded within the weights of the model, and therefore be more likely to engage with CoT reasoning without being prompted to do so i.e. instead of requiring CoT-style inputs, the model may naturally adopt CoT reasoning across a broader range of problems as a defacto approach. Additionally, if a model implicitly learns structured reasoning patterns, it may be more likely to discover novel reasoning strategies on its own, some of which may be more effective than the ones we already know!

Although DeepSeek's approach may provide more general problem solving, the understanding of specific problems and concepts may be better captured via SFT and RL of the forward and backward problems as it provides direct examples of bidirectional learning and may encourage problem formulation to occur in a more systematic way. This is something that has been seen in DeepSeek where the CoT reasoning may not initially be obvious to the user/be foreign in its approach (non-readable to a human), yet the answer is correct.

---

### Evaluation of the Paper

One of the issues with this paper is that there is a heavy reliance and dependence on manually curated backward reasoning data. This limits scalability and requires a human in the loop, which raises questions of accuracy/human-error within the dataset.

In comparison to DeepSeek's R1 model, BackMATH uses PPO, which is more computationally expensive than GPRO used by DeepSeek, which raises questions about how adoptable and scalable its approach is in comparison to DeepSeek's research.

The paper also assumes that training on backward reasoning helps models generalize to novel problems, which may not be fully true, and has not yet been fully proven. The model only improves by 1.6% on GSM8K and 2.1% on MATH, which are not significant in regards their comparison to competitor's models. Additionally, real-world problems are often less defined and require an additional step of problem formulation for accurate complex problem solving, which the SFT examples lack. A broader evaluation of its problem-solving capabilities is required before proving this approach's effectiveness in generalizaing to larger/differing domains beyond the training data.

Finally, this paper primarily highlights successes but does not deeply analyze when and why the model still fails. So although, back-reasoning is a novel approach in its attempts to produce more effective reasoning, its lacks transparency/in-depth analysis of where the model may fall short and therefore does not explore when backward reasoning is not helpful for a model.

---

### Final Thoughts

*BackMATH* is well-motivated and approach to improving mathematical reasoning. It demonstrates that *backward reasoning training can enhance problem-solving flexibility*. However, it has *scalability issues, computational inefficiencies, and uncertain generalization benefits*. Future research should explore hybrid approaches combining self-supervised learning of structured reasoning training with emergent learning to build LLMs that are both interpretable and highly generalizable.

Additionally, the comparison between DeepSeek's approach and BackMATH's approach highlights potentially differing approaches to reasoning and rationality. It also raises the following questions:
1) ***Languages, Reasoning and Rationality***
    - Are languages that we use limiting our current approach to reasoning and rationality
        - Could AI models develop more efficient or structured reasoning methods that differ from human linguistic reasoning?
        - Would non-verbal or symbolic reasoning be a more optimal approach for AI reasoning?
    - Will there be a new emergent language for reasoning that develop?
        - Could neural networks evolve their own internal structures for logic and deduction, distinct from human cognition?
        - Would such a system resemble mathematical logic, symbolic reasoning, or a novel form of abstract representation?
2) ***Trust and Interpretability of Opaque AI Reasoning***
    - How can we trust opaque reasoning processes that produce correct outputs but are not interpretable?
        - If a model like DeepSeek reasons in an unknown or non-human-readable way, how do we validate its correctness?
        - Does the emergence of "black-box" AI reasoning present a fundamental limit to AI alignment?
    - How to illuminate the opaque reasoning processes
        - What interpretability methods (e.g., activation visualization, attribution analysis, mechanistic interpretability) can be applied?
        - Are there patterns in AI-generated reasoning that can be reverse-engineered into human-understandable logic?
3) ***Ethical Fine-Tuning of AI Reasoning***
    - How to safely fine-tune these processes to remain ethical will maintaining its efficacy 
        - What safeguards are needed to prevent emergent reasoning from diverging from human moral frameworks?
        - Would we use Reinforcement Learning to develop ethics that align with human ethics
        - Can we use Supervised Training/Fine-tuning to achieve the same results?
4) ***Knowledge Representation in AI***
    - Do the weights of DeepSeek provide novel insights into our understanding of knowledge representations
        - Could analyzing DeepSeek’s internal representations help us understand how AI models encode knowledge?
        - Do AI systems create novel cognitive structures that challenge our existing understanding of epistemology and knowledge storage?
