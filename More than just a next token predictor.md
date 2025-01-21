![Post-Training](https://github.com/user-attachments/assets/c47b52e0-721e-4b9f-b85d-62c503009d3e)
Many people think that the modern LLM models are just trained to predict the next token. 

This is not an incorrect statement. But it is an incomplete statement when talking about the modern LLMs. 

The term training can often be used in place of pre-training, finetuning, and post- training, which refer to different training phases:


**Pre-training:**
Pre-training refers to training a model from scratch—the model weights are randomly initialized. For LLMs, pre-training often involves training a model for text completion.


**Post-training**

Self-supervision optimizes the model for text completion, not conversations. 
If you input “How to make pizza” into the model, the model will continue to complete this sentence, as the model has no concept that this is sup‐ posed to be a conversation.
We know that a model mimics its training data. 

To encourage a model to generate the appropriate responses, you can show examples of appropriate responses. Such examples follow the format (prompt, response) and are called demonstration data.

Some people refer to this process as behavior cloning: you demonstrate how the model should behave, and the model clones this behavior.
![[Screenshot 2025-01-19 at 2.25.01 PM.png]]

Post-training starts with a pre-trained model.


Every model’s post-training is different. However, in general, post-training consists of two steps:
1. Supervised finetuning (SFT): Finetune the pre-trained model on high-quality instruction data to optimize models for conversations instead of completion.
2. Preference finetuning: Further finetune the model to output responses that align with human preference. Preference finetuning is typically done with **reinforcement learning** (RL). Techniques for preference finetuning include RLHF ( used by GPT-3.5 and Llama 2), DPO (Direct Preference Optimization), used by Llama 3), nd reinforcement learning from AI feedback (RLAIF) (potentially used by Claude)

For language-based foundation models, pre-training optimizes token-level quality, where the model is trained to predict the next token accurately. However, users don’t care about token-level quality—they care about the quality of the entire response. Post-training, in general, optimizes the model to generate responses that users prefer. Some people compare pre-training to reading to acquire knowledge, while post- training is like learning how to use that knowledge.



## Preference Finetuning
A model that can assist users in achieving great things can also assist users in achieving terrible things. Demonstration data teaches the model to have a conversation but doesn’t teach the model what kind of conversations it should have.

Had the goal been simple, the solution could’ve been elegant. However, given the ambitious nature of the goal, the solution we have today is complicated.  The earliest successful preference finetuning algorithm, which is still popular today, is RLHF. RLHF consists of two parts:
1. Train a reward model that scores the foundation model’s outputs
2. Optimize the foundation model to generate responses for which the reward model will give maximal scores.
While RLHF is still used today, newer approaches like DPO (Rafailov et al., 2023) are gaining traction. For example, Meta switched from RLHF for Llama 2 to DPO for Llama 3 to reduce complexity.


**Reward model**
RLHF relies on a reward model.
