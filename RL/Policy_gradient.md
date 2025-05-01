Gradient: 
- its a vector of partial derivatives indicating direction of the steepest ascen
- Magnitude (length) indicating how steply the function increaes
The gradient captures both the direction and the rate of change.


In calculus class given a closed-form equation  f(x, y) = 3x^2 + 4y^2 .
You compute partial derivatives directly using calculus.

In reinforcement learning or machine learning:
You often do not have a neat closed-form function explicitly.

Instead, you have:
A model f_{\theta}(x) or policy \pi_{\theta}(a|s) with concrete numeric parameters \theta.
A dataset or environment samples, each giving you numeric outcomes (rewards, losses).


	“How can we compute gradients (derivatives) when we don’t have the explicit mathematical function, just numbers from samples?”


intuition
Imagine you’re testing the sensitivity of a complicated machine. You don’t know the internal equations explicitly—but you have knobs (parameters \theta) and an output (reward or loss).

How would you find the gradient (direction of improvement)?
	•	Move the knobs slightly (tiny changes in parameters).
	•	Observe how the output changes numerically.
	•	The ratio of output change (reward or loss) over parameter change is an approximation of the derivative.


This is exactly what the gradient is in ML:
\text{Gradient} \approx \frac{\text{change in reward or loss}}{\text{tiny change in parameters}}
You don’t need the explicit equation—only small nudges (parameter perturbations) and observations of resulting outcomes.


Suppose a policy with parameter \theta = 2.0:
	•	Current reward:  R(\theta) = 10.0 
	•	Slightly change parameter: \theta = 2.001, and now the reward  R(2.001) = 10.005 .

Then the gradient at \theta = 2.0 is approximately:
\frac{R(2.001) - R(2.0)}{2.001 - 2.0} = \frac{10.005 - 10.0}{0.001} = 5.0
This numerical calculation is your gradient—no explicit formula needed!

Classical ML (supervised learning):
	•	Data usually means pairs of (input, ground truth labels).
	•	You compute loss using model predictions vs. these labels, then take gradients w.r.t. model parameters to reduce error.

Reinforcement learning (Policy Gradient):
	•	Data isn’t labeled pairs—it’s experience (state-action-reward sequences).
	•	The “output” or “data” here refers specifically to rewards you observe after taking actions, not external labels.
	•	Gradients are computed by observing how small changes in policy parameters affect expected reward.
Supervised ML is like training a student by telling them explicitly what the correct answers are (labeled data).
	•	RL (Policy Gradient) is training by giving a student feedback (“Good job!” or “Try differently!”) without explicitly saying what’s correct—just numerical signals (rewards).


Two views of gradients:
- mathematical view the formuals are calculated by algebraic manipulcaiton
- Numerical view, we approximate using finite differences

The true defintion of a derivative (and gradient) is fundamentally numerical (limit-based).
Its defined as a limit:


so in th