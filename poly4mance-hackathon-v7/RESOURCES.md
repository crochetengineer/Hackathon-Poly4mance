# Approaches worth knowing about

There is no expected solution path and the marks do not reward one. Below are six routes, with
one or two key references each. **Read the one or two things under your chosen route properly
rather than skimming all six.** Some existing work will do a large part of this challenge for
you — that is allowed and sensible. The marks are in the reasoning, not the output: lean on a
package without understanding it and you get a ranking you cannot defend, an interval you
cannot explain, and a campaign designed by someone else's objective function.

Suggested order: skim 1 and 2 first (they frame the problem), then go deep on whichever of 3–6
fits your team.

## 1. Classical model-based design of experiments (a good default)

Fit each candidate by weighted least squares / maximum likelihood, rank by an information
criterion (AIC/BIC), then choose the next experiment to **maximise the divergence between the
surviving models' predictions**. This is exactly the shape of this challenge.

- Buzzi-Ferraris & Manenti, *Kinetic Models Analysis* (2009) — model discrimination design
  criteria. 
- Franceschini & Macchietto, "Model-based design of experiments for parameter precision"
  (*Chem. Eng. Sci.*, 2008). Open access via many institutions.

## 2. Bayesian inference and model selection

Put priors on the rate parameters, compute a posterior per model, and compare models by
marginal likelihood / posterior model probability. Handles the censored Q data naturally
(left-censored likelihood) and gives calibrated intervals almost for free.

- Gelman et al., *Bayesian Data Analysis* (3rd ed.), chapters on model comparison and
  censored data. 
- The Stan / PyMC user guides on ODE models and censoring (documentation, open access).

## 3. Symbolic regression and automated model discovery

Let the data propose the rate law. Useful as a cross-check on the closed candidate set — if
symbolic regression keeps rediscovering an `[I][A]` term you cannot see directly, that is a
signal.

- Brunton, Proctor & Kutz, "SINDy: sparse identification of nonlinear dynamics"
  (*PNAS*, 2016). Open access.

## 4. Gaussian processes and hybrid models

Fit a GP (or a mechanistic + GP-residual hybrid) to the observable species to denoise and
interpolate, then use the smoothed profiles to constrain mechanistic fits. Good for the noisy
trace channel; weak on its own for discrimination.

- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006). Free online.

## 5. Training a classifier on simulated data

Simulate all six candidates across the design space, add the observation model, and train a
classifier to map (profile -> mechanism). Fast to evaluate and a natural way to ask "which
experiment would most sharpen the classifier's decision boundary?"

- Quaglio et al., "An artificial neural network approach to recognise kinetic models from
  experimental data" (*Chem. Eng. Sci.*, 2020).

## 6. LLM-assisted discovery

Use a language model to propose mechanisms, sanity-check stoichiometry, or draft the
discrimination argument — then verify everything against the data yourself.

- Recent work on LLM-proposed reaction mechanisms (e.g. SKAI and successors); treat outputs
  as hypotheses to test, never as answers.

---

AI assistance is allowed and expected, with no penalty. You will be asked about your own code
and reasoning when you present, so make sure you can talk through what you submit.
