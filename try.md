# Lime Code Review
## Short introduction
  First make a short introduction for LIME. LIME(Local Interpretable Model-Agnostic) is a class_agnostic model which is to explain the classifer of a black box model. It explains what makes and affects the model implementing classification in a human-readable way and can explain one individual prediction once. The basic methodology is to generate a new dataset consisting of some perturbed data around the target prediction and try to train a relatively simple model(most are linear) to obtain the same or similar results with the original model. More information can be found in https://arxiv.org/abs/1602.04938
## Methodology and algorithm
