# Related Work / Positioning

> Draft positioning paragraph for the final report. Frames this project against
> the similarly-named CHI '21 system by Wang et al. Renumber the citation and
> replace the `§[Future Work]` placeholder to match the report's structure.

The name *AutoDS* is shared with Wang et al. [1], whose CHI '21 system of the
same name automates data-science model building over a classical AutoML backend
(Bayesian and joint pipeline optimization). Their contribution, however, is
empirical rather than architectural: a between-subjects study of thirty
professional data scientists that found AutoML-assisted users produced
objectively better models (0.919 vs. 0.899 ROC-AUC) with fewer errors, yet
trusted those models *less* than ones they had written by hand (2.4 vs. 3.3 on a
five-point scale). The authors attribute this "better-but-less-trusted" gap to
opacity, and call for systems that explain not merely *what* decisions were made
but *why*. This project takes that finding as its design brief. Where Wang et
al. study an AutoML tool, this work *builds* one, and it departs from their
backend in two ways: rather than searching a pipeline space with an optimizer,
it uses a multi-agent LLM architecture in which language models emit small,
schema-validated decisions that deterministic code executes ("LLM decides, code
executes"); and rather than surfacing only a model leaderboard, it addresses the
trust gap directly through SHAP-based feature attribution, a self-critique
agent, a generated report, and a retrieval-grounded chat interface over its own
outputs. The human-in-the-loop checkpoint here — the user confirms the inferred
target, task type, and PII before the pipeline runs — independently mirrors the
configuration-approval step their study validated, while the explainability and
conversational layers respond to the transparency needs their study left open. A
notable feature of their prototype not reproduced here, downloadable
human-readable notebooks, is discussed as future work (§[Future Work]).

## Reference

[1] Wang, D., Andres, J., Weisz, J., Oduor, E., and Dugan, C. *AutoDS: Towards
Human-Centered Automation of Data Science.* CHI '21, Yokohama, Japan.
https://doi.org/10.1145/3411764.3445526
