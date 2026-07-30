# Documentation

| | |
|---|---|
| [architecture.md](architecture.md) | The containers, the request path, the pipeline graph, which agents call the model, and where the leakage guarantee is enforced |
| [data-model.md](data-model.md) | The seven tables, what each is for, and the enums |
| [api.md](api.md) | All 25 endpoints, plus the things the OpenAPI schema cannot tell you |
| [RUNBOOK.md](RUNBOOK.md) | Cold start, measured demo timings, and the failure modes that look like bugs |
| [related-work.md](related-work.md) | Positioning against the similarly-named CHI '21 system by Wang et al. |

The interactive API docs are at **http://localhost:8000/docs** once the stack is
up, and are generated from the code.

## A note on how these were written

The diagrams are derived from the running system, not from the specification:
the pipeline graph from `PIPELINE_NODES` and `OPTIONAL_NODES`, the tables from
`information_schema` on a migrated database, the endpoints from the live OpenAPI
document, and the timings from real runs against a live model rather than
estimates.

Where the code and the spec disagree — pgvector rather than ChromaDB, Gemini
rather than Gemma — these describe the code and say why the difference exists.
