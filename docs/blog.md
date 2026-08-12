# Building a Local Texas Education Knowledge Base

The project now centers on a local, citation-first knowledge base for Texas education material. That is a better fit for policy and guidance work than fine-tuning alone because the hard requirement is knowing which source supports an answer.

The workflow is:

1. Catalog trusted sources in YAML.
2. Download or point at local copies of TEA, Texas law, Texas rules, and district policy documents.
3. Extract text from HTML, PDF, Markdown, or plain text.
4. Chunk the content with source metadata.
5. Search locally and return cited passages.
6. Use the cited passages as context for a local model when you want generated prose.

This keeps the corpus auditable. When a TEA manual changes, you update that source, rebuild the index, and answers cite the new material.

## Why Not Start With LoRA

LoRA is useful when you want a model to learn a repeated behavior: classify a request, emit a strict schema, or follow a house style. Texas education knowledge is different. Requirements and guidance change over time, and the answer needs to point back to the exact law, rule, manual, or district policy.

For this domain, the right first system is retrieval-augmented generation:

- retrieval keeps source material current,
- citations make answers reviewable,
- local storage preserves privacy,
- optional local models can generate answer drafts from the retrieved context.

Fine-tuning can come later for narrow tasks around the KB, but the knowledge itself should stay in documents.
