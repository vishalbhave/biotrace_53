# Welcome to BioTrace Documentation

BioTrace is a pipeline and toolset for extracting, verifying, and integrating biodiversity data. It leverages modern Natural Language Processing (NLP), hierarchical chunking, knowledge graphs, and external API integrations to build a robust system for tracking and verifying biological entities, taxonomic names, and geographic locations.

## Key Features

- **Advanced Chunking:** Implements structural, scientific, and hierarchical chunking to process complex documents effectively.
- **Entity Extraction (NER):** Uses custom Hugging Face models and rule-based extraction to identify taxonomy, morphology, traits, and localities.
- **Knowledge Graph Integration:** Builds a spatio-temporal knowledge graph to connect biological traits, taxonomic hierarchies, and geographic data.
- **Verification Pipelines:** Integrates with external APIs (GBIF, GNA, WoRMS, Catalogue of Life) to verify species names and taxonomy.
- **Interactive Staging & UI:** Provides human-in-the-loop (HITL) interfaces and UI agents to refine and correct geocoding and staging data.

---

## Getting Started

To explore the documentation:

- Check out the [**Architecture**](architecture.md) for a high-level view of the system components.
- Dive into the [**Data Pipelines**](pipelines.md) to understand how data flows through BioTrace.
- Review the [**API Reference**](#) for detailed documentation of individual modules generated directly from the source code.
