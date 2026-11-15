
# Data Sources

This project uses publicly available Intel technical documentation as the knowledge base for the GPU-Dev Assistant RAG system.

## Documents

| Document | Version | Pages | URL | License / Terms |
|---|---|---|---|---|
| Intel oneAPI GPU Optimization Guide | 2025.0 | 405 | [Official Documentation](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/introduction.html) | Public Intel documentation; not redistributed |
| Intel oneAPI Programming Guide | 2025.1 | 129 | [Official PDF](https://cdrdv2-public.intel.com/848694/oneapi_programming-guide_2025.1-771723-848694.pdf) | Public Intel documentation; not redistributed |


## Data Usage

- **Source:** Official Intel documentation.
- **Purpose:** Retrieval-Augmented Generation (RAG) for technical question answering.
- **Storage:** Source PDFs are stored locally and excluded from version control.
- **Redistribution:** Original documents are not redistributed through this repository.
- **Attribution:** Generated responses reference the source document and relevant page numbers when available.

## Reproducibility

- Document versions are recorded above.
- Source URLs are provided for downloading the documents.
- The ingestion pipeline processes locally downloaded PDFs.
- The `current` documentation URL may change over time; use the recorded document version for reproducibility.
