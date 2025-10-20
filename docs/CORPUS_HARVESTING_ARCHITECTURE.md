# Corpus Harvesting Architecture: Representative Sample Collection for Baseline Distributions

**Version:** 1.0  
**Purpose:** Systematic acquisition of diverse, high-quality human-written text for statistical baseline establishment  
**Audience:** ML Engineers, Data Scientists, System Architects  
**Integration:** SpecHO Statistical Validator (Component 5)

---

## EXECUTIVE SUMMARY

Statistical watermark detection requires robust baseline distributions derived from representative samples of genuine human writing. This architecture document describes a comprehensive system for harvesting, validating, and curating documents across multiple domains, sources, and quality tiers to establish statistically sound reference distributions.

**Core Principle**: Representative sampling requires both breadth (diverse sources) and depth (quality validation) to capture the true distribution of human writing patterns while excluding contamination from AI-generated content.

**Key Insight**: The quality of baseline distributions directly determines detection accuracy. A poorly sampled corpus leads to systematic bias in z-scores and confidence estimates, resulting in high false positive and false negative rates. A well-constructed, representative corpus enables precise statistical validation with known confidence bounds.

---

## ARCHITECTURAL OVERVIEW

### Design Philosophy

The corpus harvesting architecture embodies three fundamental principles that guide all design decisions. First is statistical representativeness. The corpus must capture the natural variation in human writing across domains, styles, authors, topics, and contexts. This requires systematic sampling strategies rather than convenience sampling. Second is provenance verification. Every document in the corpus must be verifiably human-written, with clear evidence of authorship and creation date that predates widespread LLM availability. Third is progressive automation. The system begins with manual curation and human validation, then progressively automates quality control as validation patterns become clear and reliable.

These principles reflect the dual nature of the harvesting challenge. On one hand, you need scale to achieve statistical power (hundreds to thousands of documents). On the other hand, you need quality control to ensure corpus integrity (zero tolerance for AI contamination). The architecture balances these competing demands through a multi-tiered approach that starts conservative and becomes more aggressive as confidence grows.

### System Context

The harvesting system integrates with your existing SpecHO pipeline at the Statistical Validator component (Component 5). The BaselineCorpusProcessor you implemented in Session 7 consumes the harvested corpus, processes each document through the full detection pipeline (Preprocessor, Clause Identifier, Echo Engine, Scoring Module), and generates the baseline statistics (mean, standard deviation, score distribution). These statistics then enable the ZScoreCalculator and ConfidenceConverter to transform raw document scores into interpretable confidence levels.

The harvesting system operates upstream from the detection pipeline. Its output is a curated directory structure containing validated human-written text files, organized by domain and quality tier, accompanied by comprehensive metadata that enables downstream statistical analysis. The system does not perform watermark detection itself but rather focuses on corpus construction according to principled sampling strategies.

### Component Architecture

The harvesting system consists of six major components that work together in a coordinated pipeline. The Source Registry maintains an authoritative list of document sources with associated metadata (domain, quality tier, harvesting strategy). The Harvesting Engine executes domain-specific collection strategies, interfacing with APIs, web scrapers, and data repositories. The Quality Filter applies automated checks to identify and exclude low-quality or potentially AI-generated content. The Validation Queue manages human review workflows for documents that require manual verification. The Metadata Store tracks provenance, quality metrics, and processing history for every harvested document. The Corpus Builder assembles validated documents into organized directory structures ready for baseline processing.

These components are designed for incremental deployment. You begin with manual document collection and simple file organization (Tier 1), then add automated harvesting and basic filters (Tier 2), and finally implement sophisticated quality control and continuous corpus updates (Tier 3). This staged approach allows you to establish your initial baseline quickly while building toward a production-grade harvesting infrastructure.

---

## SAMPLING STRATEGY

### Theoretical Foundation

The goal of corpus construction is to create a sample that accurately represents the population of human-written text. In statistical terms, you want your sample distribution to match the population distribution with known error bounds. This requires addressing three sources of sampling bias: source bias (over-representing certain origins), domain bias (over-representing certain topics or styles), and temporal bias (over-representing certain time periods).

Random sampling would theoretically address these biases if you had access to the full population of human text and could sample uniformly. In practice, you don't have such access. Instead, you must use stratified sampling, deliberately selecting documents across multiple strata (domains, sources, time periods) to ensure adequate representation of each. The stratification strategy determines how diverse and representative your baseline becomes.

### Stratification Dimensions

The corpus is stratified across four key dimensions. Domain stratification ensures representation across text types: news articles, academic papers, fiction, technical documentation, informal communication, and professional writing. Each domain exhibits different linguistic characteristics (formality, structure, vocabulary, clause complexity) that affect baseline statistics. Source stratification ensures diversity in authorship and publication venue, preventing any single author or publication from dominating the corpus. Temporal stratification balances documents across time periods, with emphasis on pre-LLM content (before late 2022) while including recent human-verified content. Quality stratification distinguishes between highly edited professional writing, lightly edited content, and informal unedited text, as editing intensity affects linguistic patterns.

The stratification proportions should reflect your target detection scenarios. If you primarily detect watermarks in academic writing, weight your corpus toward academic sources. If you need general-purpose detection across all domains, use balanced proportions that give each domain meaningful representation without over-emphasizing any single domain.

### Tier-Based Sampling Approach

**Tier 1: Foundation Corpus** (50-200 documents, 1-4 weeks)

The foundation corpus establishes your initial baseline using conservative, high-confidence sources. This tier prioritizes quality and verifiability over scale. You manually select documents from known-good sources with clear provenance, focusing on pre-LLM content (published before 2022) from reputable sources. The goal is to quickly establish a working baseline that enables initial system validation and detection experiments.

Recommended source mix for Tier 1: News articles (30%) from established outlets like New York Times, BBC, Reuters, with publication dates between 2015-2021. Academic papers (25%) from pre-print servers or journals, preferably with LaTeX source files that indicate human authorship. Classic fiction (15%) from Project Gutenberg or similar repositories containing verified public domain works. Technical documentation (20%) from official software project documentation written by known human contributors. Professional writing (10%) such as reports, white papers, or corporate communications with verified human authorship.

Each document undergoes manual review to verify language quality, check for obvious AI patterns (even in older documents, as some may have been edited with AI assistance), confirm appropriate length (100+ words for reliable processing), and validate domain classification. You maintain a simple spreadsheet or JSON file tracking document ID, source, domain, publication date, word count, and validation status.

**Tier 2: Production Corpus** (500-2000 documents, 1-3 months)

The production corpus extends your baseline with semi-automated harvesting and filtering. This tier introduces programmatic collection while maintaining rigorous quality control. You implement automated harvesters for major sources, apply statistical filters to detect potential AI contamination, and use stratified sampling to ensure balanced domain representation.

Harvesting strategies by domain: For news content, you use newspaper APIs (Guardian, New York Times) with date filters for pre-2022 articles and RSS feeds from verified human columnists. For academic content, you access arXiv API for pre-prints with specified date ranges and PubMed for biomedical literature with MeSH term filtering. For fiction, you harvest Project Gutenberg bulk downloads and Archive.org public domain texts. For technical content, you collect GitHub documentation files (with commit history showing human authors) and Stack Overflow answers (from high-reputation users, pre-2022). For professional writing, you gather corporate blogs (with verified author profiles) and think tank reports from established institutions.

Quality filtering at this tier includes automated checks for minimum length requirements (100+ words), language detection to confirm English, encoding validation to ensure clean UTF-8, duplicate detection to prevent redundant samples, and AI pattern screening using simple heuristics (repetitive structures, unusual token distributions, known GPT signatures). Documents passing automated filters enter a lightweight review queue where human validators perform spot checks on 10-20% of samples to verify filter effectiveness.

You implement statistical monitoring to track corpus balance across strata, identify under-represented domains, and detect quality drift over time. The metadata store expands to include automated quality scores, harvesting timestamps, filter decisions, and sample weights for stratified analysis.

**Tier 3: Continuous Corpus** (ongoing, incremental updates)

The continuous corpus implements production-grade infrastructure for sustained baseline maintenance and improvement. This tier focuses on automation, scalability, and adaptive sampling as language and detection requirements evolve. You build fully automated pipelines, implement online baseline updates, and develop sophisticated contamination detection.

Advanced features include domain-specific baseline maintenance (separate distributions for news, academic, fiction), temporal baseline tracking to capture language evolution, adversarial filtering to detect sophisticated AI generation, active learning to identify documents that improve baseline diversity, and continuous validation through automated outlier detection and human audit sampling.

The harvesting infrastructure scales to handle thousands of documents per month, with distributed processing, incremental updates to baseline statistics, versioned baseline releases (enabling reproducible detection), and integration with production detection workflows. You establish feedback loops where detection performance metrics inform corpus improvement priorities, identifying domains or document types that need better representation.

---

## SOURCE REGISTRY

### Registry Architecture

The Source Registry serves as the authoritative catalog of all document sources, their harvesting configurations, and quality metadata. This centralized registry enables consistent corpus management, reproducible harvesting runs, and systematic quality tracking. The registry is implemented as a structured data store (JSON, YAML, or database) that separates source definitions from harvesting logic.

Each source entry contains identification metadata (unique source ID, human-readable name, description), harvesting configuration (source type such as API, web scraping, file repository, access credentials or API keys, rate limiting parameters, retry policies), stratification metadata (primary domain classification, secondary domains if applicable, expected quality tier, temporal coverage), and quality metadata (known reliability score, contamination risk assessment, manual review requirements, success rate history).

The registry supports hierarchical organization, allowing source groups such as "News:NYTimes" containing sub-sources for different sections (Politics, Technology, Opinion), "Academic:ArXiv" containing sub-sources for different categories (cs.AI, cs.CL, physics), and "GitHub:Documentation" containing sub-sources for major projects. This hierarchy enables fine-grained control over sampling strategies while maintaining organizational clarity.

### Source Classification

Sources are classified along three primary axes that determine harvesting and validation strategies. The quality axis ranges from High (professionally edited, institutional backing, known human authors) through Medium (community-edited, verified contributors, possible minor quality issues) to Low (user-generated, minimal moderation, requires heavy filtering). The automation axis ranges from Automated (API access, reliable metadata, minimal manual intervention) through Semi-automated (requires parsing, selective harvesting, moderate manual review) to Manual (hand-selected documents, full manual validation). The risk axis ranges from Low Risk (pre-LLM content, strong provenance, verified authorship) through Medium Risk (recent content, weaker provenance, requires validation) to High Risk (modern content, unclear authorship, heavy filtering required).

High-quality, automated, low-risk sources form your core harvesting targets for Tier 1 and Tier 2. These sources provide reliable, high-throughput document collection with minimal quality concerns. Medium-quality sources require filtering and selective sampling but can significantly expand corpus diversity. Low-quality or high-risk sources are generally avoided in Tier 1-2 but may be included in Tier 3 with sophisticated filtering and validation.

### Example Source Definitions

```json
{
  "sources": [
    {
      "id": "nytimes_archive_2015_2021",
      "name": "New York Times Archive (2015-2021)",
      "type": "api",
      "domain": "news",
      "quality_tier": "high",
      "risk_level": "low",
      "config": {
        "api_endpoint": "https://api.nytimes.com/svc/archive/v1",
        "date_range": {"start": "2015-01-01", "end": "2021-12-31"},
        "rate_limit": "4000/day",
        "sections": ["technology", "science", "opinion", "world"]
      },
      "validation": {
        "automated_checks": ["length", "language", "encoding"],
        "manual_review_rate": 0.05
      }
    },
    {
      "id": "arxiv_cs_pre2022",
      "name": "arXiv Computer Science (pre-2022)",
      "type": "api",
      "domain": "academic",
      "quality_tier": "high",
      "risk_level": "low",
      "config": {
        "api_endpoint": "http://export.arxiv.org/api/query",
        "categories": ["cs.AI", "cs.CL", "cs.LG", "cs.CV"],
        "date_range": {"start": "2015-01-01", "end": "2021-12-31"},
        "format": "pdf",
        "max_per_category": 500
      },
      "validation": {
        "automated_checks": ["length", "language", "pdf_extraction"],
        "manual_review_rate": 0.10
      }
    },
    {
      "id": "gutenberg_fiction",
      "name": "Project Gutenberg Fiction",
      "type": "bulk_download",
      "domain": "fiction",
      "quality_tier": "high",
      "risk_level": "low",
      "config": {
        "base_url": "https://www.gutenberg.org/",
        "catalog_file": "gutenberg_catalog.csv",
        "filter": {"language": "en", "type": "Text"},
        "format": "txt"
      },
      "validation": {
        "automated_checks": ["length", "encoding"],
        "manual_review_rate": 0.02
      }
    },
    {
      "id": "github_docs_major_projects",
      "name": "GitHub Documentation (Major Projects)",
      "type": "git_clone",
      "domain": "technical",
      "quality_tier": "medium",
      "risk_level": "low",
      "config": {
        "projects": [
          {"repo": "tensorflow/tensorflow", "docs_path": "docs/"},
          {"repo": "pytorch/pytorch", "docs_path": "docs/source/"},
          {"repo": "django/django", "docs_path": "docs/"}
        ],
        "date_range": {"start": "2015-01-01", "end": "2021-12-31"},
        "file_patterns": ["*.md", "*.rst"]
      },
      "validation": {
        "automated_checks": ["length", "language", "git_history"],
        "manual_review_rate": 0.15
      }
    }
  ]
}
```

---

## HARVESTING ENGINE

### Engine Architecture

The Harvesting Engine implements domain-specific collection strategies defined in the Source Registry. The engine operates as a modular plugin system where each source type (API, web scraper, file repository, git clone) has a corresponding harvester plugin that implements a standard interface. This architecture enables easy extension to new source types while maintaining consistent error handling, logging, and rate limiting.

The core engine provides orchestration services that schedule harvesting jobs across multiple sources, manage rate limits and retry logic, handle authentication and credential rotation, implement progress tracking and resumption, log all harvesting activities, and store raw documents in a staging area. Each harvester plugin implements source-specific logic for authentication, pagination, format conversion, metadata extraction, and error recovery.

The engine supports three execution modes that match different operational needs. Batch mode processes entire source catalogs (suitable for initial corpus construction), incremental mode harvests only new documents since last run (suitable for corpus maintenance), and targeted mode collects specific document types or time ranges (suitable for filling gaps in stratification). All modes produce consistent output formats and metadata structures.

### Harvester Implementations

**API-Based Harvesters**

API harvesters interface with structured data sources through REST APIs. They handle authentication (API keys, OAuth tokens), implement pagination through result sets, respect rate limits (with exponential backoff), parse structured responses (JSON, XML), extract text content and metadata, and handle API-specific errors (quota exceeded, malformed queries).

For news APIs like New York Times, the harvester requests articles by date range and section, extracts article body text and metadata, filters by word count and language, downloads and stores article content, and logs API usage for quota management. For academic APIs like arXiv, the harvester queries by category and date, downloads PDF or LaTeX sources, extracts text from PDFs (using PyPDF2 or pdfplumber), parses LaTeX to plain text, and extracts abstracts and body text.

**Web Scraping Harvesters**

Web scrapers extract content from HTML pages using frameworks like Scrapy or Beautiful Soup. They implement polite crawling (respecting robots.txt, reasonable delays), handle dynamic content (JavaScript rendering with Selenium or Playwright), extract main content (using readability libraries), filter navigation and boilerplate, and store both raw HTML and extracted text.

Scraping requires careful design to avoid legal and ethical issues. You target only publicly accessible content, respect website terms of service, implement reasonable rate limiting (1-5 requests per second), identify your crawler in user-agent strings, and maintain an exclusion list for sites that prohibit scraping. For many sources, official APIs provide better alternatives to scraping.

**Repository Harvesters**

Repository harvesters access bulk datasets from archives like Project Gutenberg, Common Crawl, or institutional repositories. They download catalog metadata, filter by language and type, download bulk archives (tar, zip), extract individual documents, and validate file formats and encodings.

For Project Gutenberg, the harvester downloads the catalog CSV, filters for English fiction and non-fiction, downloads plain text versions, validates UTF-8 encoding, and organizes by author and publication date. For GitHub repositories, the harvester clones specific repositories (or uses GitHub API), extracts documentation files (matching file patterns), checks git history to verify human authorship (commit messages, author metadata), and filters to pre-2022 commits.

### Harvesting Pipeline

The harvesting pipeline processes documents through a sequence of stages, each performing specific transformations and validations. The pipeline operates on batches of documents for efficiency while maintaining per-document error handling.

**Stage 1: Acquisition** - The harvester plugin connects to the source, retrieves raw documents, stores raw content in staging area, and records source metadata (URL, timestamp, source ID).

**Stage 2: Extraction** - Text extractors identify content type (HTML, PDF, Markdown), extract main text content, remove boilerplate and navigation, normalize whitespace and encoding, and store extracted text with original content.

**Stage 3: Metadata Enrichment** - Metadata extractors parse document metadata (title, author, date), compute document statistics (word count, sentence count), extract domain indicators (technical terms, writing style), and assign preliminary quality scores.

**Stage 4: Initial Filtering** - Basic filters check minimum length requirements (100+ words), validate language (English detection), verify encoding (UTF-8 or convertible), detect duplicates (using content hashing), and remove obviously invalid documents.

**Stage 5: Staging** - Valid documents move to staging area with assigned document IDs, indexed by source and domain, tagged with processing status (pending validation), and tracked in metadata store.

The pipeline implements comprehensive error handling. Individual document failures don't stop batch processing. Failed documents are logged with error details. Recoverable errors trigger retry with exponential backoff. Persistent failures are flagged for manual investigation. The pipeline tracks success rates per source and domain, alerting when success rates drop below expected thresholds.

---

## QUALITY CONTROL

### Multi-Stage Filtering

Quality control operates through multiple stages of increasing sophistication. Early stages apply simple, fast checks that eliminate obviously unsuitable documents. Later stages apply more complex analysis to detect subtle quality issues or AI contamination. This staged approach balances thoroughness with computational efficiency.

**Stage 1: Structural Validation**

Structural validation checks basic document properties. Length validation requires minimum 100 words (for reliable clause identification) and maximum 50,000 words (to avoid memory issues, outlier effects). Encoding validation verifies UTF-8 or convertible encoding and checks for mojibake or corruption. Format validation identifies and rejects binary content mixed with text, unusual character distributions, and excessive whitespace or formatting marks.

**Stage 2: Language Identification**

Language identification confirms English content using language detection libraries (langdetect, fasttext). You require high confidence scores (>0.95 probability of English), allow small amounts of foreign language (quotes, technical terms), but reject multilingual documents that mix languages extensively. This stage prevents non-English content from contaminating the baseline, which would affect phonetic and structural analysis in the Echo Engine.

**Stage 3: Content Quality Assessment**

Content quality assessment evaluates writing quality and coherence. Readability metrics compute Flesch-Kincaid grade level and sentence complexity to identify excessively simple or complex text. Vocabulary richness measures type-token ratio and hapax legomena to detect repetitive or limited vocabulary. Grammatical correctness uses rule-based checks or simple parsers to identify excessive errors. Coherence measures topic consistency and logical flow.

These metrics help identify low-quality content (poorly written, machine-translated, OCR errors) that would add noise to the baseline. You establish thresholds based on manual review of example documents, setting conservative thresholds initially and tightening as corpus quality stabilizes.

**Stage 4: AI Pattern Detection**

AI pattern detection applies heuristics and statistical tests to identify potentially AI-generated content. This is the most critical and challenging stage, as modern LLMs produce increasingly human-like text. Detection strategies evolve as generation techniques improve.

Heuristic checks look for known GPT signatures (phrases like "as an AI language model", "it's important to note that"), unusual repetition patterns (repeated sentence structures, phrase recycling), unnatural word choice (overly formal vocabulary in informal contexts, hedging language like "arguably", "potentially"), and structural artifacts (perfectly balanced paragraph lengths, unnaturally consistent sentence complexity).

Statistical checks analyze vocabulary diversity over document length, perplexity under a language model trained on pre-LLM text, burstiness (variance in sentence lengths and structures), and n-gram frequency distributions compared to human baselines. Documents with anomalous statistical profiles are flagged for manual review.

Temporal checks provide strong evidence for human authorship. Documents with publication dates before November 2022 (ChatGPT release) are considered low-risk. Documents from late 2022 to present are high-risk and require careful validation. Documents with verifiable authorship (known human writers, journalism bylines, academic author lists) receive higher trust scores.

### Validation Queue and Human Review

Documents flagged by automated filters or randomly sampled for quality assurance enter the validation queue for human review. The queue system prioritizes documents by review urgency, balancing coverage across domains, and tracking reviewer decisions for quality control.

**Review Interface**

The review interface presents documents with contextual information. Reviewers see the document text with highlighting of flagged patterns, source metadata (origin, date, author if available), automated quality scores, and comparison to similar documents. Reviewers make binary decisions (Accept or Reject with reasons) and can add annotations about quality issues or interesting patterns.

The interface tracks inter-rater reliability by having multiple reviewers evaluate the same documents. High agreement (kappa > 0.8) indicates clear guidelines and consistent decisions. Low agreement suggests ambiguous cases or unclear guidelines that need refinement.

**Review Guidelines**

Review guidelines provide consistent decision criteria. Acceptance criteria require clear human authorship indicators (natural variation in style, occasional errors or typos, personal voice or perspective), appropriate quality for domain (news articles should be edited, social media can be informal), no obvious AI generation markers, and alignment with target domain classification.

Rejection criteria include obvious AI generation patterns (GPT signatures, unnatural uniformity), poor quality (excessive errors, incoherent content, spam), wrong language or domain, and duplicate or near-duplicate content. Reviewers mark documents with uncertain classification for discussion and potential guideline updates.

**Review Workflow**

The review workflow supports efficient human validation at scale. Tier 1 requires manual review of all documents to establish high-confidence foundation corpus. Tier 2 transitions to sampling-based review, manually reviewing 5-20% of automatically filtered documents, focusing review effort on high-risk sources or borderline cases, and using reviewer feedback to improve automated filters. Tier 3 implements adaptive sampling, reviewing more documents from sources with low automation success, skipping review for sources with consistently high quality, and using active learning to identify informative documents.

The workflow integrates reviewer feedback into the quality control pipeline. Documents rejected by reviewers trigger investigation of why automated filters missed them. Common rejection patterns inform filter improvements. Reviewer annotations become training data for machine learning filters in Tier 3.

---

## METADATA MANAGEMENT

### Metadata Schema

Comprehensive metadata enables corpus analysis, reproducible harvesting, and quality tracking. The metadata schema captures information at three levels: document-level metadata for individual documents, source-level metadata for harvesting configurations, and corpus-level metadata for aggregate statistics.

**Document Metadata**

Each document has a unique identifier (UUID or hash-based ID), source attribution (source ID, original URL or file path, harvesting timestamp), temporal metadata (publication date if available, access date, temporal classification such as pre-LLM or post-LLM), content statistics (word count, sentence count, character count, language), domain classification (primary domain, secondary domains, confidence scores), quality metrics (automated quality score, AI pattern flags, reviewer decisions, validation timestamp), and processing history (filter decisions, error logs, version of harvesting pipeline).

Example document metadata:
```json
{
  "document_id": "doc_a3f9e2c4",
  "source": {
    "source_id": "nytimes_archive_2015_2021",
    "url": "https://www.nytimes.com/2020/05/15/technology/article-title",
    "harvested_at": "2025-10-19T14:23:11Z"
  },
  "temporal": {
    "published_at": "2020-05-15",
    "temporal_class": "pre_llm",
    "confidence": "high"
  },
  "content": {
    "word_count": 1247,
    "sentence_count": 52,
    "language": "en",
    "language_confidence": 0.99
  },
  "domain": {
    "primary": "news",
    "secondary": ["technology"],
    "confidence": 0.95
  },
  "quality": {
    "automated_score": 0.87,
    "ai_flags": [],
    "human_review": {
      "decision": "accept",
      "reviewer": "reviewer_42",
      "timestamp": "2025-10-19T15:30:00Z"
    }
  },
  "processing": {
    "pipeline_version": "1.0.0",
    "filters_passed": ["length", "language", "encoding", "ai_patterns"],
    "baseline_score": 0.18
  }
}
```

**Corpus Metadata**

At the corpus level, metadata tracks aggregate statistics. Corpus version (semantic versioning for baseline releases) identifies the corpus snapshot. Statistics include total document count, documents per domain, word count distribution, temporal distribution, and quality distribution. Stratification summary shows actual vs. target proportions by domain, source diversity metrics, and temporal coverage. Processing history includes harvest start and end dates, number of documents harvested vs. validated, filter pass/fail rates, and reviewer statistics.

### Metadata Storage

Metadata storage supports efficient querying and updates. For Tier 1, simple JSON or YAML files suffice, organized by document or source, with one metadata file per document or aggregated by source. For Tier 2, SQLite or similar embedded database enables structured queries (find all news documents from 2019), indexing on common fields (domain, source, quality score), and atomic updates during validation. For Tier 3, production database (PostgreSQL, MongoDB) supports concurrent access from multiple harvesters, complex analytical queries, and versioning of corpus snapshots.

The metadata store integrates with corpus directories through file naming conventions. Each document file has a corresponding metadata sidecar (document_123.txt has document_123.json). Directory structure reflects domain stratification (news/, academic/, fiction/, etc.). Metadata enables reconstruction of corpus state at any point, supporting reproducible baseline calculations.

---

## CORPUS ORGANIZATION

### Directory Structure

The corpus is organized in a hierarchical directory structure that reflects stratification and facilitates processing. The structure separates raw harvested documents, validated documents, and processed baselines, enabling clear separation of concerns and supporting incremental validation.

```
corpus/
├── raw/                          # Raw harvested documents (pre-validation)
│   ├── news/
│   │   ├── nytimes/
│   │   │   ├── 2020/
│   │   │   │   ├── doc_001.txt
│   │   │   │   ├── doc_001.json
│   │   │   │   └── ...
│   │   │   └── 2021/
│   │   ├── bbc/
│   │   └── guardian/
│   ├── academic/
│   │   ├── arxiv_cs/
│   │   ├── arxiv_physics/
│   │   └── pubmed/
│   ├── fiction/
│   │   ├── gutenberg/
│   │   └── archive_org/
│   ├── technical/
│   │   ├── github_docs/
│   │   └── official_docs/
│   └── metadata/
│       ├── harvest_log.json
│       └── source_stats.json
│
├── validated/                    # Human-validated documents (baseline-ready)
│   ├── news/
│   │   ├── doc_012.txt
│   │   ├── doc_012.json
│   │   └── ...
│   ├── academic/
│   ├── fiction/
│   ├── technical/
│   ├── metadata/
│   │   ├── corpus_stats.json
│   │   └── validation_log.json
│   └── README.md                 # Corpus documentation
│
└── baselines/                    # Processed baseline statistics
    ├── tier1_v1.0/
    │   ├── baseline.pkl          # Baseline statistics (mean, std, etc.)
    │   ├── metadata.json         # Corpus composition and processing info
    │   └── scores.csv            # Individual document scores
    ├── tier2_v1.0/
    └── current -> tier1_v1.0     # Symlink to active baseline
```

### Document Naming Convention

Documents follow a consistent naming convention that encodes metadata in filenames. The format is: `{source}_{domain}_{date}_{id}.{ext}`, for example: `nytimes_news_20200515_a3f9e2c4.txt`. This convention enables quick identification of document origin, supports alphabetical sorting by date, and facilitates batch processing by source or domain.

Metadata sidecars use matching names with `.json` extension: `nytimes_news_20200515_a3f9e2c4.json`. This pairing ensures metadata travels with documents during file operations.

### Corpus Documentation

Each corpus version includes comprehensive documentation in a README file at the corpus root. The documentation describes corpus composition (documents per domain, source breakdown, temporal range), validation procedures (automated filters applied, human review statistics, known limitations), baseline statistics (mean and standard deviation by domain, overall distribution, quality metrics), and usage guidelines (recommended applications, known biases, update schedule).

This documentation enables reproducible research and informed usage. External users can understand corpus characteristics and limitations. Internal teams can track corpus evolution and make informed decisions about baseline updates.

---

## STATISTICAL VALIDATION

### Representativeness Assessment

Statistical validation ensures the harvested corpus adequately represents human writing for baseline purposes. Validation operates at three levels: distribution validity (does the corpus follow expected statistical properties), stratification balance (are domains appropriately represented), and contamination detection (is the corpus free of AI-generated content).

**Distribution Analysis**

Distribution analysis examines whether corpus scores follow expected patterns. You process the corpus through the full SpecHO pipeline using the BaselineCorpusProcessor, generating a score for each document. Then you analyze the score distribution: compute summary statistics (mean, median, standard deviation, quartiles), test for normality (Shapiro-Wilk test, Q-Q plots), identify outliers (z-score > 3), and compare to theoretical expectations (human writing should score low, typically 0.1-0.3 for echo-based watermarks).

Non-normal distributions or unexpected means suggest corpus issues. High mean indicates possible AI contamination. High variance indicates excessive heterogeneity (mixing very different text types). Outliers may represent genuinely unusual human writing or undetected AI content, and require investigation.

**Stratification Balance**

Stratification balance checks whether domain representation matches target proportions. You compute actual proportions per domain, compare to target proportions, calculate χ² goodness-of-fit test, and identify underrepresented domains. Significant imbalance (χ² p-value < 0.05) suggests biased sampling that may affect baseline generalizability.

You also analyze score distributions within each domain. Compute per-domain statistics (mean, std), test for significant differences between domains (ANOVA or Kruskal-Wallis), and decide whether to maintain separate domain-specific baselines or use pooled baseline. Large inter-domain differences suggest domain-specific baselines improve accuracy.

**Contamination Detection**

Contamination detection uses multiple strategies to identify AI-generated content that passed initial filters. Temporal analysis flags documents with publication dates near LLM release dates (late 2022 onward) for enhanced scrutiny and requires stronger provenance evidence for recent documents. Statistical analysis identifies documents with extreme scores (z > 3 within corpus) and unusual patterns (perfect uniformity, suspicious repetition). Provenance auditing verifies authorship claims for high-scoring documents and cross-references with known AI text databases.

Documents flagged as potentially contaminated are removed from the baseline corpus and added to a quarantine directory for investigation. If investigation confirms AI generation, you update filters to catch similar documents. If investigation confirms human authorship, you refine your understanding of human writing variance.

### Quality Metrics

Quality metrics quantify corpus characteristics and track changes over time. Coverage metrics measure domain representation (percentage per domain, underrepresented domains), source diversity (number of sources, documents per source), and temporal coverage (time span, documents per year). Quality metrics track human review rate (percentage manually validated), acceptance rate (percentage passing validation), and contamination estimate (percentage flagged as suspicious). Statistical metrics report mean and standard deviation (overall and per-domain), normality test p-values, and outlier count.

These metrics form the basis for corpus quality reports that summarize current status, compare to previous versions, highlight concerns or gaps, and recommend improvements. Regular quality reports (weekly during active harvesting, monthly during maintenance) ensure corpus integrity over time.

---

## INTEGRATION WITH BASELINE PROCESSING

### Pipeline Integration

The harvesting system outputs validated documents in the format expected by your existing BaselineCorpusProcessor. Integration requires minimal changes to existing code while enabling automated corpus updates and version management.

**Corpus Loading**

The BaselineCorpusProcessor is extended to support versioned corpus directories. Instead of hardcoding a corpus path, it accepts a corpus version parameter that resolves to the appropriate validated directory. A configuration file maps version identifiers to directory paths and tracks the current active version.

```python
class BaselineCorpusProcessor:
    def __init__(self, corpus_version: str = "current", ...):
        # Resolve version to directory path
        corpus_dir = self._resolve_corpus_version(corpus_version)
        
        # Load corpus with existing logic
        self.corpus_files = self._discover_text_files(corpus_dir)
        
    def _resolve_corpus_version(self, version: str) -> Path:
        # Load corpus registry
        with open("corpus/registry.json") as f:
            registry = json.load(f)
            
        if version == "current":
            version = registry["current_version"]
            
        return Path(registry["versions"][version]["path"])
```

**Metadata Integration**

The BaselineCorpusProcessor is enhanced to load and utilize document metadata. Metadata enables domain-specific baseline calculation, weighted sampling by quality scores, and provenance tracking in baseline statistics.

```python
def process_corpus(self, use_metadata: bool = True) -> dict:
    scores = []
    metadata_list = []
    
    for file_path in self.corpus_files:
        # Process document
        score = self._process_single_document(text)
        scores.append(score)
        
        # Load metadata if available
        if use_metadata:
            meta_path = file_path.with_suffix('.json')
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                    metadata['score'] = score
                    metadata_list.append(metadata)
    
    # Calculate statistics (potentially stratified by domain)
    baseline_stats = self._calculate_statistics(scores, metadata_list)
    
    return baseline_stats
```

**Version Management**

Baseline statistics are versioned in sync with corpus versions. Each baseline calculation includes corpus version identifier, processing pipeline version, timestamp, and configuration parameters. This versioning enables reproducible detection (use the same baseline version for comparison) and A/B testing (compare detection performance across baseline versions).

---

## TOOLING AND IMPLEMENTATION

### Technology Stack

The harvesting system uses proven open-source technologies that balance functionality with implementation complexity.

**Core Framework** (Python 3.9+)

Python provides the implementation language for consistency with the existing SpecHO pipeline. Key libraries include requests for HTTP requests and API calls, Beautiful Soup or Scrapy for web scraping, schedule or APScheduler for task scheduling, and tqdm for progress tracking.

**Data Processing**

Text processing uses standard libraries: pdfplumber or PyPDF2 for PDF text extraction, python-docx for Word document parsing, charset-normalizer for encoding detection, and langdetect or fasttext for language identification.

**Storage and Databases**

For Tier 1, file-based storage with JSON metadata suffices. For Tier 2, SQLite provides lightweight structured storage. For Tier 3, PostgreSQL offers production-grade reliability, or MongoDB provides flexible schema for evolving metadata.

**Quality Control**

Automated quality assessment employs textstat for readability metrics, spaCy for grammatical analysis (already in SpecHO dependencies), and sentence-transformers for semantic similarity detection (contamination detection).

**Infrastructure**

Harvesting infrastructure can start simple (cron jobs on single machine) and scale as needed (distributed task queue with Celery, containerization with Docker, cloud storage for corpus files like S3 or GCS, and monitoring with Prometheus and Grafana).

### Implementation Phases

Implementation proceeds through three phases aligned with tier progression.

**Phase 1: Foundation (Weeks 1-2)**

In this phase, you implement manual corpus construction. You manually collect 50-100 documents from high-quality sources, organize in simple directory structure (by domain), create basic metadata files (JSON per document), implement simple validation script (length, language, encoding), run BaselineCorpusProcessor to generate initial baseline, and validate detection performance on known test cases.

Deliverables include a validated corpus (50-100 documents), baseline statistics (mean, std), and validation report (performance metrics).

**Phase 2: Automation (Weeks 3-8)**

This phase implements semi-automated harvesting. You develop API harvesters for major sources (news, academic), implement automated quality filters, build validation queue and review interface, develop metadata management system, extend BaselineCorpusProcessor for metadata integration, and harvest and validate 500-1000 documents.

Deliverables include a harvesting framework (plugin architecture, 3-5 source plugins), quality control pipeline (automated filters, review workflow), expanded corpus (500-1000 documents), and versioned baselines (Tier 2).

**Phase 3: Production (Months 3+)**

The production phase implements continuous harvesting. You develop additional harvesters (covering 10+ sources), implement advanced contamination detection, build automated monitoring and alerting, develop adaptive sampling strategies, integrate with production detection pipeline, and establish baseline maintenance schedule (monthly or quarterly updates).

Deliverables include a production harvesting system (fully automated, monitoring, version management), comprehensive corpus (1000+ documents, multiple versions), and operational procedures (maintenance schedule, quality assurance, incident response).

---

## OPERATIONAL PROCEDURES

### Corpus Maintenance

Corpus maintenance ensures baseline quality over time. Maintenance activities include regular harvesting runs (daily for high-volume sources, weekly for others), validation queue processing (review pending documents, maintain target review rates), quality monitoring (track metrics, investigate anomalies), baseline updates (monthly or quarterly reprocessing), and archival (version previous baselines, document changes).

A maintenance schedule structures these activities. Daily tasks run automated harvesters, monitor job success rates, and review high-priority documents. Weekly tasks process validation queue, analyze new document quality, and update source configurations. Monthly tasks calculate baseline statistics, compare to previous versions, prepare quality report, and release new baseline version if warranted. Quarterly tasks conduct comprehensive corpus audit, review source performance, update harvesting strategies, and plan improvements.

### Quality Assurance

Quality assurance procedures catch corpus degradation early. Continuous monitoring tracks key metrics (harvest success rates, filter pass rates, validation acceptance rates, baseline score distributions, per-source quality). Automated alerts trigger on anomalies (sudden changes in baseline statistics, unusually low acceptance rates, source failures, contamination flags).

Periodic audits provide deeper quality assessment. Human audits involve randomly sampling validated documents (10-20 per month), verifying acceptance decisions were correct, checking for missed AI content, and updating guidelines based on findings. Statistical audits compare current baseline to previous versions, test for distribution shifts, identify new outliers, and validate stratification balance.

### Incident Response

Despite careful quality control, incidents may occur (AI contamination discovered, source reliability degraded, baseline drift detected). An incident response procedure provides structured handling. For contamination incidents, you quarantine affected documents, trace contamination source (which source, which time period), update filters to prevent recurrence, remove contaminated documents from baseline, reprocess baseline statistics, and document incident and remediation. For drift incidents, you investigate cause (language evolution, source changes, detection changes), determine if drift is expected or problematic, adjust baseline or detection threshold if needed, and update documentation.

---

## TIER SUMMARY AND ROADMAP

### Tier 1: Foundation (Current State)

**Goal**: Establish working baseline quickly with high-confidence corpus

**Characteristics**: Manual document collection and validation, simple directory organization, basic metadata (JSON files), single-domain or balanced multi-domain, 50-200 documents

**Timeline**: 1-4 weeks

**Deliverables**: Validated corpus directory, baseline statistics file (pickle), corpus documentation (README)

### Tier 2: Production (Next 2-3 months)

**Goal**: Implement semi-automated harvesting with quality control

**Characteristics**: Automated API harvesters for major sources, quality filter pipeline, validation queue with review interface, structured metadata storage (SQLite), domain-stratified sampling, 500-2000 documents

**Timeline**: 2-3 months

**Deliverables**: Harvesting framework (plugin architecture), quality control pipeline, expanded corpus, versioned baselines, operational procedures

### Tier 3: Scale (Future)

**Goal**: Production-grade continuous corpus maintenance

**Characteristics**: Comprehensive source coverage (10+ sources), advanced contamination detection, automated monitoring and alerting, adaptive sampling and active learning, production database (PostgreSQL), domain-specific baselines, online baseline updates

**Timeline**: Ongoing (months 4+)

**Deliverables**: Fully automated system, extensive corpus (1000+ documents), sophisticated quality control, continuous improvement process

---

## CONCLUSION

The corpus harvesting architecture provides a systematic approach to building representative baseline distributions for watermark detection. By combining principled sampling strategies, rigorous quality control, and progressive automation, the system balances statistical requirements with practical constraints.

Key architectural decisions reflect fundamental tradeoffs. Stratified sampling ensures representativeness but requires more complex harvesting. Human validation ensures quality but limits scale. Automated filtering enables scale but risks false negatives. The multi-tier approach resolves these tradeoffs by starting conservative (manual, small-scale) and progressively automating as confidence grows.

The architecture integrates cleanly with your existing SpecHO pipeline. The BaselineCorpusProcessor consumes harvested documents without modification. Metadata enables enhanced analysis without breaking existing workflows. Versioned baselines support reproducible detection and continuous improvement.

Implementation follows a clear path. Begin with manual corpus construction (Tier 1) to establish your initial baseline and validate detection performance. Expand to semi-automated harvesting (Tier 2) as you understand quality patterns and source reliability. Scale to production infrastructure (Tier 3) as usage grows and requirements evolve.

The quality of your baseline distribution fundamentally determines detection accuracy. Invest in corpus quality early, establish rigorous validation procedures, and maintain discipline in quality control. A well-constructed corpus enables confident watermark detection with known error bounds. A poorly constructed corpus leads to unreliable detection regardless of how sophisticated your detection algorithms become.

This architecture provides the foundation for building that high-quality corpus at scale.

---

**Document Version**: 1.0  
**Last Updated**: October 19, 2025  
**Next Review**: Upon Tier 2 implementation start  
**Related Documents**: 
- REFERENCE_DISTRIBUTION_GUIDE.md (statistical theory)
- architecture.md (detection pipeline)
- SPECS.md (Component 5: Statistical Validator)
