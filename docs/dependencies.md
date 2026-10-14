# Dependency and artifact attribution

Tomte's MIT license covers this project's source, not its dependencies. The built
wheel contains Tomte source and its own MIT notice, not bundled dependency code.
Dependency installations retain the libraries' separate license/notice files.
Do not remove those notices if redistributing an environment or container later.

The locked Python 3.12 installation was inventoried during release hardening:
Pydantic 2.13.2, pydantic-settings 2.13.1, Pydantic-AI 1.84.0, pgvector 0.4.2,
Redis Python client 7.4.0, Anthropic 0.96.0 and VoyageAI 0.3.7 use MIT;
OpenAI 2.32.0 uses Apache-2.0; scikit-learn 1.8.0 uses BSD-3-Clause;
schedule 1.2.2 uses MIT. Psycopg 3.3.3 declares LGPL-3.0-only.
These are installed metadata/license observations, not a relicensing statement.
VoyageAI's installed LICENSE retains both OpenAI and VoyageAI attribution.

No vendored third-party source or removed upstream notice was identified in the
candidate. Synthetic evaluation fixtures and SDK examples are retained from the
approved owner-provided source. An automated scan cannot prove original
authorship of every example: owner review remains required before publication.

Transitive dependency metadata was inventoried privately (174 distributions
including the project and test tooling). This is not a comprehensive legal or
vulnerability audit. Optional security extras were not installed or verified.
Docker image contents are downloaded separately, not shipped in the Python
artifacts, and retain their upstream licenses. A future redistributable image or
vendored dependency requires its own notice/compliance review.
