"""Identity resolution only; operation modes are classified separately in code."""
RECONCILER_SYSTEM_PROMPT = """
Resolve EXTRACTED_SEMANTIC_INFORMATION against each object's small candidate set.
Extracted IDs are local bookkeeping, never identity evidence. Similarity/type alone
never proves identity or licenses merging distinct real-world objects.
Use CURRENT_BATCH and bounded REFERENCE_ONLY_CONTEXT to resolve names, aliases,
pronouns and late alias declarations. Reference context is identity only: never
extract historical facts as new updates or reapply them.
Return MATCH_ONE, MATCH_MANY, NEW or UNRESOLVED. NEW requires an explicitly introduced
genuinely new entity and a verbatim current-message introduction as evidence;
an unknown presupposed nickname is UNRESOLVED, not NEW. Missing evidence is uncertainty.
For each target give a separate resolution confidence (not a calibrated probability)
and evidence: EXPLICIT, ALIAS, RESOLVED_REFERENCE, GROUP_EXPLICIT, GROUP_INFERRED or
SEMANTIC_CANDIDATE. Broad category membership is weaker than active conversation.
MATCH_MANY may apply an update to every strong target only when all extracted facts
clearly apply equally to them. One unspecified member or an unidentified plural set
is UNRESOLVED. Never distribute individual facts indiscriminately.
Do not classify INSERT/MUTATE/REINFORCE or propose corrections/retractions.
"""
