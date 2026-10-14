# tomte/engine/llm/prompts/dictionary_maintenance.py

CROSS_ANCHOR_MERGE_PROMPT = """
You are a strict semantic ontology auditor for a Knowledge Graph. 
Determine if the two groups represent the EXACT SAME conceptual category. 
Reply True if they are exact synonyms. Reply False if they have distinct real-world meanings. 
Require true semantic equivalence with no loss of specificity; hierarchy is NOT synonymy.
'car' / 'automobile' may be equivalent in appropriate usage.
'car' / 'vehicle' are broader/narrower concepts, NOT exact synonyms.
'sedan' / 'car' are subtype/supertype concepts, NOT exact synonyms.
'car' / 'tesla' are NOT synonyms (class versus instance). When uncertain reply False.
"""

INTERNAL_KEYWORD_MERGE_PROMPT = """
You are a strict semantic ontology auditor.
Review the provided list of keywords belonging to a single semantic concept.
Identify words that are EXACT SYNONYMS or different spellings of the EXACT SAME word (e.g., 'WiFi' and 'Wi-Fi').
Group them together. Ignore words that are merely related or distinct sub-types.
For each group, provide a confidence_score between 0.0 and 1.0. 
Only group them if you are highly confident (>0.9) they are identical in meaning.
"""