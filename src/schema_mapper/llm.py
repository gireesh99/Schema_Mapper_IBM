"""Generate mapping reasoning and refinement using LLM."""

import json
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class MappingRefinement:
    """Use LLM to refine mappings and generate reasoning."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def refine_mapping(self, source_field, target_field, similarity_score: float, all_candidates: list) -> dict:
        """Use LLM to refine mapping and generate reasoning."""
        candidates_text = self._format_candidates(all_candidates)

        # Detect if this is a primary key
        is_primary_key = source_field.name in ['emp_id', 'dept_id', 'loc_id'] and source_field.table_or_collection in [
            'emp_master', 'dept_info', 'locations'
        ]

        pk_hint = ""
        if is_primary_key:
            pk_hint = f"\n- This is a PRIMARY KEY field ({source_field.name}) and MUST map to _id in the target collection"

        prompt = f"""You are a database schema expert. Analyze this field mapping.

Source Field:
- Name: {source_field.name}
- Type: {source_field.type}
- Table: {source_field.table_or_collection}
- Description: {source_field.description}

Top Candidate Target Fields:
{candidates_text}

Task:
1. Choose the best candidate (or null if no valid match exists)
2. Provide reasoning for your choice
3. Generate transformation notes if needed

Special rules:
- PRIMARY KEYS: emp_id→employees._id, dept_id (from dept_info)→departments._id, loc_id→locations._id{pk_hint}
- Foreign keys: Identify and map to collection._id references
- Fields with very different semantics → null (e.g., sal_currency, work_email, work_phone)
- Nested fields should use full path (e.g., fullName.firstName)

Respond with valid JSON only:
{{
    "destination_field": "best match or null",
    "type_transform": "transformation description",
    "reasoning": "one sentence why this mapping is correct (or why null)",
    "notes": "details or null",
    "enum": "enum values or null"
}}"""

        logger.info(f"Refining mapping for {source_field.name}...")

        response = self.client.chat.completions.create(
            model="gpt-5.4-mini-2026-03-17",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        result_text = response.choices[0].message.content.strip()

        # Strip markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.replace("```json", "").replace("```", "").strip()

        try:
            result = json.loads(result_text)
            return {
                "destination_field": result.get("destination_field"),
                "type_transform": result.get("type_transform", ""),
                "reasoning": result.get("reasoning", ""),
                "notes": result.get("notes"),
                "enum": result.get("enum"),
            }
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response: {result_text}")
            # Fallback: use the best candidate if score is high enough
            if similarity_score > 0.7:
                return {
                    "destination_field": target_field.full_path,
                    "type_transform": "",
                    "reasoning": f"Mapped based on semantic similarity",
                    "notes": None,
                    "enum": None,
                }
            return {
                "destination_field": None,
                "type_transform": "",
                "reasoning": "No valid mapping found",
                "notes": None,
                "enum": None,
            }

    @staticmethod
    def _format_candidates(candidates):
        """Format candidate fields for the prompt."""
        lines = []
        for i, (field, score) in enumerate(candidates[:5], 1):
            lines.append(f"{i}. {field.full_path} ({field.type}) - Score: {score:.2%}")
            lines.append(f"   Description: {field.description}")
            lines.append(f"   Collection: {field.table_or_collection}")
        return "\n".join(lines)
