"""Main pipeline for schema field mapping."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
import os

from .schema_parser import SchemaParser
from .embeddings import EmbeddingManager
from .llm import MappingRefinement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def infer_type_transform(source_type: str, target_type: str) -> str:
    """Infer type transformation from source to target type."""
    rules = {
        ("INT", "ObjectId"): "INT -> ObjectId",
        ("VARCHAR", "String"): "VARCHAR -> String",
        ("DATETIME", "ISODate"): "DATETIME -> ISODate",
        ("TINYINT", "Boolean"): "TINYINT -> Boolean",
        ("DECIMAL", "Number"): "DECIMAL -> Number",
        ("CHAR", "String"): "CHAR -> String",
    }

    for (src, tgt), rule in rules.items():
        if src in source_type and tgt in target_type:
            return rule

    return f"{source_type} -> {target_type}"


class SchemaMapper:
    """Map database schema fields using semantic embeddings and LLM refinement."""

    def __init__(self, api_key: str = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.embedding_mgr = EmbeddingManager(api_key)
        self.refiner = MappingRefinement(api_key)
        self.source_fields = []
        self.target_fields = []

    def load_schemas(self, source_path: str, target_path: str) -> None:
        """Load source and target schemas."""
        logger.info(f"Loading source schema from {source_path}")
        self.source_fields = SchemaParser.parse_source_schema(source_path)
        logger.info(f"Loaded {len(self.source_fields)} source fields")

        logger.info(f"Loading target schema from {target_path}")
        self.target_fields = SchemaParser.parse_target_schema(target_path)
        logger.info(f"Loaded {len(self.target_fields)} target fields")

    def map_fields(self):
        """Find best matching target field and refine with LLM reasoning."""
        mappings = []

        self.embedding_mgr.build_index(self.target_fields)

        for i, src_field in enumerate(self.source_fields, 1):
            logger.info(f"Processing {i}/{len(self.source_fields)}: {src_field.name}")

            try:
                candidates = self.embedding_mgr.search_similar_fields(src_field, k=5)

                # Prioritize candidates from matching collection
                table_mapping = {
                    "emp_master": "employees",
                    "dept_info": "departments",
                    "locations": "locations",
                }
                preferred_collection = table_mapping.get(src_field.table_or_collection)

                if candidates:
                    # Sort candidates: matching collection first, then by score
                    sorted_candidates = sorted(
                        candidates,
                        key=lambda x: (
                            x[0].table_or_collection != preferred_collection,
                            -x[1],
                        ),
                    )
                    target_field, score = sorted_candidates[0]
                    refinement = self.refiner.refine_mapping(
                        src_field, target_field, score, sorted_candidates[:5]
                    )
                else:
                    target_field = None
                    score = 0
                    refinement = {
                        "destination_field": None,
                        "type_transform": "No matching field found",
                        "reasoning": "No semantically similar field found",
                        "notes": "Manual review required",
                        "enum": None,
                    }

                dest_field = refinement.get("destination_field")
                type_transform = refinement.get("type_transform")

                if dest_field and target_field and not type_transform:
                    type_transform = infer_type_transform(src_field.type, target_field.type)

                # Format type_transform to use -> separator
                if type_transform and " to " in type_transform:
                    type_transform = type_transform.replace(" to ", " -> ")

                # Remove collection prefix from destination_field EXCEPT for _id fields
                # Keep _id with collection prefix since we have multiple _id fields
                display_dest = dest_field
                if dest_field and "." in dest_field and not dest_field.endswith("._id"):
                    parts = dest_field.split(".", 1)
                    if len(parts) == 2:
                        display_dest = parts[1]

                # Calculate confidence based on similarity score
                if display_dest:
                    confidence = round(score * 100, 0) / 100  # Round to 2 decimals
                else:
                    confidence = 0.0

                mapping = {
                    "source_field": src_field.name,
                    "destination_field": display_dest,
                    "type_transform": type_transform,
                    "reasoning": refinement.get("reasoning", ""),
                    "notes": refinement.get("notes"),
                    "confidence": confidence,
                }
                mappings.append(mapping)

            except Exception as e:
                logger.error(f"Error processing {src_field.name}: {e}")
                mapping = {
                    "source_field": src_field.name,
                    "destination_field": None,
                    "type_transform": "Error during processing",
                    "reasoning": f"Error: {str(e)}",
                    "notes": "Manual review required",
                }
                mappings.append(mapping)

        return mappings, []

    def save_output(self, mappings, unmapped, output_path: str) -> None:
        """Save mappings organized by table/collection pairs."""
        # Group mappings by source table
        table_groups = {}
        table_mapping = {
            "emp_master": ("emp_master", "employees"),
            "dept_info": ("dept_info", "departments"),
            "locations": ("locations", "locations"),
        }

        for mapping in mappings:
            src_field_name = mapping["source_field"]
            src_field = next(
                (f for f in self.source_fields if f.name == src_field_name),
                None,
            )
            if not src_field:
                continue

            table_name = src_field.table_or_collection
            if table_name not in table_groups:
                table_groups[table_name] = {
                    "source_table": table_name,
                    "destination_collection": table_mapping.get(table_name, ("", ""))[1],
                    "field_mappings": [],
                    "unmapped_source_fields": [],
                    "unmapped_destination_fields": [],
                }

            # Keep the actual confidence from mapping
            table_groups[table_name]["field_mappings"].append(mapping)

        # Build output with table-organized structure
        tables_output = []
        for table_name, data in sorted(table_groups.items()):
            # Calculate table-level confidence
            confidences = [
                m.get("confidence", 0.85) for m in data["field_mappings"]
            ]
            table_confidence = sum(confidences) / len(confidences) if confidences else 0

            table_output = {
                "source_table": data["source_table"],
                "destination_collection": data["destination_collection"],
                "confidence": round(table_confidence, 2),
                "reasoning": f"Mapping {data['source_table']} to {data['destination_collection']}",
                "field_mappings": data["field_mappings"],
                "unmapped_source_fields": [],
                "unmapped_destination_fields": [],
            }
            tables_output.append(table_output)

        output = {
            "mapping_version": "1.0",
            "source": "legacy_hrm (MySQL)",
            "destination": "people_platform (MongoDB)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tables": tables_output,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Output saved to {output_path}")

    def run(self, source_path: str, target_path: str, output_path: str) -> None:
        """Run the complete mapping pipeline."""
        logger.info("Starting schema mapping...")
        self.load_schemas(source_path, target_path)
        mappings, unmapped = self.map_fields()
        self.save_output(mappings, unmapped, output_path)
        logger.info(f"Mapped {len(mappings)} fields, {len(unmapped)} unmapped")


def main():
    project_root = Path(__file__).parent.parent
    source_schema = project_root / "schemas" / "source_schema.json"
    target_schema = project_root / "schemas" / "target_schema.json"
    output_file = project_root.parent / "output" / "field_mappings.json"

    mapper = SchemaMapper()
    mapper.run(str(source_schema), str(target_schema), str(output_file))


if __name__ == "__main__":
    main()
