"""Parse database schemas from JSON files."""

import json
from dataclasses import dataclass


@dataclass
class Field:
    """Database field with metadata."""
    name: str
    type: str
    table_or_collection: str
    description: str = ""
    nested_path: str = ""

    @property
    def full_path(self) -> str:
        """Return full path for nested or simple fields."""
        path = self.nested_path or self.name
        return f"{self.table_or_collection}.{path}"

    def get_search_text(self) -> str:
        """Text to embed: name, description, type."""
        return f"{self.name} {self.description} {self.type}".strip()


class SchemaParser:
    """Parse MySQL and MongoDB schema JSON files."""

    @staticmethod
    def parse_source_schema(file_path: str) -> list[Field]:
        """Parse MySQL source schema."""
        with open(file_path) as f:
            schema = json.load(f)

        fields = []
        for table_name, table_data in schema.get("tables", {}).items():
            for field in table_data.get("fields", []):
                fields.append(
                    Field(
                        name=field["name"],
                        type=field["type"],
                        table_or_collection=table_name,
                        description=field.get("description", ""),
                    )
                )
        return fields

    @staticmethod
    def parse_target_schema(file_path: str) -> list[Field]:
        """Parse MongoDB target schema, handling nested objects."""
        with open(file_path) as f:
            schema = json.load(f)

        fields = []
        for collection_name, collection_data in schema.get("collections", {}).items():
            for field in collection_data.get("fields", []):
                if field.get("nested"):
                    for nested_name, nested_type in field["nested"].items():
                        fields.append(
                            Field(
                                name=nested_name,
                                type=nested_type,
                                table_or_collection=collection_name,
                                description=field.get("description", ""),
                                nested_path=f"{field['name']}.{nested_name}",
                            )
                        )
                else:
                    fields.append(
                        Field(
                            name=field["name"],
                            type=field["type"],
                            table_or_collection=collection_name,
                            description=field.get("description", ""),
                        )
                    )
        return fields
