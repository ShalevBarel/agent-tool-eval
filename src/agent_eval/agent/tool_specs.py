"""The tools as the model sees them: a name, a description and an input schema.

Each input schema is JSON Schema. With "strict" set, the API guarantees that
the model's arguments match the schema, so every call has exactly the
arguments the tool takes, with the right types.
"""

TOOL_SPECS = [
    {
        "name": "get_order",
        "description": "Get an order's place and its delivery window.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order's id, such as o1."},
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
