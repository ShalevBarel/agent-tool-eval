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
    {
        "name": "roads_from",
        "description": "Get a list of the roads out of a place, sorted by destination",
        "input_schema": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": "The place from which the roads are out of, such as p1.",
                },
            },
            "required": ["place"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "route_minutes",
        "description": "Get the minutes it takes to drive a route, stop by stop.",
        "input_schema": {
            "type": "object",
            "properties": {
                "route": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The list of places the route goes through, by order.",
                },
            },
            "required": ["route"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "shortest_route",
        "description": "Get the fastest open route from start to end, and its minutes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "string",
                    "description": "The place from which the courier starts.",
                },
                "end": {"type": "string", "description": "The place that the courier is going to."},
            },
            "required": ["start", "end"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "check_schedule",
        "description": "Drive a delivery order from the courier's start, by the fastest open route,"
        " and report its detailes, such as: on_time, finish_minute, problem, stops."
        " Stops reports at each stop the: order_id, place, arrival_minute,"
        " delivery_minute",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The list of the order id's of the orders"
                    " that needs to be delivered.",
                },
            },
            "required": ["order_ids"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_answer",
        "description": "Submit a final answer. Mention wether is is solvable or not,"
        " and the order"
        " id's by the order the courier delivers them. If it is not"
        " solvable, order_ids must be empty.",
        "input_schema": {
            "type": "object",
            "properties": {
                "solvable": {
                    "type": "boolean",
                    "description": "Wether the problem is solvable or not.",
                },
                "order_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The answer of the problem. The list of the order id's by the"
                    " order that the courier delivers them to achive minimal total delivery time.",
                },
            },
            "required": ["solvable", "order_ids"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
