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
        "description": (
            "Get the roads out of a place, sorted by destination. Each road has its "
            "destination, its minutes, and whether it is open today. Closed roads "
            "are listed too, with open set to false."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "place": {
                    "type": "string",
                    "description": "The place the roads leave from, such as p1.",
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
        "description": (
            "Drive the courier from its start through the given orders, in the given "
            "order, by the fastest open route. Report on_time, finish_minute, problem, "
            "and the stops, with order_id, place, arrival and delivery at each one. "
            "Checking stops at the first order that can't be delivered on time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The order ids, in the order the courier delivers them. They "
                        "may be just the first part of a full delivery order."
                    ),
                },
            },
            "required": ["order_ids"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_answer",
        "description": (
            "Submit your final answer: whether the task is solvable, and the order "
            "ids in the order the courier delivers them. If it is not solvable, "
            "order_ids must be empty."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "solvable": {
                    "type": "boolean",
                    "description": "Whether the problem is solvable or not.",
                },
                "order_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The order ids in the order the courier delivers them, so that "
                        "the last delivery is as early as possible. Empty if not solvable."
                    ),
                },
            },
            "required": ["solvable", "order_ids"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

# Longer descriptions for the same tools: what each returns, when it helps,
# and how it combines with the others. Only the descriptions change; the
# input schemas stay the same.
IMPROVED_DESCRIPTIONS = {
    "get_order": (
        "Get an order's place and its delivery window. The window is two whole "
        "minutes, window_start and window_end, and a delivery at either end is on "
        "time. Ask for several orders in one turn by making several calls at once."
    ),
    "roads_from": (
        "Get the roads that leave a place: each one's destination, its minutes, "
        "and whether it is open today. Closed roads are listed too, with open set "
        "to false. You rarely need this, since shortest_route already finds the "
        "fastest open route between two places."
    ),
    "route_minutes": (
        "Get the total minutes of a route given stop by stop, such as "
        '["p0", "p3", "p5"]. It fails with an error if two stops in a row have no '
        "road between them, or if that road is closed today."
    ),
    "shortest_route": (
        "Get the fastest open route from start to end, its places and its minutes. "
        "If end can't be reached on open roads, reachable is false and minutes is "
        "null. Roads can be one-way, so the minutes from a to b can differ from "
        "the minutes from b to a. Many calls can go in one turn, for example from "
        "the start and from every order's place to every other order's place."
    ),
    "check_schedule": (
        "Simulate a delivery order exactly as it would run: the courier takes the "
        "fastest open route to each order in turn and waits if it arrives before "
        "the window opens. Returns on_time, finish_minute (the minute of the last "
        "delivery), problem (which order fails, and why), and each stop's arrival "
        "and delivery. The list may be just the first orders of a full order, so "
        "you can check a beginning before you extend it. Checking several "
        "candidate orders in one turn is a cheap way to compare their finish "
        "minutes before you submit."
    ),
    "submit_answer": (
        "Submit your final answer, once. Set solvable to true and list every order "
        "id in delivery order, so that every delivery is on time and the last one "
        "is as early as possible. Set solvable to false, with an empty list, only "
        "if no delivery order is on time. The run ends when an answer is accepted."
    ),
}

IMPROVED_TOOL_SPECS = [
    {**spec, "description": IMPROVED_DESCRIPTIONS[spec["name"]]} for spec in TOOL_SPECS
]

# The sets of tool descriptions an agent configuration can choose, by name.
TOOL_SPEC_SETS = {"basic": TOOL_SPECS, "improved": IMPROVED_TOOL_SPECS}
