"""
Tool Registry — Decoupled tool registration system.

Tools register themselves with their definitions and handler classes.
The Agent layer reads from the registry without importing tool modules directly.

Usage:
    from .registry import ToolRegistry

    ToolRegistry.register(
        name="parse_pdf",
        description="Parse a PDF file...",
        input_schema={...},
        handler_class=DocumentTools,
        is_write_tool=False,
    )
"""

from typing import Any, Callable, Optional
from copy import deepcopy


class ToolRegistry:
    """
    Global registry for agent tools.

    Decouples the three layers:
    - Tools layer: registers tools (name, description, schema, handler)
    - Agent layer: reads definitions and dispatches calls
    - Skills layer: uses get_tools_summary() to inject into prompts

    Adding a new tool only requires registering it here — no changes
    needed in agent/ or skills/.
    """

    _definitions: list[dict] = []
    _handler_map: dict[str, tuple[type, str]] = {}
    _instances: dict[type, object] = {}
    _write_tools: set[str] = set()

    # ── Registration ──────────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        input_schema: dict,
        handler_class: type,
        method_name: Optional[str] = None,
        is_write_tool: bool = False,
    ):
        """
        Register a tool with the registry.

        Args:
            name: Tool name exposed to the LLM (e.g. "parse_pdf").
            description: Human-readable description for the LLM.
            input_schema: JSON Schema for the tool's parameters.
            handler_class: Class that implements the tool.
            method_name: Method name on handler_class (defaults to `name`).
            is_write_tool: If True, this tool is skipped in dry-run mode.
        """
        method = method_name or name

        # Avoid duplicate registration
        existing_names = {d["name"] for d in cls._definitions}
        if name in existing_names:
            return

        input_schema = deepcopy(input_schema)
        def forbid_extra_fields(schema):
            if isinstance(schema, dict):
                if schema.get("type") == "object":
                    schema.setdefault("additionalProperties", False)
                for value in schema.values():
                    forbid_extra_fields(value)
            elif isinstance(schema, list):
                for value in schema:
                    forbid_extra_fields(value)
        forbid_extra_fields(input_schema)
        cls._definitions.append({
            "name": name,
            "description": description,
            "input_schema": input_schema,
        })
        cls._handler_map[name] = (handler_class, method)

        if is_write_tool:
            cls._write_tools.add(name)

    # ── Queries ───────────────────────────────────────────────────────────

    @classmethod
    def get_definitions(cls) -> list[dict]:
        """Return all registered tool definitions (for LLM function-calling)."""
        return list(cls._definitions)

    @classmethod
    def get_handler(cls, name: str, instances: Optional[dict] = None) -> Optional[Callable]:
        """
        Get a callable handler for the given tool name.
        Lazily instantiates handler classes (singleton per class).
        """
        if name not in cls._handler_map:
            return None

        handler_class, method_name = cls._handler_map[name]

        # The caller owns handler instances; mutable state never crosses routers.
        instances = instances if instances is not None else {}
        if handler_class not in instances:
            instances[handler_class] = handler_class()

        instance = instances[handler_class]
        return getattr(instance, method_name)

    @classmethod
    def is_write_tool(cls, name: str) -> bool:
        """Check if a tool is a write tool (should be skipped in dry-run)."""
        return name in cls._write_tools

    @classmethod
    def get_tools_summary(cls) -> str:
        """
        Generate a human-readable summary of all registered tools.
        Injected into the system prompt so the LLM knows what tools are available
        without SKILL.md hardcoding tool names.
        """
        if not cls._definitions:
            return "(No tools registered)"

        lines = ["## 可用工具", ""]
        for d in cls._definitions:
            name = d["name"]
            desc = d["description"]
            props = d["input_schema"].get("properties", {})
            required = d["input_schema"].get("required", [])

            params = []
            for pname, pinfo in props.items():
                req_mark = " (必填)" if pname in required else ""
                params.append(f"    - `{pname}`{req_mark}: {pinfo.get('description', '')}")

            param_text = "\n".join(params) if params else "    (无参数)"
            lines.append(f"### {name}")
            lines.append(f"{desc}")
            lines.append(f"{param_text}")
            lines.append("")

        return "\n".join(lines)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @classmethod
    def clear(cls):
        """Clear all registered tools. Mainly useful for testing."""
        cls._definitions.clear()
        cls._handler_map.clear()
        cls._instances.clear()
        cls._write_tools.clear()
