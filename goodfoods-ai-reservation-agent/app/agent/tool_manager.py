import inspect
import logging
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Central registry for all available tools.
    Each tool must register with:
        - a name (string)
        - a callable function
        - an argument schema (dict of expected arg names/types)
    """

    def __init__(self):
        self.registry = {}

    def register(self, name: str, fn: Callable, schema: dict):
        """Register a new tool with name, function, and expected argument schema."""
        if not callable(fn):
            raise ValueError(f"Tool {name} is not callable")
        self.registry[name] = {"fn": fn, "schema": schema}
        logger.info(f"Registered tool: {name}")

    def register_all(self, tool_functions: Dict[str, Callable], tool_spec: Dict[str, Dict[str, Any]]):
        """Bulk register tools using TOOL_FUNCTIONS and TOOL_SPEC definitions."""
        for name, fn in tool_functions.items():
            schema = {}
            if name in tool_spec:
                # Default all args to string if not typed
                schema = {arg: str for arg in tool_spec[name].get("args", [])}
            self.register(name, fn, schema)
        logger.info(f"Registered {len(tool_functions)} tools successfully")

    def get_tool(self, name: str):
        """Return tool entry by name or None."""
        return self.registry.get(name)

    def list_tools(self):
        """Return a list of all registered tools."""
        return list(self.registry.keys())

    def execute(self, name: str, **kwargs):
        """Execute a registered tool, validating required arguments."""
        tool = self.get_tool(name)
        if not tool:
            return {"ok": False, "error": f"Tool '{name}' not found"}

        schema = tool["schema"]
        fn = tool["fn"]

        # Validate required args
        missing = [k for k in schema.keys() if k not in kwargs]
        if missing:
            return {"ok": False, "error": f"Missing args: {missing}"}

        # Type validation (optional)
        for key, expected_type in schema.items():
            if kwargs.get(key) is not None and not isinstance(kwargs[key], expected_type):
                logger.warning(f"Arg type mismatch for {key}: expected {expected_type}, got {type(kwargs[key])}")

        try:
            result = fn(**kwargs)
            return {"ok": True, "result": result}
        except Exception as e:
            logger.exception(f"Tool execution failed for {name}")
            return {"ok": False, "error": str(e)}
