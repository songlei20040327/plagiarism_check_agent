"""Tools layer — document processing, vector search, and report generation."""

from .registry import ToolRegistry

# Import tool modules to trigger registration with ToolRegistry
from . import document_tools  # noqa: F401
from . import report_tools    # noqa: F401
