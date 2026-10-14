"""
Tomte logging facade.

Prefer a single import at the top of your module::

    from tomte.engine import tomtelog

    tomtelog.info("ready")

``from tomte.engine.log import tomtelog`` is equivalent; both imports
initialize the parent ``tomte.engine`` package.
"""

from . import tomtelog

__all__ = ["tomtelog"]
