"""r3trace's app: a page that runs r3trace, and the local server behind it (``r3trace app``).

It needs the ``app`` extra.
"""

from r3trace.app.server import create_app, serve

__all__ = ["create_app", "serve"]
