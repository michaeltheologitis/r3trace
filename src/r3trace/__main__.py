"""``python -m r3trace``: the command line, as ``r3trace`` is, except in the Mac app, which
Briefcase starts this way: there it opens the app's window (:mod:`r3trace.app.window`)."""

from importlib import metadata

# Briefcase writes an "r3trace" distribution of its own into the Mac app, as its
# INSTALLER says; pip and uv install one of that name too.
MAC_APP_INSTALLER = "briefcase"

if __name__ == "__main__":
    installers = {
        (found.read_text("INSTALLER") or "").strip()
        for found in metadata.distributions(name="r3trace")
    }
    if MAC_APP_INSTALLER in installers:
        from r3trace.app import window

        window.main()
    else:
        from r3trace import cli

        raise SystemExit(cli.main())
