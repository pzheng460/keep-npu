# Releasing KeepNPU

KeepNPU publishes the normalized distribution name `keep-npu`. A release
provides these commands after `python -m pip install keep-npu`:

- `keep-npu`
- `keep-npu-mcp-server`
- `keep-npu-alive`

## One-time PyPI setup

Configure a pending trusted publisher for the `keep-npu` project with:

- PyPI project name: `keep-npu`
- GitHub owner: `pzheng460`
- GitHub repository: `keep-npu`
- Workflow filename: `publish-pypi.yml`
- Environment name: `pypi`

No long-lived PyPI token is stored in GitHub. PyPI verifies the GitHub Actions
OIDC identity when the publish job runs.

## Publish a version

1. Update `__version__` and the matching `tool.bumpversion.current_version`.
2. Run the Python, dashboard, build, and installed-wheel checks.
3. Push the release commit and create a GitHub release tagged `vX.Y.Z`.
4. The release event builds a fresh sdist and wheel, then publishes both through
   PyPI trusted publishing.

The workflow can also be started manually after the same tag/version checks.
