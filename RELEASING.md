# Releasing the Evalt SDK

From the repository root, build with an isolated tool environment so stale `build/`,
`dist/`, or `*.egg-info` output cannot enter the acceptance result:

```powershell
$sdk = "."
$releaseDist = "release-dist"
if (Test-Path -LiteralPath $releaseDist) {
  Remove-Item -LiteralPath $releaseDist -Recurse -Force
}
python -m unittest discover -s tests -v
python -m build "$sdk" --outdir $releaseDist
python -m twine check "$releaseDist/*"
```

Then create a fresh virtual environment, install only the new wheel, and run:

```powershell
evalt --version
evalt init evalt.json
evalt validate evalt.json
evalt check evalt-result.json --min-pass-rate 0.95
```

`check` requires a real exported result fixture. Expected exit codes are `0` for a
passing gate, `1` for a measured quality/cost/coverage failure, and `2` for invalid
input or a runtime/provider error. Prefer GitHub OIDC Trusted Publishing for releases;
never commit a PyPI token or provider key.
