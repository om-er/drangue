# Contributing to drangue

Thanks for considering it. Two documents define what fits here:

- `ROADMAP.md`: where the core is going and why.
- `ECOSYSTEM.md`: the seam contract. Most new capability belongs in a
  battery behind a seam, not in the core. If your idea needs a core change,
  open an issue about the seam first; a battery is never the reason the core
  grows.

## Developing

```bash
pip install -e ".[dev]"
python run_tests.py              # core suite (offline, no API keys)
python extensions/run_tests.py   # extension suites
```

Tests are plain async functions run by a tiny runner; no pytest required.
Keep changes offline-testable: the suite must pass with no network and no
credentials, because CI holds it to that.

## License and sign-off (DCO)

drangue is MIT licensed, and contributions are accepted under MIT. To keep
the provenance of every line clear, this project uses the
[Developer Certificate of Origin](https://developercertificate.org/) (DCO):
you certify that you wrote your contribution or otherwise have the right to
submit it under the project's license.

Certifying is one flag. Sign off each commit:

```bash
git commit -s
```

which adds a `Signed-off-by: Your Name <you@example.com>` trailer. CI checks
that every commit in a pull request carries one. Forgot it? Amend and
force-push:

```bash
git commit --amend -s --no-edit && git push -f
```

### Developer Certificate of Origin 1.1

```
By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
