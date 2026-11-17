# Upload scCS 0.8.1rc2 to GitHub

Extract the release ZIP. Use the directory containing `pyproject.toml`, `scCS/`,
`docs/`, and `tutorials/` as the source directory.

Clone your existing repository separately, then copy the release source into it
while preserving the repository's `.git` directory. Include `.gitignore` and
`.readthedocs.yaml`. Do not copy datasets, caches, tutorial_outputs, environments,
build directories or egg-info.

For example, replace the source path below with your extracted directory:

```bash
git clone https://github.com/mcrewcow/scCS.git
cd scCS
git switch -c release/0.8.1rc2
rsync -av --exclude='.git/' /absolute/path/to/scCS-0.8.1rc2/ ./
git status --short
git diff --stat
git diff --check
git add .
git diff --cached --stat
git commit -m "Prepare scCS 0.8.1rc2"
git push -u origin release/0.8.1rc2
```

Review staged files before committing. If the current repository already tracks
generated artifacts, `.gitignore` will not untrack them: remove those tracked
artifacts explicitly after checking `git ls-files`.

Open the branch on GitHub and create a pull request to your default branch.
Merge after reviewing the changes and checks. Read the Docs should rebuild from
its configured branch once the changes are merged; inspect its build log.

After merging, tag the reviewed commit:

```bash
git switch main
git pull --ff-only
git tag -a v0.8.1rc2 -m "scCS 0.8.1rc2"
git push origin v0.8.1rc2
```

Use your actual default branch name if it differs from `main`. Create a GitHub
release from `v0.8.1rc2`, mark it **Pre-release**, paste RELEASE_NOTES_0.8.1rc2.md,
and attach the full source ZIP. Publishing a GitHub release does not upload to PyPI.
