# Contributing to VFEEL Reproduction Study

Thank you for your interest in contributing! This is a reproduction study of the VFEEL paper (arXiv:2512.03374). All contributions should maintain scientific rigor and clearly distinguish between:
- **Paper-accurate results**: Results directly derived from the paper's equations and algorithms
- **Reproduction approximations**: Analytical approximations used when end-to-end simulation is impractical

## Ways to Contribute

1. **Report reproduction discrepancies** — If you find the reproduced trends differ from the paper, open an issue with specific figure number and expected vs observed behavior
2. **Improve approximation accuracy** — Suggest better analytical approximations that more closely match the paper's quantitative results
3. **Add end-to-end simulation** — If you have access to the original dataset (EdgeMP / 7-class HAR), contributions for full GPU-based simulation are welcome
4. **Documentation improvements** — Clarify methodology, fix typos, improve readability
5. **Test coverage** — Add unit tests for edge cases in the mathematical models

## Development Workflow

```bash
# 1. Fork and clone
git clone https://github.com/your-username/paper-repro-iscc-vfeel-2025.git
cd paper-repro-iscc-vfeel-2025

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run existing tests (must pass)
pytest tests/ -v

# 5. Make your changes
# ... edit code ...

# 6. Add/update tests
pytest tests/ -v

# 7. Commit and push
git commit -m "feat: describe your change"
git push origin feature/your-feature-name

# 8. Open a Pull Request
```

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Add docstrings to all functions referencing the paper equation number they implement
- Keep functions focused and testable

## Issue Guidelines

When reporting issues, please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Reference to specific paper equation/figure

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New tests added for any new functionality
- [ ] Documentation updated if needed
- [ ] Changes are clearly commented with paper equation references
- [ ] No merge conflicts

## Code of Conduct

Please be respectful and constructive in all interactions. This is an academic reproduction study — precision and transparency are paramount.
