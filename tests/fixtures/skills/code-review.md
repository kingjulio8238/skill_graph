---
description: Code review a pull request
allowed-tools: Bash(gh pr view:*), Bash(gh pr diff:*), Bash(gh pr comment:*)
category: code-quality
depends-on: []
---

Provide a code review for the given pull request.

Review the changes for bugs, style issues, and compliance.
Use [[static-analysis]] tools to catch common patterns before manual review.

## Review Process

Launch parallel agents to independently review different aspects.
Apply [[code-quality-standards]] to evaluate naming, structure, and documentation.
Check that [[test-coverage]] requirements are met for modified code.

## Feedback

Filter false positives and comment on the PR with findings.
When suggesting improvements, reference [[refactoring-patterns]] for established approaches.
