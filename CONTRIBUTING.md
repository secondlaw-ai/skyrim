# ✌️ Contributing to Secondlaw Research

Hey! Thanks for checking out Skyrim, your support would be invaluable advance the frontier of AI applied to chaotic systems, first step is taming the weather and this is Skyrim.

## 🤝 Submitting a contribution

Steps:

1. Fork and clone this repository
2. Do the changes on your fork
3. Ensure all tests pass & pre-commit installed (see below).
4. When all is done, send in your PR!

We are following more or less this [guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

## ⚡️ Setting up your dev environment

### Pre-commit

```bash
pre-commit install
```

### Installation

If you plan to work on the non-core (i.e. libs etc.) this would be sufficient:

```bash
make install
```

For full core contribution:

```bash
make install-core
```

### Linting

```bash
make lint
```

Ensure the linter does not have any warnings before submitting a pull request.

### Formatting

```bash
make format
```

### 🧪 Testing

For unittest only:

```bash
make test-unit
```

For all tests:

```bash
make test
```

## 🔥 Release Process

Currently, it done by the core team and not automated, but will keep here updated if there is any change.
