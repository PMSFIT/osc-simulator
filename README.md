# Baseline Simulator for ASAM OpenSCENARIO XML

[![Build and Test](https://github.com/PMSFIT/osc-simulator/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/PMSFIT/osc-simulator/actions/workflows/build-and-test.yml)
[![Validation](https://github.com/PMSFIT/osc-simulator/actions/workflows/validation.yml/badge.svg)](https://github.com/PMSFIT/osc-simulator/actions/workflows/validation.yml)
[![Validation Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/wiki/PMSFIT/osc-simulator/validation-suite-tests.md)](https://github.com/PMSFIT/osc-simulator/actions/workflows/validation.yml)

This repository provides a baseline simulator for ASAM OpenSCENARIO XML scenarios.
It serves as an adjunct to the [OpenSCENARIO XML Validation Suite](https://github.com/PMSFIT/osc-validation), which provides test cases to validate a scenario engine against a subset of the [ASAM OpenSCENARIO XML standard](https://www.asam.net/standards/detail/openscenario-xml/).

The goal of the baseline simulator is to provide a deminimis implementation of ASAM OpenSCENARIO XML, outputting ASAM OSI trace files, to serve as a validation tool for the validation suite itself.
It is not intended as a suitable basis for production ready scenario engines or for other purposes, and features are taylored towards the core goals of the implementation.

## Build

Requires Python 3.10+.

```bash
python -m pip install --upgrade pip
python -m pip install hatch
hatch build
```

## Install for local development

```bash
python -m pip install -e ".[dev]"
```

## Run

Use the CLI on a `.xosc` file and choose an output directory for generated `.osi` trace files:

```bash
osc-simulator examples/simple_scenario.xosc --output-dir out
```

You can inspect all CLI options with:

```bash
osc-simulator --help
```

Feel free to contact osi@pmsf.eu for further details or information regarding this implementation.
