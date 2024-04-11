#!/bin/bash
pytest .
mypy .
pylint zxpass/ test/
