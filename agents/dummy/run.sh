#!/bin/bash
# Pipeline check without an LLM: submits the sample submission.
cp sample_submission.csv submission.csv && python validate.py submission.csv
