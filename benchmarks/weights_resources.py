"""Reproduce weighted 100,000-unit fits/aggregation under a 1 GB RSS watchdog."""

from anticipation_resources import main

if __name__ == "__main__":
    main(sampling_weights=True)
