"""Build the daily/event datasets, fit basic time-series models and diagnostics."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    if args.build_only and args.fit_only:
        parser.error("Choose either --build-only or --fit-only")
    if not args.fit_only:
        from milestone2_timeseries.build_data import build
        build()
        from milestone2_timeseries.verify_data import verify
        verify()
    if not args.build_only:
        from milestone2_timeseries.fit_models import run
        run()


if __name__ == "__main__":
    main()
