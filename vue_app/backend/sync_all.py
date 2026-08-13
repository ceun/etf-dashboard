import sys

from .core_loader import core


def main():
    if not core.DATABASE_URL:
        print("DATABASE_URL_POOLER or DATABASE_URL is not configured", file=sys.stderr)
        return 1
    targets = core.load_targets_from_db()
    if not targets:
        print("No targets found", file=sys.stderr)
        return 1
    failures = []
    total_written = 0
    for target in targets.values():
        try:
            _, _, written = core.sync_target_data(target["index_code"])
            total_written += int(written or 0)
            print(f"OK {target['name']} ({target['index_code']}): {written}", flush=True)
        except Exception as error:
            failures.append(target["index_code"])
            print(f"FAIL {target['name']} ({target['index_code']}): {error}", file=sys.stderr, flush=True)
    print(f"Completed {len(targets) - len(failures)}/{len(targets)}; wrote {total_written} rows", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
