import os
import glob

print("=" * 60)
print("CLUSTER TRAINING STATUS CHECK")
print("=" * 60)

# Check checkpoints
ckpts = sorted(glob.glob("/home/ubuntu/demsni/checkpoints/*.pt"))
print(f"Checkpoints found ({len(ckpts)}):")
for c in ckpts:
    sz = os.path.getsize(c) / (1024**3)
    mtime = os.path.getmtime(c)
    print(f"  {os.path.basename(c)}: {sz:.2f} GB (mtime: {mtime})")

# Check training log
log_files = sorted(glob.glob("/home/ubuntu/demsni/logs/training_*.log"))
if log_files:
    latest_log = log_files[-1]
    print(f"\nLatest log: {latest_log} ({os.path.getsize(latest_log)} bytes)")
    with open(latest_log, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20000))
        lines = f.read().decode("utf-8", errors="ignore").splitlines()
        print("\nLast 20 log lines:")
        for line in lines[-20:]:
            print(" ", line)

print("=" * 60)
