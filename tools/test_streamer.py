import time
from dataset_streamer import DistributedDatasetStreamer, StreamConfig, load_streaming_tokenizer

print("=" * 70)
print("TESTING STREAMING DATA LOADER (FineWeb-Edu)")
print("=" * 70)

tok = load_streaming_tokenizer("gpt2")
print(f"Tokenizer loaded: vocab_size={len(tok)}")

cfg = StreamConfig(
    dataset_name="HuggingFaceFW/fineweb-edu",
    dataset_subset="sample-10BT",
    seq_len=2048,
    micro_batch_size=2,
    rank=0,
    world_size=4,
    prefetch_buffer_size=16,
    pin_memory=False,
)

print("Initializing DistributedDatasetStreamer...")
streamer = DistributedDatasetStreamer(cfg)

print("Fetching first 3 batches...")
for i in range(3):
    t0 = time.time()
    batch = streamer.get_batch(device="cpu")
    t1 = time.time()
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    print(f"  Batch {i+1}: input_ids shape={input_ids.shape}, labels shape={labels.shape}, time={t1-t0:.3f}s")
    assert input_ids.shape == (2, 2048)

streamer.stop()
print("=" * 70)
print(">> STREAMER VERIFICATION PASSED! <<")
print("=" * 70)
