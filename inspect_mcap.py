"""Run this from the project root to see what topics are in your MCAP files."""
import sys
from pathlib import Path
from mcap.reader import make_reader

def inspect(mcap_path: str):
    print(f"\n{'='*60}")
    print(f"File: {mcap_path}")
    print('='*60)
    topics = {}
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages():
            t = channel.topic
            if t not in topics:
                topics[t] = {"schema": schema.name if schema else "?", "count": 0}
            topics[t]["count"] += 1

    for topic, info in sorted(topics.items()):
        print(f"  {info['count']:>6} msgs  [{info['schema']}]  {topic}")

if __name__ == "__main__":
    data_dir = Path("data/hackathon")
    files = list(data_dir.glob("*.mcap"))
    if not files:
        print("No MCAP files found in data/hackathon/")
        sys.exit(1)
    # Inspect just the first file — they should all share the same schema
    inspect(str(files[0]))
    print(f"\nDone. Inspected: {files[0].name}")
