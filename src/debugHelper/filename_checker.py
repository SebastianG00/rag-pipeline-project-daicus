import os

DATA_DIR = "data/raw/"

print("Actual filenames in your data/raw/ directory:")
print("=" * 50)

if os.path.exists(DATA_DIR):
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.pdf'):
            print(f"Original filename: {filename}")
            print(f"After .lower():    {filename.lower()}")
            print(f"os.path.basename + lower: {os.path.basename(filename).lower()}")
            print("-" * 30)
else:
    print(f"Directory {DATA_DIR} does not exist!")

# Also check what your ingestion would actually produce
print("\nWhat your ingestion code would produce:")
print("=" * 50)

test_paths = [
    "data/raw/AllstatePolicy.pdf",
    "data/raw/allStatePolicy.pdf", 
    "data/raw/americanFamilyPolicy.pdf"
]

for path in test_paths:
    result = os.path.basename(path).lower()
    print(f"Path: {path}")
    print(f"Result: {result}")
    print("-" * 30)