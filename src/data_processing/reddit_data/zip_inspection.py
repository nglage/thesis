import os
import zipfile
import json
from pathlib import Path

data_path = Path(r"C:\Users\norag\OneDrive\MasterSDS\Thesis\src\data\raw\reddit\subreddits.zip")

with zipfile.ZipFile(data_path, "r") as z:
    # check zip
    files = z.namelist()
    print(f"Total files: {len(files)}")
    print("\nFirst 20 files:")
    for f in files[:20]:
        print(f)
    
    # check format and fields
    first_file = z.namelist()[0]
    print(f"Looking at: {first_file}\n")

    with z.open(first_file) as f:
        for i, line in enumerate(f):
            post = json.loads(line)
            print(f"--- Post {i+1} ---")
            print(f"Keys: {list(post.keys())}")
            print(f"Sample fields:")
            for key in ['id', 'author', 'created_utc', 'subreddit', 
                       'title', 'selftext', 'body', 'score', 'num_comments']:
                if key in post:
                    val = str(post[key])[:100]
                    print(f"  {key}: {val}")
            print()
            if i >= 2:
                break
    
    # check submissions file
    sub_file = "texas_submissions"
    print(f"Looking at: {sub_file}\n")

    with z.open(sub_file) as f:
        for i, line in enumerate(f):
            post = json.loads(line)
            print(f"--- Submission {i+1} ---")
            print(f"Keys: {list(post.keys())}")
            for key in ['id', 'author', 'created_utc', 'subreddit',
                        'title', 'selftext', 'score', 'num_comments',
                        'url', 'is_self', 'domain']:
                if key in post:
                    val = str(post[key])[:150]
                    print(f"  {key}: {val}")
            print()
            if i >= 2:
                break
    
    # check for all subreddit names
    print("\nAll files in zip:")
    for f in z.namelist():
        print(f" ", f)



