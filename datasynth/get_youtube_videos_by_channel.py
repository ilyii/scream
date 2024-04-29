import argparse
import os

import scrapetube


def main(args):
    outdir = args["outdir"].strip()
    channel_name = args["channel"].strip().lower()
    limit = args.get("limit", None)
    videos = scrapetube.get_channel(
        channel_username=channel_name, sleep=0.1, limit=limit
    )

    try:
        first_video = videos.__next__()
    except Exception:
        print(f"No videos found for channel {channel_name}")
        return

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, f"{channel_name}.txt")

    print(f"Channel: {channel_name}")
    print(f"Saving {limit if limit else 'all'} video URLs to {outfile}")

    with open(outfile, "w") as f:
        f.write(first_video["videoId"] + "\n")
        for video in videos:
            f.write(video["videoId"] + "\n")

    print("Done!")
    return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-c", "--channel", required=True, help="Name of the YouTube channel"
    )
    ap.add_argument("-o", "--outdir", required=True, help="Output Directory")
    ap.add_argument("-l", "--limit", required=False, help="Limit number of videos")
    args = vars(ap.parse_args())
    main(args)
