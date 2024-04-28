import scrapetube

CHANNEL_URL = "https://www.youtube.com/@germania"
OUTFILE = "urls.txt"

videos = scrapetube.get_channel(channel_url=CHANNEL_URL)

for video in videos:
    with open(OUTFILE, "a+") as f:
        f.write(video['videoId'] + "\n")
    

