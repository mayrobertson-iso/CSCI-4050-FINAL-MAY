import urllib.request
import pprint
import json
import requests
import IPython
import librosa
import os
import subprocess
import pickle
import numpy as np
import sys
import re
import musicbrainzngs
import pprint as pp
from datetime import datetime
from thefuzz import fuzz
from thefuzz import process

# a file I have repurposed from another project. It is only here for the search functionality

#--------------------------------SEARCH FUNCTION--------------------------------

def deezer_search(artist_name, track_name):
        

    track_name = "\""+track_name+"\""

    track_name = track_name.replace(" ", "+")
    # artist_name = artist_name.replace(" ", "+")

    # contents = urllib.request.urlopen("https://api.deezer.com/search/track?q=eminem").read()

    print("https://api.deezer.com/search?q=track:"+track_name)
    track_search = urllib.request.urlopen("https://api.deezer.com/search?q=track:"+track_name).read()

    print("SEARCHED")

    # Load the JSON to a Python list & dump it back out as formatted JSON
    data = json.loads(track_search)
    print("LOADED")

    iter=0
    select_index = 0
    output = []
    matches = []

    for x in data["data"]:
        curr_artist = x["artist"]["name"]
        curr_track = x["title"]

        artist_similarity = fuzz.token_sort_ratio(artist_name.lower(), curr_artist.lower())
        track_similarity = fuzz.token_sort_ratio(track_name.lower(), curr_track.lower())

        artist_partial = fuzz.partial_ratio(artist_name.lower(), curr_artist.lower())
        track_partial = fuzz.partial_ratio(track_name.lower(), curr_track.lower())
        
        artist_score = max(artist_similarity, artist_partial)
        track_score = max(track_similarity, track_partial)

        release_date = x.get("release_date", "Unknown")
        
        total_score = (artist_score + track_score) / 2
        
        if total_score >= 60:
            matches.append({
                "track": curr_track,
                "artist": curr_artist,
                "album": x["album"]["title"],
                "id": x["id"],
                "artist_score": artist_score,
                "track_score": track_score,
                "total_score": total_score,
                "preview": x.get("preview", "")
            })

    # Sort by total score
    matches.sort(key=lambda x: x["total_score"], reverse=True)
    print("MATCHES1",matches[0])
          
    # print("MATCHESSSSS",matches)
    if matches:
        for x in matches:
            # print("MATCHES FOUND")
            # print(x)

            # print(x["track"])

            # track = {"track" : x["data"][iter]["title"]}
            # album = {"album" : x["data"][iter]["album"]["title"]}
            # id = {"id" : x["data"][iter]["id"]}

            # a = [{"track" : x["data"][iter]["title"], 
            #      "album" : x["data"][iter]["album"]["title"], 
            #      "id" : x["data"][iter]["id"]}]
        
            output.append({"track" : x["track"], 
                        "artist" : x["artist"],        
                        "album" : x["album"], 
                        "id" : x["id"]})

        
            
            # print("\ntype "+str(select_index)+" to select:")
            # print("Track Name:", x["track"])
            # print("Artist:", x["artist"])
            # print("Album:", x["album"])
            # print("id:", x["id"])
            # select_index+=1
    else:
       return("NO MATCHES FOUND\nPlease check your search again")
    

    lyrics = lyrics_genius(artist_name, track_name)
    
    return(output[0], lyrics)
           


def analyze_selected_track(output):
    # print("\nPlease enter number from 0-"+str(len(output)-1)+" to select a track for analysis!\n")

    # selection = int(input())
    # print("Performing Analysis On --> ",output[int(selection)])
    # print("ID = ",output[selection]['id'])
    track_id = str(output['id'])

    print("ACCESSING DEEZER API")
    contents = urllib.request.urlopen("https://api.deezer.com/track/"+track_id).read()
    data = json.loads(contents)
    print("DATA LOADED")

    track_url = "https://api.deezer.com/track/"+track_id
    preview_url = data["preview"]
    track_name = data["title_short"]
    artist_name = data["artist"]["name"]
    album_title = data["album"]["title"]
    release_date = data["release_date"]

    # with open('x_file.pkl', 'wb') as outf:
    #     pickle.dump([track_url, track_id, preview_url, track_name, artist_name,album_title, release_date], outf)

    print("\nTRACK INFO:")
    print("Track:",track_name,"\nArtist:",artist_name,"\nAlbum Title:", album_title,"\nRelease Date:",release_date)



    '''GET PREVIEW URL'''


    urllib.request.urlretrieve(preview_url, "audio_files/"+track_name+".mp3")




    # convert mp3 to wav file
    subprocess.call(['ffmpeg', '-i', 'audio_files/'+track_name+'.mp3',
                    'audio_files/'+track_name+'.wav'])

    # import required modules



    # wav conversion code
    from os import path
    from pydub import AudioSegment

    # assign files
    input_file = "audio_files/"+track_name+".mp3"
    output_file = "audio_files/"+track_name+".wav"

    # convert mp3 file to wav file
    sound = AudioSegment.from_mp3(input_file)
    sound.export(output_file, format="wav")

    os.remove(input_file)
    # #PICKLING PROCESS
    # # track_id = sys.argv[1]
    # preview_url = data["preview"]
    # track_name = data["title_short"]
    # artist_name = data["artist"]["name"]
    # album_title = data["album"]["title"]
    # release_date = data["release_date"]

    # with open('x_file.pkl', 'wb') as outf:
    #     pickle.dump([track_url, track_id, preview_url, track_name, artist_name,album_title, release_date], outf)

    # print("\nTRACK INFO:")
    # print("Track:",track_name,"\nArtist:",artist_name,"\nAlbum Title:", album_title,"\nRelease Date:",release_date)
    




def librosa_analysis(track_name):
    #--------------------------------LIBROSA ANALYSIS--------------------------------

    # input_file = sys.argv[1]
    IPython.display.Audio('audio_files/'+track_name+".wav")

    y, sr = librosa.load('audio_files/'+track_name+".wav", sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr)

    # Compute the Chroma Short-Time Fourier Transform (chroma_stft)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    # Calculate the mean chroma feature across time
    mean_chroma = np.mean(chromagram, axis=1)
    # Define the mapping of chroma features to keys
    chroma_to_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Find the key by selecting the maximum chroma feature
    estimated_key_index = np.argmax(mean_chroma)
    estimated_key = chroma_to_key[estimated_key_index]





    # Print the detected key
    print("---Track Information---")
    # print(track_name)
    print("Detected Key:", estimated_key)
    print("Tempo: ", tempo)










def musicbrainz_search(artist_name, track_name, album_title):
    #--------------------------------MUSICBRAINZ INFO--------------------------------
    musicbrainzngs.set_useragent("ApplicationName", "0.1")
    print("\nMUSICBRAINZ SEARCH\n")
    releases = []
    print(artist_name)
    print(track_name)
    print(album_title)
    try:
        result = musicbrainzngs.search_recordings(artist=artist_name, recording=track_name, release=album_title, strict=True)
        print("RESULT")
        
        target_album = "The Rise and Fall of a Midwest Princess"
        
        for recording in result['recording-list']:
            # Check all releases for this recording
            for release in recording.get('release-list', []):
                if release['title'] == target_album:
                    print("\nFOUND MATCHING RELEASE\n")
                    
                    # Print recording information
                    print("\nRECORDING INFORMATION:")
                    print("-" * 30)
                    print(f"Recording ID: {recording.get('id')}")
                    print(f"Title: {recording.get('title')}")
                    print(f"Length: {recording.get('length')} ms")
                    print(f"Disambiguation: {recording.get('disambiguation', 'None')}")
                    print(f"Artist: {recording.get('artist-credit-phrase')}")
                    
                    # Print ISRC codes if available
                    if 'isrc-list' in recording:
                        print(f"ISRCs: {', '.join(recording['isrc-list'])}")
                    
                    # Print artist credit details
                    print("\nARTIST CREDITS:")
                    print("-" * 30)
                    for artist_credit in recording.get('artist-credit', []):
                        artist = artist_credit.get('artist', {})
                        print(f"  - {artist.get('name')} (ID: {artist.get('id')})")
                    
                    # Print release information
                    print("\nRELEASE INFORMATION:")
                    print("-" * 30)
                    print(f"Release ID: {release.get('id')}")
                    print(f"Title: {release.get('title')}")
                    print(f"Status: {release.get('status')}")
                    print(f"Disambiguation: {release.get('disambiguation', 'None')}")
                    print(f"Country: {release.get('country', 'Not specified')}")
                    print(f"Date: {release.get('date', 'Not specified')}")
                    
                    # Print release group information
                    if 'release-group' in release:
                        rg = release['release-group']
                        print(f"Release Group: {rg.get('title')} (ID: {rg.get('id')})")
                        print(f"Primary Type: {rg.get('primary-type')}")
                        print(f"Type: {rg.get('type')}")
                    
                    # Print medium/track information
                    print("\nTRACK LISTING:")
                    print("-" * 30)
                    for medium in release.get('medium-list', []):
                        print(f"Medium {medium.get('position')}: {medium.get('format', 'Unknown format')}")
                        print(f"Track count: {medium.get('track-count')}")
                        
                        for track in medium.get('track-list', []):
                            print(f"  Track {track.get('number')}: {track.get('title')} ({track.get('length', 'Unknown length')} ms)")
                    
                    # Print tags if available
                    if 'tag-list' in recording:
                        print("\nTAGS:")
                        print("-" * 30)
                        for tag in recording['tag-list']:
                            print(f"  - {tag.get('name')} (count: {tag.get('count')})")
                    
                    print("=" * 60)
                    print("\n")
                    break  # Found the target album, no need to check other releases for this recording

    except Exception as e:
        print(f"Error: {e}")




def lyrics_genius(artist_name, track_name):
    #--------------------------------MUSICBRAINZ INFO--------------------------------

    from lyricsgenius import Genius

    # artist_name = "Chappell Roan"
    # track_name = "Pink Pony Club"
    # album_title = "The Rise and Fall of a Midwest Princess"

    genius = Genius("VxB86MKPo-MoXweld4GX5zPFWJx27FYR7UaaeG10fm1luhkq0dygrWfDghSHYvcH")
    song = genius.search_song(track_name, artist_name)
    return song.lyrics
