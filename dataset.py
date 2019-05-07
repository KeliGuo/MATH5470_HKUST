import os
import glob
import pandas as pd
import tqdm
import numpy as np

import spotipy
from spotipy import oauth2
import spotipy.util as util
import billboard



def get_hot_100(begin='1990-01-01', save_path='./tmp'):
    '''
    Download the hot_100 data
    begin: the beginning time of our collection
    '''
    
    chartTable = pd.DataFrame(columns=['title', 'artist', 'Year', 'Month', 'rank', 'isNew', 'weeks', 'lastPos', 'peakPos'])

    chart = billboard.ChartData('hot-100')
    year = '2019'
    
    # iteratively get the last chart before "begin"
    while chart.previousDate>=begin:
        date = chart.date
        Year, Month = date.split('-')[:2]
        
        if not date.startswith(year):
            print(date)
            year = date.split('-')[0] 
            
        for i in range(100):
            track = chart[i]
            title = track.title
            # if we have encounter this song before, skip it
            if title in chartTable.title.tolist():
                continue
            artist = track.artist
            rank = track.rank
            weeks = track.weeks
            lastPos = track.lastPos
            peakPos = track.peakPos
            isNew = track.isNew
            
            chartTable.loc[chartTable.shape[0]+1] = [title, artist, Year, Month, rank, isNew, weeks, lastPos, peakPos]
        
        chart = billboard.ChartData('hot-100', chart.previousDate)
        
    # save the file
    chartTable.to_excel(os.path.join(save_path, 'hot_100.xlsx'))


        
def get_MSD(save_path='./tmp'):
    '''
    MSD data is about 1.8G, but we only use the AdditionalFiles/subset_unique_tracks.txt
        since we only need the track title and its artist
        This subset of dataset includes 10000 tracks totally
    '''
    
    MSD = pd.DataFrame(columns=['title', 'artist'])
    path = './MSD/AdditionalFiles/subset_unique_tracks.txt'
    with open(path, 'r') as file:
        lines = file.read().split('\n')
        for line in lines:
            if line:
                artist, title = line.split('<SEP>')[-2:]
                MSD.loc[MSD.shape[0]+1] = [title, artist]
    MSD.to_excel(os.path.join(save_path, 'MSD_10000.xlsx'))
    


def get_track_feature(artist, title):
    '''
    Search the spotify feature of given artist+title
    Features are listed as below
    '''
    
    track_info = sp.search('artist:'+artist+' track:'+title)
    items = track_info['tracks']['items']
    
    if items:
        year, *month = items[0]['album']['release_date'].split('-')
        if month:
            month = month[0]
        else:
            month = None
        track_id = items[0]['id']
        
        feat = sp.audio_features(tracks=track_id)
        
        if feat:
            danceability = feat[0]['danceability']
            energy = feat[0]['energy']
            speechiness = feat[0]['speechiness']
            acousticness = feat[0]['acousticness']
            instrumentalness = feat[0]['instrumentalness']
            liveness = feat[0]['liveness']
            valence = feat[0]['valence']
            loudness = feat[0]['loudness']
            tempo = feat[0]['tempo']
            key = feat[0]['key']
            mode = feat[0]['mode']
            
            if year<'1990':
                return None
            else: 
                return [year,
                        month,
                        danceability,
                        energy,
                        speechiness,
                        acousticness,
                        instrumentalness,
                        liveness,
                        valence,
                        loudness,
                        tempo,
                        key,
                        mode]
    
    return None
    


def get_spotify_feature(save_path='./tmp'):
    '''
    Get the features of all tracks in hot-100 and MSD datasets
    '''
       
    MSD = pd.read_excel(os.path.join(save_path, 'MSD_10000.xlsx'))
    hot_100 = pd.read_excel(os.path.join(save_path, 'hot_100.xlsx'))
    hot_100.sort_values(by=['Year', 'Month'], inplace=True, ascending=True)
    
    feature = pd.DataFrame(columns=['Title', 
                                    'Artist',
                                    'Year', 
                                    'Month', 
                                    'Danceability', 
                                    'Energy',
                                    'Speechiness', 
                                    'Acousticness', 
                                    'Instrumentalness', 
                                    'Liveness', 
                                    'Valence',
                                    'Loudness', 
                                    'Tempo',
                                    'Key',
                                    'Mode'])
    
    labels = []    
    tracks = []
    
    for i in tqdm.tqdm(range(hot_100.shape[0]), desc='Finding features of the hot_100'):        
        row = hot_100.iloc[i]
        title = row['title']
        artist = row['artist'] 
        # if we have already encounter this track, we skip it
        if title in tracks:
            continue
        tracks.append(title)
        if type(title) is str:
            feat = get_track_feature(artist, title)    
        else:
            continue
        # we add it to feature only when feat is valid (not None)
        if feat:
            labels.append(1)
            feature.loc[feature.shape[0]+1] = [title, artist] + feat


    for i in tqdm.tqdm(range(MSD.shape[0]), desc='Finding features of MSD'):        
        row = MSD.iloc[i]
        title = row['title']
        artist = row['artist']        
        if title in tracks:
            continue        
        tracks.append(title) 
        if type(title) is str:
            feat = get_track_feature(artist, title)
        else:
            continue
        if feat:
            labels.append(0)
            feature.loc[feature.shape[0]+1] = [title, artist] + feat
    
    feature['label'] = labels
    feature.to_excel(os.path.join(save_path, 'feature_1990_2019.xlsx'))
    


def add_artist_score(save_path='./tmp'):  
    '''
    We add the artist_score feature to our data.
    We sort the time value by ascending order. 
    The we scan each track (row), and perform the following logic:
        if the artist appeared in hot_artist before:
            artist_score = 1
        else if this track is hot-100 song (label as 1):
            we add its artist to hot_artist,
            and we set artist_score = 0
        else if this track is not hot-100 song (label as 0)
            we set artist_score = 0
    '''
    
    feature = pd.read_excel(os.path.join(save_path, 'feature_1990_2019.xlsx'))
    feature.sort_values(by=['Year', 'Month'], inplace=True)
    
    hot_artist = []
    artist_score = []
        
    for i in range(feature.shape[0]):
        row = feature.iloc[i]
        label = row['label']
        artist = row['Artist']
        if artist in hot_artist:
            artist_score.append(1)
        elif label==1:
            hot_artist.append(artist)
            artist_score.append(0)
        else:
            artist_score.append(0)
    
    feature['Artist_Score'] = artist_score    
    feature.to_excel(os.path.join(save_path,'feature_complete_1990_2019.xlsx'),index=False)    


def normalize(x):
    return (x - x.mean())/(x.std()+1e-16)


def get_normalized_feature(save_path='./tmp'): 
    '''
    normalize data to have zero mean and 1 std
    '''
    feature = pd.read_excel(os.path.join(save_path,'feature_complete_1990_2019.xlsx'))   
    
    feature['Danceability'] = normalize(feature['Danceability'])
    feature['Energy'] = normalize(feature['Energy'])
    feature['Speechiness'] = normalize(feature['Speechiness'])
    feature['Acousticness'] = normalize(feature['Acousticness'])
    feature['Instrumentalness'] = normalize(feature['Instrumentalness'])
    feature['Liveness'] = normalize(feature['Liveness'])
    feature['Valence'] = normalize(feature['Valence'])
    feature['Loudness'] = normalize(feature['Loudness'])
    feature['Tempo'] = normalize(feature['Tempo'])
    feature['Key'] = normalize(feature['Key'])
    
    feature.to_excel(os.path.join(save_path,'feature_complete_normalized_1990_2019.xlsx'), index=False)
    

def get_train_test_set(save_path='./tmp'):
    '''
    Split the data into train_set and test_set by ratio 75:25,
        and store them into corresponding folder 
    '''
    feature = pd.read_excel(os.path.join(save_path,'feature_complete_normalized_1990_2019.xlsx'))  
    mask = np.arange(feature.shape[0])
    np.random.shuffle(mask)
    split = int(mask.shape[0]*0.75)
    train_set = feature.iloc[mask[:split]]
    test_set = feature.iloc[mask[split:]]
    train_set.to_excel('./train_set/train.xlsx',index=False)
    test_set.to_excel('./test_set/test.xlsx',index=False)
    
    
    
if __name__ == '__main__':
    
    save_path = './tmp'
    
    get_hot_100(begin='1990-01-01', save_path=save_path)
    get_MSD(begin='1990-01-01', basedir='./MSD/data/', save_path=save_path)
        
    token = util.oauth2.SpotifyClientCredentials(client_id='81da2e7e661749f0a86024a8c2a04dd3', client_secret='3d4324c79f2a441bb64c7a55d6a226fa')
    cache_token = token.get_access_token()
    sp = spotipy.Spotify(auth=cache_token)
    
    get_spotify_feature(save_path=save_path)
    
    add_artist_score(save_path)
    
    get_normalized_feature(save_path)

    get_train_test_set(save_path)

