import pickle

with open('/path/to/SEVA/syncnet_python/test/pywork/test/activesd.pckl', 'rb') as f:
    tracks = pickle.load(f)

print(type(tracks))
print(tracks)
