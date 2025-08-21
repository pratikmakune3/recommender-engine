import gensim.downloader as api

model = api.load("glove-wiki-gigaword-100")

v_banana = model["banana"]

v_apple = model["apple"]
v_aeroplane = model["aeroplane"]

similarity_banana_all = model.cosine_similarities(v_banana, [v_banana, v_apple, v_aeroplane])
print(similarity_banana_all) # [1.0000001  0.5054469  0.12576418]

v_king = model["king"]
v_queen = model["queen"]
v_man = model["man"]
print(model.most_similar(v_man + v_queen - v_king, topn=1)) # woman

v_france = model["france"]
v_germany = model["germany"]
v_berlin = model["berlin"]

print(model.most_similar(v_france + v_berlin - v_germany, topn=1)) # paris