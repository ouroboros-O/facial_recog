import os


for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(os.path.join('data', 'neg'), file)
        os.replace(EX_PATH, NEW_PATH)