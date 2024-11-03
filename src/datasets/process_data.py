

# zip 2 folders annotations and mvsa into a single zip file

import os
import zipfile

def zipdir(folders, ziph):
    # ziph is zipfile handle
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            for file in files:
                ziph.write(os.path.join(root, file))

folders = ['annotations', 'mvsa', 'train']
zipf = zipfile.ZipFile('annotations_and_mvsa.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir(folders, zipf)
zipf.close()