from brats_toolkit.preprocessor import Preprocessor
import os
import tqdm

input_path = "data"
output_path = "brats_processed"

if not os.path.exists(output_path):
    os.makedirs(output_path)

for label in os.listdir(input_path):
    if not os.path.exists(os.path.join(output_path, label)):
        os.makedirs(os.path.join(output_path, label))
    for patient in tqdm.tqdm(os.listdir(os.path.join(input_path, label))):
        if not os.path.exists(os.path.join(output_path, label, patient)):
            os.makedirs(os.path.join(output_path, label, patient))
        
        flair_path = os.path.join(input_path, label, patient, "flair.nii.gz")
        prep = Preprocessor()
        prep.single_preprocess(t1File=flair_path, t1cFile=flair_path, 
                               t2File=flair_path, flaFile=flair_path, 
                               outputFolder=os.path.join(output_path, label, patient), mode="cpu", confirm=False, skipUpdate=True)

