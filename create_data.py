import csv
import os

input_path = "brats_processed"
with open('data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["path", "tumor", "id", "label"])
    for label in os.listdir(input_path):
        for patient in os.listdir(os.path.join(input_path, label)):
            label_id = 0 if label == "LGG" else 1
            writer.writerow([os.path.join(input_path, label, patient, 'hdbet_brats-space', f'{patient}_hdbet_brats_fla.nii.gz'), os.path.join(input_path, label, patient, 'hdbet_brats-space', 'tumor_centered.pt'), patient, label_id])
