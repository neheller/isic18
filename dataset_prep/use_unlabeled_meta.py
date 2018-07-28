# ==============================================================================
"""A script to examine the meta which came along with all the data not included
in the task 3 dataset

NOTE: python >= 3.0
"""

from pathlib import Path
import json
import shutil

from progress.bar import Bar

challenge_dict = {
    "melanoma": 0,
    "melanocytic nevus": 1,
    "basal cell carcinoma": 2,
    "actinic keratosis": 3,
    "seborrheic keratosis": 4,
    "solar lentigo": 4,
    "lichenoid keratosis": 4,
    "dermatofibroma": 5,
    "vascular lesion": 6
}

if __name__ == '__main__':
    idir = Path("/home/helle246/data/ISIC18/task3_unlabeled_meta")
    im_dir = Path("/home/helle246/data/ISIC18/task3_unlabeled")
    out_dir = Path("/home/helle246/data/ISIC18/task3_unlabeled_split")
    if not out_dir.exists():
        out_dir.mkdir()
    diagnoses = []
    bar = Bar("Inspecting...", max=sum([1 for _ in idir.glob("*.json")]))
    for json_file in idir.glob("*.json"):
        bar.next()
        with open(str(json_file), 'r') as f:
            data = json.load(f)
            try:
                diagnosis = data["meta"]["clinical"]["diagnosis"]
                if diagnosis is None or diagnosis == "other":
                    diagnosis = "error"
            except KeyError:
                diagnosis = "error"
            try:
                if diagnosis == "nevus" and data["meta"]["clinical"]["melanocytic"]:
                    diagnosis = "melanocytic nevus"
            except KeyError:
                pass
            diagnoses = diagnoses + [diagnosis]
            if diagnosis in challenge_dict:
                out_loc_name = str(challenge_dict[diagnosis])
            else:
                out_loc_name = "unlabeled"
            out_loc = out_dir / out_loc_name
            try:
                image_file = [f for f in im_dir.glob(json_file.stem+"*")][0]
            except IndexError:
                print("Error: %s" % json_file.stem)
            if not out_loc.exists():
                out_loc.mkdir()
            shutil.copy(str(image_file), str(out_loc))
    bar.finish()

    unq = set(diagnoses)
    for u in unq:
        pnt = u
        if u in challenge_dict:
            pnt = pnt + " ({0})".format(challenge_dict[u])
        else:
            pnt = pnt + " (unlabeled)"
        count = 0
        for d in diagnoses:
            if u == d:
                count = count + 1
        print(pnt + ":", count)
