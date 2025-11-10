import pickle
import os
from PIL import Image, ImageOps
import matplotlib.image as mpimg
from textwrap import wrap
import pandas as pd

# Load jsondata pickle which may be either a pandas DataFrame (wikiart)
# or a list of dicts (our car dataset). Normalize to a DataFrame for
# easier lookup, but keep track of which format we're handling.
jsondata_raw = pickle.load(open(os.path.join("pickles", "jsondata.pkl"), 'rb'))
if isinstance(jsondata_raw, list):
    # list of dicts: expect keys like 'resized_name' and 'label'
    jsondata = pd.DataFrame(jsondata_raw)
    _is_car_dataset = ('resized_name' in jsondata.columns and 'label' in jsondata.columns)
else:
    # assume DataFrame-like
    try:
        jsondata = pd.DataFrame(jsondata_raw)
    except Exception:
        # fallback: convert to empty DataFrame
        jsondata = pd.DataFrame()
    _is_car_dataset = False

# Keep the original wikiart replacements only if those columns exist
if 'artistUrl' in jsondata.columns:
    try:
        jsondata.loc[jsondata.artistUrl == "ethel-reed", "artistName"] = "Ethel Reed"
        jsondata.loc[jsondata.artistUrl == "ancient-greek-painting", "artistName"] = "Ancient Greek Painting"
    except Exception:
        pass


def get_image(name):
    # Try images/ first, then resized/ as a fallback
    for d in ("images", "resized"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            img = Image.open(p)
            return img.convert('RGB')
    # If not found, raise a clear error
    raise FileNotFoundError(f"Image file not found in images/ or resized/: {name}")

def add_border(name, border):
    img = ImageOps.expand(name, border = 2, fill = "white")
    return ImageOps.expand(img, border = (border - 2))

def get_label(name):
    try:
        # Keep original name for error messages
        orig_name = name
        # Remove file extension if present
        if len(name) > 4 and name[-4] in ('.'):
            name_no_ext = name[:-4]
        else:
            name_no_ext = name

        # If this is the car dataset normalized to DataFrame
        if _is_car_dataset:
            # Match by resized filename
            row = jsondata[jsondata['resized_name'] == name]
            if not row.empty:
                lbl = str(row.iloc[0]['label'])
                # Make label more readable
                pretty = lbl.replace('_', ' ')
                return pretty, name_no_ext
            else:
                # fallback: return filename as title
                return "Car Image", name_no_ext

        # Otherwise, try wikiart style metadata
        # Split only once at the first underscore
        parts = name_no_ext.split("_", 1)
        if len(parts) < 2:
            # Unknown format
            return "Unknown Artist", f"Unknown Title ({name_no_ext})"

        artist_url, image_url = parts

        # Filter metadata if available
        if 'artistUrl' in jsondata.columns and 'url' in jsondata.columns:
            x = jsondata.loc[(jsondata.artistUrl == artist_url) & (jsondata.url == image_url), :].reset_index(drop=True)
            if not x.empty:
                title = str(x.iloc[0].get("title", f"{image_url}"))
                artist = str(x.iloc[0].get("artistName", f"{artist_url}"))
                return artist, title

        # If metadata missing or no match
        return "Unknown Artist", f"Unknown Title ({name_no_ext})"

    except Exception as e:
        print(f"[Error in get_label for {name}]: {e}")
        return "Unknown Artist", f"Unknown Title ({name})"


def set_axes(ax, image_name, query = False, **kwargs):
    value = kwargs.get("value", None)
    model = kwargs.get("model", None)
    artist, title = get_label(image_name)
    title = '\n'.join(wrap(title, 50))
    if query:
        ax.set_xlabel("{2} Query Image\n{0}\n{1}".format(artist, title, model.replace("_", " ").title()), fontsize = 12)
    else:
        ax.set_xlabel("Similarity value {2:1.3f}\n{0}\n{1}".format(artist, title, value), fontsize = 12)
    ax.set_xticks([])
    ax.set_yticks([])
    
