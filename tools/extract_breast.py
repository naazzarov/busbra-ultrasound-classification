#!/usr/bin/env python3
"""Download the official public BrEaST release and extract a separate 102-feature table.

All source files, case identifiers and feature rows stay in the private output folder.
This does not reconstruct the missing BUS-BRA extraction environment.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import shutil
import http.client
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE='https://www.cancerimagingarchive.net/wp-content/uploads/'
FILES={'images':'BrEaST-Lesions_USG-images_and_masks-Dec-15-2023.zip',
       'clinical':'BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx'}
CLASSES=('firstorder','shape2D','glcm','glrlm','glszm','gldm','ngtdm')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_clinical(path):
    needed={'Case_ID','Image_filename','Mask_tumor_filename','Classification'}
    canonical={''.join(c for c in name.lower() if c.isalnum()):name for name in needed}
    observed={}
    for sheet in pd.ExcelFile(path).sheet_names:
        table=pd.read_excel(path,sheet_name=sheet)
        observed[sheet]=[str(column) for column in table]
        table=table.rename(columns={column:canonical.get(''.join(c for c in str(column).lower() if c.isalnum()),column) for column in table})
        if needed<=set(table):
            if len(table)!=256 or not table.Case_ID.is_unique:raise ValueError('Unexpected source case coverage')
            return table
    raise ValueError('Clinical sheet does not match the documented schema; headers only: '+json.dumps(observed))


def feature_labels(table):
    labels=table.Classification.astype(str).str.strip().str.lower()
    counts=labels.value_counts().to_dict()
    if counts!={'benign':154,'malignant':98,'normal':4}:raise ValueError('Unexpected source diagnosis counts')
    keep=labels!='normal'
    cohort=table.loc[keep].copy()
    cohort['diagnosis']=(labels[keep]=='malignant').astype(int)
    return cohort,counts


def download(url, path, *, attempts=3, timeout=60):
    """Retry transient source failures; publish only a complete, exclusive file."""
    path = Path(path)
    if attempts < 1:
        raise ValueError('attempts must be positive')
    if path.exists():
        raise FileExistsError(path)
    request = urllib.request.Request(url, headers={'User-Agent': 'BUSBRA-independent-research/1.0'})
    for attempt in range(attempts):
        # The temporary file is owned by this attempt, and never used as a source.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.part') as output:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    shutil.copyfileobj(response, output)
                    expected = response.headers.get('Content-Length')
                    if expected is not None and output.tell() != int(expected):
                        raise http.client.IncompleteRead(b'', int(expected))
            except urllib.error.HTTPError as error:
                if error.code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
                if attempt + 1 == attempts:
                    raise
            else:
                output.flush()
                # Hard linking publishes atomically and refuses to overwrite an existing path.
                path.hardlink_to(output.name)
                return
        logging.warning('Public data download interrupted; retrying attempt %s/%s', attempt + 2, attempts)
        time.sleep(2 ** (attempt + 1))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.output
    if root.exists():p.error('Output exists; preserve prior source and extraction records')
    root.mkdir(parents=True)
    for name in FILES.values():download(BASE+name,root/name)
    raw=root/'images';raw.mkdir()
    with zipfile.ZipFile(root/FILES['images']) as archive:
        for entry in archive.infolist():
            target=(raw/entry.filename).resolve()
            if not target.is_relative_to(raw.resolve()):raise ValueError('Unsafe archive member')
        archive.extractall(raw)
    import SimpleITK as sitk
    import radiomics
    from PIL import Image
    from radiomics.featureextractor import RadiomicsFeatureExtractor
    if radiomics.__version__.lstrip('v')!='3.0.1':raise ValueError('PyRadiomics version differs from extraction protocol')
    radiomics.setVerbosity(logging.ERROR)
    extractor=RadiomicsFeatureExtractor(binWidth=16,force2D=True,force2Ddimension=0,label=1,normalize=False)
    extractor.disableAllImageTypes();extractor.enableImageTypeByName('Original')
    extractor.disableAllFeatures()
    for name in CLASSES:extractor.enableFeatureClassByName(name)
    cohort,counts=feature_labels(read_clinical(root/FILES['clinical']))
    def locate(value):
        paths=list(raw.rglob(Path(str(value)).name))
        if len(paths)!=1:raise ValueError('Image or tumor mask missing/ambiguous in official archive')
        return paths[0]
    records=[];seen=set();names=None
    for _,row in cohort.iterrows():
        pixels=np.asarray(Image.open(locate(row.Image_filename)).convert('L'))
        mask=(np.asarray(Image.open(locate(row.Mask_tumor_filename)).convert('L'))>0).astype(np.uint8)
        if pixels.shape!=mask.shape or mask.sum()<2:raise ValueError('Invalid source image/mask geometry')
        pixel_hash=hashlib.sha256(np.asarray(pixels.shape,dtype='<i8').tobytes()+pixels.tobytes()).hexdigest()
        if pixel_hash in seen:raise ValueError('Duplicate image pixels found; do not silently remove cases')
        seen.add(pixel_hash)
        image=sitk.GetImageFromArray(pixels);region=sitk.GetImageFromArray(mask)
        features=extractor.execute(image,region)
        numeric={k:float(v) for k,v in features.items() if k.startswith('original_')}
        if len(numeric)!=102 or not np.isfinite(list(numeric.values())).all():raise ValueError('Incomplete or nonfinite extraction')
        if names is None:names=sorted(numeric)
        if set(names)!=set(numeric):raise ValueError('Inconsistent feature schema')
        records.append({'image_id':str(row.Image_filename),'patient_id':f"breast-patient-{row.Case_ID}",
                        'diagnosis':int(row.diagnosis),**numeric})
    frame=pd.DataFrame(records)[['image_id','patient_id','diagnosis',*names]]
    frame.to_csv(root/'features.csv',index=False)
    provenance={'dataset':'BrEaST','doi':'10.7937/9WKK-Q141','license':'CC BY 4.0',
        'source_sha256':{key:digest(root/name) for key,name in FILES.items()},
        'feature_csv_sha256':digest(root/'features.csv'),'source_cases':256,'normal_cases_excluded':4,
        'n_cases':252,'n_benign':154,'n_malignant':98,'n_features':102,'duplicate_pixel_hashes':0,
        'extraction_failures':0,'versions':{n:version(n) for n in ['pyradiomics','SimpleITK','numpy','pandas','PyWavelets','Pillow','openpyxl']},
        'protocol_sha256':digest(Path(__file__).resolve().parents[1]/'evaluation/external_ultrasound_protocol.json')}
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Public cohort extraction completed:',len(frame),'cases;',len(names),'features; no row data logged')


if __name__=='__main__':main()
