from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import json
import shutil
from sampling_allocation import ProportionalAllocator, NeymanAllocator
from unbiased_estimation import AreaEstimator
import tempfile
import pandas as pd

app = FastAPI()

# In-memory storage for sampling design
sampling_design = {}

class SamplingDesignInput(BaseModel):
    method: str
    expected_uas: dict
    target_error: float

@app.post("/create_sampling_design")
def create_sampling_design(input_data: SamplingDesignInput):
    global sampling_design
    
    if input_data.method == 'proportional':
        allocator = ProportionalAllocator(
            expected_uas=input_data.expected_uas,
            target_error=input_data.target_error
        )
    elif input_data.method == 'neyman':
        allocator = NeymanAllocator(
            expected_uas=input_data.expected_uas,
            target_error=input_data.target_error
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid sampling method")
    
    sampling_design = {
        "method": input_data.method,
        "expected_uas": input_data.expected_uas,
        "target_error": input_data.target_error,
        "allocator": allocator
    }
    return {"message": "Sampling design created successfully"}

@app.put("/adapt_sampling_design")
def adapt_sampling_design(new_design: SamplingDesignInput):
    global sampling_design
    return create_sampling_design(new_design)

@app.post("/create_sampling_sets")
def create_sampling_sets():
    if not sampling_design:
        raise HTTPException(status_code=400, detail="No sampling design found")
    
    estimator = AreaEstimator(
        raster_path="path/to/raster",
        mask_paths=["path/to/mask"],
        class_merge_dict={},
        epsg=4326,
        sample_allocator=sampling_design["allocator"]
    )
    estimator.prepare_raster()
    sampling_sets = estimator.create_sample_sets([])
    return {"sampling_sets": sampling_sets}

@app.post("/unbiased-area-estimates")
def unbiased_area_estimates(file: UploadFile = File(...), predicted_column: str = 'stratum_id', true_column: str = 'class_id'):
    file_location = tempfile.mktemp(".csv")
    
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Read the file and process it
    annotation = pd.read_csv(file_location)
    predicted = annotation[predicted_column]
    true = annotation[true_column]

    unbiased_area_estimates = unbiased_area_estimates(predicted, true)

    
    # Placeholder for processing annotations and computing metrics
    return {"metrics": "Metrics computed successfully", "unbiased_area_estimates": unbiased_area_estimates}
