import asyncio

from nicegui import ui


# Placeholder backend functions
def read_raster_classes(file_path):
    # Simulate reading raster classes
    import time

    time.sleep(2)  # Simulating a delay
    return {1: "Forest", 2: "Water", 3: "Urban"}


def process_samples():
    # Simulate processing samples
    import time

    for i in range(5):
        time.sleep(1)  # Simulating processing time
    return {1: 50, 2: 30, 3: 20}


# UI Components
raster_file = None
classes = {}
class_mapping = {}
selected_masks = []
target_epsg = ""
sampling_method = "Proportional"
target_standard_error = 5
users_accuracies = {}
sample_counts = {}
output_folder = ""


async def load_classes():
    loading.open()
    await asyncio.sleep(0.1)  # Ensure UI updates
    global classes, class_mapping
    classes = read_raster_classes(raster_file)
    class_mapping = {
        cls: cls for cls in classes.keys()
    }  # Default mapping to same class
    loading.close()
    update_class_selection()


def update_class_selection():
    with class_container:
        ui.clear()
        for cls, name in classes.items():
            class_mapping[cls] = ui.number(
                label=f"Merge {name} (Class {cls}) to:", value=cls
            ).bind_value()


def start_processing():
    progress.open()
    ui.notify("Processing samples...")
    asyncio.create_task(process_samples_async())


async def process_samples_async():
    global sample_counts
    await asyncio.sleep(0.1)
    sample_counts = process_samples()
    progress.close()
    update_sample_counts()


def update_sample_counts():
    with sample_container:
        ui.clear()
        for cls, count in sample_counts.items():
            sample_counts[cls] = ui.number(
                label=f"Samples for {classes[cls]}", value=count
            ).bind_value()


def submit():
    ui.notify(f"Sample set created at {output_folder}")


def set_raster_file(files):
    global raster_file
    raster_file = files[0] if files else None
    if raster_file:
        asyncio.create_task(load_classes())  # Load classes after selection


# UI Layout
ui.label("Sample Set Creator").classes("text-2xl font-bold")

with ui.card():
    ui.label("Step 1: Select Raster File")
    ui.upload(label="Choose Raster", on_upload=set_raster_file)


loading = ui.dialog()
with loading:
    ui.label("Loading classes...")
    ui.spinner()


def show_loading():
    loading.open()


def hide_loading():
    loading.close()


progress = ui.dialog()
with progress:
    ui.label("Processing...")
    ui.spinner()


def show_progress():
    progress.open()


def hide_progress():
    progress.close()


class_container = ui.column()

with ui.card():
    ui.label("Step 2: Select Masking Shapefiles")
    ui.button("Choose Mask Files", on_click=lambda: ui.open_files()).bind_value(
        lambda v: globals().update(selected_masks=v)
    )
    ui.input("Target EPSG", on_change=lambda v: globals().update(target_epsg=v))

with ui.card():
    ui.label("Step 3: Sampling Method")
    ui.radio(["Proportional", "Neyman"], value=sampling_method).bind_value(
        lambda v: globals().update(sampling_method=v)
    )
    ui.number("Target Standard Error", value=target_standard_error).bind_value(
        lambda v: globals().update(target_standard_error=v)
    )

with ui.card():
    ui.label("Step 4: Expected User's Accuracies")
    for cls in classes:
        users_accuracies[cls] = ui.number(
            label=f"Accuracy for {classes[cls]}", value=80
        ).bind_value()

ui.button("Start Processing", on_click=start_processing)

sample_container = ui.column()

with ui.card():
    ui.label("Step 5: Output Folder")
    ui.button("Choose Output Folder", on_click=lambda: ui.open_folder()).bind_value(
        lambda v: globals().update(output_folder=v)
    )
    ui.button("Submit", on_click=submit)

ui.run(title="Sample Set Creator")
